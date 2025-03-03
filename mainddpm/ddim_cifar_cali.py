'''
First, remember to uncomment line 23-34 in ./mainddpm/ddpm/functions/denoising.py and comment them after finish collecting.
''' 
import sys
sys.path.append("./mainldm")
sys.path.append("./mainddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')
print(sys.path)
import argparse
import traceback
import shutil
import logging
import yaml
import random
import os, logging, gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from tqdm import tqdm

from ddpm.utils.tools import set_random_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
from quant.utils import AttentionMap, AttentionMap_add, seed_everything, Fisher 

import matplotlib.pyplot as plt
torch.set_printoptions(sci_mode=False)
logger = logging.getLogger(__name__)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_calibration(model, runner, args):
    logging.info("sample cali start......")
    seq, loop_fn = runner.obtain_generator_para()
    config = runner.config
    hooks = []
    select_layer, select_block = args.branch//3, args.branch%3
    if select_block == 2:
        up_select_block = 2
        up_select_layer = select_layer + 1
    else:
        up_select_layer = select_layer
        up_select_block = 1 -select_block
    if up_select_layer == 1:
        hooks.append(AttentionMap_add(model.up[up_select_layer].attn[up_select_block-1], interval_seq=args.interval_seq, end_t=args.timesteps))
    else:
        hooks.append(AttentionMap_add(model.up[up_select_layer].block[up_select_block-1], interval_seq=args.interval_seq, end_t=args.timesteps))

    maps = []
    samples = []
    ts = []

    shape = (args.calib_batch, 3, config.data.image_size, config.data.image_size)
    with torch.no_grad():
        for i in tqdm(range(int(args.calib_num_samples/args.calib_batch)), desc="Generating image samples for cali-data"):
            img = torch.randn(*shape, device=args.device) 
            _ = loop_fn(model=model, seq=seq, x=img, b=runner.betas, eta=runner.args.eta)

            import ldm.globalvar as globalvar   
            input_list = globalvar.getInputList()

            maps.append([sample.cpu() for sample in hooks[0].out])
            samples.append([sample[0].cpu() for sample in input_list])                                    
            ts.append([sample[1].cpu() for sample in input_list])

            for hook in hooks:
                hook.removeInfo()
            globalvar.removeInput()
            torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

    all_maps = []
    all_samples = []
    all_ts = []

    for t_sample in range(len(args.interval_seq)):
        t_one = torch.cat([sub[t_sample] for sub in maps])
        all_maps.append(t_one)

    for t_sample in range(args.timesteps):
        t_one = torch.cat([sub[t_sample] for sub in samples])
        all_samples.append(t_one)
        t_one = torch.cat([sub[t_sample] for sub in ts])
        all_ts.append(t_one)

    del(samples, ts, maps)
    gc.collect()
    torch.cuda.empty_cache()

    all_cali_data = []
    all_t = []
    all_cali_t = []
    all_cache = []
    now_cache = 0
    for now_rt, sample_t in enumerate(all_samples):
        if now_rt not in args.interval_seq:
            idx = torch.randperm(sample_t.size(0))[:32]
        else:
            now_cache = args.interval_seq.index(now_rt)
            idx = torch.randperm(sample_t.size(0))[:64]

        all_cali_data.append(all_samples[now_rt][idx])
        all_t.append(all_ts[now_rt][idx])
        all_cali_t.append(torch.full_like(all_ts[now_rt][idx], now_rt).to(torch.int))
        all_cache.append(all_maps[now_cache][idx])
    del(all_samples, all_ts, all_maps)
    gc.collect()
    return all_cali_data, all_t, all_cali_t, all_cache


def get_interval_seq(model, runner, args):
    seq, loop_fn = runner.obtain_generator_para()
    config = runner.config

    if args.dps_steps:
        logging.info("get dps steps......")
        hooks = []
        select_layer, select_block = args.branch//3, args.branch%3
        if select_block == 2:
            up_select_block = 2
            up_select_layer = select_layer + 1
        else:
            up_select_layer = select_layer
            up_select_block = 1 -select_block
        if up_select_layer == 1:
            hooks.append(AttentionMap_add(model.up[up_select_layer].attn[up_select_block-1], interval_seq=range(args.timesteps), end_t=args.timesteps))
        else:
            hooks.append(AttentionMap_add(model.up[up_select_layer].block[up_select_block-1], interval_seq=range(args.timesteps), end_t=args.timesteps))

        shape = (args.calib_batch, 3, config.data.image_size, config.data.image_size)
        img = torch.randn(*shape, device=args.device) 
        with torch.no_grad():
            _ = loop_fn(model=model, seq=seq, x=img, b=runner.betas, eta=runner.args.eta)

        import ldm.globalvar as globalvar   
        globalvar.removeInput()
        torch.cuda.empty_cache()

        feature_maps = hooks[0].out
        timesteps = len(feature_maps)
        feature_maps = [maps.cuda() for maps in feature_maps]
        time_list = np.arange(timesteps)
        groups_num = timesteps/args.cache_interval
        if groups_num - int(groups_num) > 0:
            groups_num = int(groups_num) + 1
        groups_num = int(groups_num)

        fisher = Fisher(samples=feature_maps, class_num=groups_num)
        # interval_seq = fisher.feature_to_interval_seq()
        interval_seq = fisher.feature_to_interval_seq_optimal(args.cache_interval)
        logging.info(interval_seq)
        for hook in hooks:
            hook.remove()
    else:
        logging.info("get uniform steps......")
        interval_seq = list(range(0, args.timesteps, args.cache_interval))
        logging.info(interval_seq)
    return interval_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default="./mainddpm/configs/cifar10.yml", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234+9, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp", type=str, default="deepcache", help="Path for saving running related data.")
    parser.add_argument("--image_folder", type=str, default="./error_dec/cifar/image", help="folder name for storing the sampled images")
    parser.add_argument("--fid", action="store_true", default=True)
    parser.add_argument("--interpolation", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument("--ni", action="store_true", default=True, help="No interaction. Suitable for Slurm Job launcher",)
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    parser.add_argument("--sample_type", type=str, default="generalized", help="sampling approach (generalized or ddpm_noisy)",)
    parser.add_argument("--skip_type", type=str, default="quad", help="skip according to (uniform or quadratic)",)
    parser.add_argument("--timesteps", type=int, default=100, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="eta used to control the variances of sigma",)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--select_step", type=int, default=None)
    parser.add_argument("--select_depth", type=int, default=None)
    parser.add_argument("--cache", action="store_true", default=True)
    parser.add_argument("--cache_interval", type=int, default=10,)
    parser.add_argument("--non_uniform", action="store_true", default=False)
    parser.add_argument("--pow", type=float, default=None,)
    parser.add_argument("--center", type=int, default=None,)
    parser.add_argument("--branch", type=int, default=2,)
    parser.add_argument('--calib_num_samples', type=int, default=512)
    parser.add_argument('--calib_batch', type=int, default=128)
    parser.add_argument("--dps_steps", action="store_true", default=False)
    args = parser.parse_args()
    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.select_step = args.select_step
    new_config.select_depth = args.select_depth
    torch.backends.cudnn.benchmark = True

    args, config = args, new_config
    accelerator = Accelerator()
    args.accelerator = accelerator
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("./run.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logging.info("start!")
    seed_everything(args.seed)

    from ddpm.runners.diffusion import Diffusion
    runner = Diffusion(args, config)
    model = runner.creat_model()

    interval_seq = get_interval_seq(model=model, runner=runner, args=args)
    args.interval_seq = interval_seq
    all_cali_data, all_t, all_cali_t, all_cache = get_calibration(model=model, runner=runner, args=args)
    
    torch.save((interval_seq, all_cali_data, all_t, all_cali_t, all_cache), \
                "./calibration/cifar{}_cache{}_{}.pth".format(args.timesteps, args.cache_interval, args.mode))

    logging.info("sample cali finish!")



