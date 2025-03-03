'''
First, remember to uncomment line 987-988 in ./mainldm/ldm/models/diffusion/ddpm.py and comment them after finish collecting.
''' 
import sys
sys.path.append("./mainldm")
sys.path.append("./mainddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')
print(sys.path)
import argparse
import os, gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import logging
import wandb
import numpy as np
import torch.distributed as dist

import torch
# torch.set_grad_enabled(False)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_trainer
from PIL import Image
from einops import rearrange
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from quant.utils import AttentionMap, AttentionMap_add, seed_everything, Fisher , AttentionMap_input_add
from quant.quant_model import QModel
from quant.quant_block import QuantResnetBlock
from quant.set_quantize_params import set_act_quantize_params, set_weight_quantize_params
from quant.recon_Qmodel import recon_Qmodel, skip_Model
from quant.quant_layer import QuantModule
logger = logging.getLogger(__name__)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("./mainldm/models/ldm/lsun_beds256/config.yaml")  
    model = load_model_from_config(config, "./mainldm/models/ldm/lsun_beds256/model.ckpt")
    return model


def get_calibration(model, args, device):
    logging.info("sample cali start......")

    sampler = DDIMSampler(model, slow_steps=args.interval_seq)
    model.model.reset_no_cache(no_cache=True)
    shape = [args.calib_batch,
            model.model.diffusion_model.in_channels,
            model.model.diffusion_model.image_size,
            model.model.diffusion_model.image_size]
    bs = shape[0]
    shape = shape[1:]
    hooks = []
    hooks.append(AttentionMap_add(model.model.diffusion_model.output_blocks[-2], interval_seq=args.interval_seq, end_t=args.ddim_steps))

    maps = []
    samples = []
    ts = []

    with torch.no_grad():
        for i in tqdm(range(int(args.calib_num_samples/args.calib_batch)), desc="Generating image samples for cali-data"):
            _, intermediates = sampler.sample(args.ddim_steps, batch_size=bs, shape=shape, eta=args.ddim_eta, verbose=False,)

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

    for t_sample in range(args.ddim_steps):
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


def get_interval_seq(model, args, device):
    if args.dps_steps:
        logging.info("get my steps......")
        batch_size = 32
        sampler = DDIMSampler(model, slow_steps=range(args.ddim_steps))
        model.model.reset_no_cache(no_cache=True)
        shape = [batch_size,
                model.model.diffusion_model.in_channels,
                model.model.diffusion_model.image_size,
                model.model.diffusion_model.image_size]
        bs = shape[0]
        shape = shape[1:]

        hooks = []
        hooks.append(AttentionMap_add(model.model.diffusion_model.output_blocks[-2], interval_seq=range(args.ddim_steps), end_t=args.ddim_steps))

        with torch.no_grad():
            _, intermediates = sampler.sample(args.ddim_steps, batch_size=bs, shape=shape, eta=args.ddim_eta, verbose=False,)

        import ldm.globalvar as globalvar   
        globalvar.removeInput()
        torch.cuda.empty_cache()

        feature_maps = hooks[0].out
        feature_maps = [maps.cuda() for maps in feature_maps]
        time_list = np.arange(args.ddim_steps)
        groups_num = args.ddim_steps/args.replicate_interval
        if groups_num - int(groups_num) > 0:
            groups_num = int(groups_num) + 1
        groups_num = int(groups_num)

        fisher = Fisher(samples=feature_maps, class_num=groups_num)
        # interval_seq = fisher.feature_to_interval_seq()
        interval_seq = fisher.feature_to_interval_seq_optimal(args.replicate_interval)
        logging.info(interval_seq)
        for hook in hooks:
            hook.remove()
    else:
        logging.info("get uniform steps......")
        interval_seq = list(range(0, args.ddim_steps, args.replicate_interval))
        logging.info(interval_seq)
    return interval_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_num_samples', type=int, default=256)
    parser.add_argument('--calib_batch', type=int, default=32)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument('--ddim_steps', default=100, type=int)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument('--seed', default=1234+9, type=int)
    parser.add_argument("--dps_steps", action='store_true', default=False)

    parser.add_argument("--replicate_interval", type=int, default=2)
    parser.add_argument("--nonuniform", action='store_true')
    parser.add_argument("--pow", type=float, default=1.5)
    args = parser.parse_args()
    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"

    seed_everything(args.seed)
    # torch.set_grad_enabled(False)
    device = torch.device("cuda", args.local_rank)

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
    logging.info(args)
    model = get_model()

    interval_seq = get_interval_seq(model=model, args=args, device=device)
    args.interval_seq = interval_seq
    all_cali_data, all_t, all_cali_t, all_cache = get_calibration(model=model, args=args, device=device)
    
    torch.save((interval_seq, all_cali_data, all_t, all_cali_t, all_cache), \
                "./calibration/bedroom{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))

    logging.info("sample cali finish!")