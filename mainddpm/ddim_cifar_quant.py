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
from quant.utils import AttentionMap, seed_everything, Fisher 
from quant.quant_model import QModel
from quant.set_quantize_params import set_act_quantize_params, set_weight_quantize_params
from quant.recon_Qmodel import recon_Qmodel, skip_Model

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


def block_train_w(q_unet, args, kwargs, cali_data, t, cali_t, cache):
    recon_qnn = recon_Qmodel(args, q_unet, kwargs)
    # recon_qnn = recon_lora_layer_Qmodel(args, q_unet, kwargs)

    q_unet.block_count = 0
    '''weight'''
    kwargs['cali_data'] = (cali_data, t, cache)
    kwargs['cali_t'] = cali_t
    kwargs['branch'] = args.branch
    recon_qnn.kwargs = kwargs
    recon_qnn.down_name = None
    del (cali_data, t, cache)
    gc.collect()
    q_unet.set_steps_state(is_mix_steps=True)
    q_unet = recon_qnn.recon()
    q_unet.set_steps_state(is_mix_steps=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default="./mainddpm/configs/cifar10.yml", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234+9, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp", type=str, default="deepcache", help="Path for saving running related data.")
    parser.add_argument("--image_folder", type=str, default="./error_dec/bedroom/image", help="folder name for storing the sampled images")
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
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sample_batch', type=int, default=500)

    parser.add_argument("--sm_abit",type=int, default=8,help="attn softmax activation bit")
    parser.add_argument("--quant_act", action="store_true", default=True, help="if to quantize activations when ptq==True")
    parser.add_argument("--weight_bit",type=int,default=8, help="int bit for weight quantization",)
    parser.add_argument("--act_bit",type=int,default=8, help="int bit for activation quantization",)
    parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["qdiff"], help="quantization mode to use")
    parser.add_argument("--lr_w",type=float, default=5e-1)
    parser.add_argument("--lr_a", type=float, default=1e-4)
    parser.add_argument("--lr_z",type=float, default=1e-1)
    parser.add_argument("--split", action="store_true", default=True, help="split shortcut connection into two parts")
    parser.add_argument("--ptq", action="store_true", default=True)
    parser.add_argument("--dps_steps", action="store_true", default=False)
    parser.add_argument("--recon", action="store_true", default=False)
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
    logger.info("load calibration...")
    interval_seq, all_cali_data, all_t, all_cali_t, all_cache \
        = torch.load("./calibration/cifar{}_cache{}_{}.pth".format(args.timesteps, args.cache_interval, args.mode))
    args.interval_seq = interval_seq
    logger.info(f"The interval_seq: {args.interval_seq}")

    from ddpm.runners.deepcache import Diffusion
    runner = Diffusion(args, config, interval_seq = args.interval_seq)
    model = runner.creat_model()

    (a_list, b_list) = torch.load(f"./error_dec/cifar/pre_cacheerr_abCov_interval{args.cache_interval}_list.pth")
    model.a_list = torch.stack(a_list)
    model.b_list = torch.stack(b_list)
    model.timesteps = args.timesteps

    if args.ptq:
        wq_params = {'n_bits': args.weight_bit, 'symmetric': False, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'symmetric': False, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 1.0, "num_timesteps": args.timesteps}
        q_unet = QModel(model, args, wq_params=wq_params, aq_params=aq_params)
        q_unet.cuda()
        q_unet.eval()

        print("Setting the first and the last layer to 8-bit")
        q_unet.set_first_last_layer_to_8bit()
        q_unet.set_quant_state(False, False)

        if args.split == True:
            q_unet.model.config.split_shortcut = True

        cali_data = torch.cat(all_cali_data)
        t = torch.cat(all_t)
        idx = torch.randperm(len(cali_data))[:32]
        cali_data = cali_data[idx]
        t = t[idx]

        set_weight_quantize_params(q_unet, cali_data=(cali_data, t))
        set_act_quantize_params(args.interval_seq, q_unet, all_cali_data, all_t, all_cache)

        pre_err_list = torch.load(f"./error_dec/cifar/pre_quanterr_abCov_weight{args.weight_bit}_interval{args.cache_interval}_list.pth")
        q_unet.model.up[1].block[2].nin_shortcut.pre_err = pre_err_list

        q_unet.set_quant_state(True, True)

        if args.recon:
            skip_model = skip_Model(q_unet)
            q_unet = skip_model.set_skip()
            # block-wise training For other layers
            kwargs = dict(iters=3000,
                            act_quant=True, 
                            weight_quant=True, 
                            asym=True,
                            opt_mode='mse', 
                            lr_z=args.lr_z,
                            lr_a=args.lr_a,
                            lr_w=args.lr_w,
                            p=2.0,
                            weight=0.01,
                            b_range=(20,2), 
                            warmup=0.2,
                            batch_size=32,
                            batch_size1=64,
                            num_split=1,
                            input_prob=1.0,
                            recon_w=True,
                            recon_a=True,
                            keep_gpu=False,
                            interval_seq=args.interval_seq,
                            )
            all_cali_data = torch.cat(all_cali_data)
            all_t = torch.cat(all_t)
            all_cali_t = torch.cat(all_cali_t)
            all_cache = torch.cat(all_cache)
            idx = torch.randperm(len(all_cali_data))[:1024]
            cali_data = all_cali_data[idx].detach()
            t = all_t[idx].detach()
            cali_t = all_cali_t[idx].detach()
            cache = all_cache[idx].detach()
            del (all_cali_data, all_t, all_cali_t, all_cache)
            gc.collect()
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)
            block_train_w(q_unet, args, kwargs, cali_data, t, cali_t, cache)
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)
            
        seed_everything(args.seed)
        model.time = 0
        runner.sample_fid(q_unet, total_n_samples=args.num_samples)
    else:
        seed_everything(args.seed)
        runner.sample_fid(model, total_n_samples=args.num_samples)

    logging.info("sample cali finish!")


