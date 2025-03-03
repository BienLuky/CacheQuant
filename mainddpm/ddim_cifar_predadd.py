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
    parser.add_argument('--calib_batch', type=int, default=512)
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
        hooks.append(AttentionMap_add(model.up[up_select_layer].attn[up_select_block-1], interval_seq=range(args.timesteps), end_t=args.timesteps))
    else:
        hooks.append(AttentionMap_add(model.up[up_select_layer].block[up_select_block-1], interval_seq=range(args.timesteps), end_t=args.timesteps))

    shape = (args.calib_batch, 3, config.data.image_size, config.data.image_size)
    img = torch.randn(*shape, device=args.device) 
    with torch.no_grad():
        _ = loop_fn(model=model, seq=seq, x=img, b=runner.betas, eta=runner.args.eta)

    feature_maps = hooks[0].out
    torch.save(feature_maps, "./calibration/feature_maps.pt")

    logging.info("sample predadd finish!")



