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
from quant.utils import AttentionMap, AttentionMap_add, AttentionMap_input_add, seed_everything, Fisher 
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
    config = OmegaConf.load("./mainldm/configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "./models/ldm/cin256/model.ckpt")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--calib_num_samples', type=int, default=256) #64 #256
    parser.add_argument('--calib_batch', type=int, default=32) #16
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument('--ddim_steps', default=250, type=int)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument('--seed', default=1234+9, type=int)
    parser.add_argument("--dps_steps", action='store_true', default=True)

    parser.add_argument("--replicate_interval", type=int, default=20)
    parser.add_argument("--nonuniform", action='store_true')
    parser.add_argument("--pow", type=float, default=1.5)
    args = parser.parse_args()
    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"

    print(args)
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

    model = get_model()

    (interval_seq, all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2) = \
        torch.load("./calibration/imageNet{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))
    del (all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2)
    logging.info(interval_seq)

    batch_size = 32
    uc = model.get_learned_conditioning(
        {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
        )
    xc = torch.randint(0, args.num_classes, (batch_size,)).to(model.device)
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    shape = [3, 64, 64]
    sampler = DDIMSampler(model)
    model.model.reset_no_cache(no_cache=True)
    hooks = []
    hooks.append(AttentionMap_add(model.model.diffusion_model.output_blocks[-2], interval_seq=range(args.ddim_steps), end_t=args.ddim_steps))

    logging.info("sample predadd start!")
    with torch.no_grad():
        _, intermediates = sampler.sample(S=args.ddim_steps,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=args.scale,
                                        unconditional_conditioning=uc,
                                        eta=args.ddim_eta,
                                        )
    
    feature_maps = hooks[0].out
    torch.save(feature_maps, "./calibration/feature_maps.pt")

    logging.info("sample predadd finish!")