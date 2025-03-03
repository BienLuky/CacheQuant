import argparse, os, datetime, yaml, sys, gc
sys.path.append("./mainldm")
sys.path.append("./mainddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')
print(sys.path)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import cv2
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
import random
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

from quant.coco_prompt import get_prompts, center_resize_image
from quant.utils import AttentionMap, AttentionMap_add, seed_everything, Fisher , AttentionMap_input_add

logger = logging.getLogger(__name__)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("./mainldm/configs/stable-diffusion/v1-inference.yaml")  
    model = load_model_from_config(config, "./mainldm/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt")
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-file",type=str,help="if specified, load prompts from this file",default="./coco/annotations/captions_val2014.json")
    parser.add_argument("--logdir",type=str,default="none")
    parser.add_argument("--dataset",type=str,default="./coco/val2014_resize")
    parser.add_argument("--ddim_steps",type=int,default=50,)
    parser.add_argument("--plms",action='store_true',help="use plms sampling",default=True,)
    parser.add_argument("--ddim_eta",type=float,default=0.0,)
    parser.add_argument("--scale",type=float,default=7.5,help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
    parser.add_argument("--H",type=int,default=512,help="image height, in pixel space",)
    parser.add_argument("--W",type=int,default=512,help="image width, in pixel space",)
    parser.add_argument("--C",type=int,default=4,help="latent channels",)
    parser.add_argument("--f",type=int,default=8,help="downsampling factor",)
    parser.add_argument("--seed",type=int,default=1234+9,)
    # pecific configs
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--cond", action="store_true",default=True)
    parser.add_argument("--verbose", action="store_true",help="print out info like quantized model arch")

    parser.add_argument("--calib_num_samples",default=32,type=int)
    parser.add_argument("--batch_samples",default=8,)

    parser.add_argument("--replicate_interval", type=int, default=5)
    parser.add_argument("--dps_steps", action='store_true', default=True)

    parser.add_argument("--nonuniform", action='store_true', default=False)
    parser.add_argument("--pow", type=float, default=1.5)
    args = parser.parse_args()
    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"
    # benchmark = "coco"
    benchmark = "parti"
    print(args)
    seed_everything(args.seed)
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
    model.cuda()
    model.eval()

    if benchmark == "coco":
        # logging.info(f"reading prompts from {args.from_file}")
        # data = get_prompts(args.from_file)
        file_path = './mainldm/prompt/MSCOCO.txt'
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        data = [line.strip() for line in lines]
    elif benchmark == "parti":
        file_path = './mainldm/prompt/PartiPrompts.txt'
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        data = [line.strip() for line in lines] 
    args.list_prompts = random.sample(data, args.calib_num_samples)

    (interval_seq, all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2) = \
        torch.load("./calibration/stable{}_cache{}_{}_{}.pth".format(args.ddim_steps, args.replicate_interval, benchmark, args.mode))
    del (all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2)
    logging.info(interval_seq)

    logging.info("sample predadd start!")
    hooks = []
    feature_maps = []
    hooks.append(AttentionMap_add(model.model.diffusion_model.output_blocks[-2], interval_seq=range(args.ddim_steps+1), end_t=args.ddim_steps+1))
    
    shape = [args.C, args.H // args.f, args.W // args.f]
    start_code = None
    if args.plms:
        sampler = PLMSSampler(model)
    model.model.reset_no_cache(no_cache=True)
    with torch.no_grad():
        for i in range(int(args.calib_num_samples/args.batch_samples)):
            batch_size = args.batch_samples
            uc = model.get_learned_conditioning(batch_size * [""])
            prompts = args.list_prompts[i * batch_size: (i + 1) * batch_size]
            c = model.get_learned_conditioning(prompts)
            _, intermediates = sampler.sample(S=args.ddim_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=uc,
                                            eta=args.ddim_eta,
                                            x_T=start_code)
            
            feature_maps.append(hooks[0].out[:])
            for hook in hooks:
                hook.removeInfo()
    maps = []
    for i in range(len(feature_maps[0])):
        feature_map = torch.cat([feature_map[i] for feature_map in feature_maps])
        maps.append(feature_map)
    torch.save(maps, "./calibration/feature_maps.pt")

    logging.info("sample predadd finish!")

