'''
First, remember to uncomment line 987-988 in ./mainldm/ldm/models/diffusion/ddpm.py and comment them after finish collecting.
'''
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
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler

from quant.coco_prompt import get_prompts, center_resize_image
from quant.utils import AttentionMap, AttentionMap_add, seed_everything, Fisher 
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


def get_calibration(model, args, device):
    logging.info("sample cali start......")
    uc = model.get_learned_conditioning(args.calib_num_samples * [""])
    prompts = args.list_prompts[:args.calib_num_samples]
    c = model.get_learned_conditioning(prompts)
    shape = [args.C, args.H // args.f, args.W // args.f]
    start_code = None
    if args.plms:
        sampler = PLMSSampler(model, slow_steps=args.interval_seq)
    model.model.reset_no_cache(no_cache=True)
    hooks = []
    hooks.append(AttentionMap_add(model.model.diffusion_model.output_blocks[-2], interval_seq=args.interval_seq, end_t=args.ddim_steps+1))

    maps1 = []
    maps2 = []
    samples = []
    ts = []
    conds = []
    unconds = []
    with torch.no_grad():
        for i in tqdm(range(int(args.calib_num_samples/args.batch_samples)), desc="Generating image samples for cali-data"):
            _, intermediates = sampler.sample(S=args.ddim_steps,
                                            conditioning=c[i*args.batch_samples:(i+1)*args.batch_samples],
                                            batch_size=args.batch_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=uc[i*args.batch_samples:(i+1)*args.batch_samples],
                                            eta=args.ddim_eta,
                                            x_T=start_code)
            import ldm.globalvar as globalvar   
            input_list = globalvar.getInputList()

            maps1.append([sample[:args.batch_samples].cpu() for sample in hooks[0].out])
            maps2.append([sample[args.batch_samples:].cpu() for sample in hooks[0].out])
            samples.append([sample[0][:args.batch_samples].cpu() for sample in input_list])                                    
            ts.append([sample[1][:args.batch_samples].cpu() for sample in input_list])
            conds.append([sample[2][args.batch_samples:].cpu() for sample in input_list])
            unconds.append([sample[2][:args.batch_samples].cpu() for sample in input_list])
            for hook in hooks:
                hook.removeInfo()
            globalvar.removeInput()
            torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

    all_maps1 = []
    all_maps2 = []
    all_samples = []
    all_ts = []
    all_conds = []
    all_unconds = []

    for t_sample in range(len(args.interval_seq)):
        t_one = torch.cat([sub[t_sample] for sub in maps1])
        all_maps1.append(t_one)
        t_one = torch.cat([sub[t_sample] for sub in maps2])
        all_maps2.append(t_one)

    for t_sample in range(args.ddim_steps+1):
        t_one = torch.cat([sub[t_sample] for sub in samples])
        all_samples.append(t_one)
        t_one = torch.cat([sub[t_sample] for sub in ts])
        all_ts.append(t_one)
        t_one = torch.cat([sub[t_sample] for sub in conds])
        all_conds.append(t_one)
        t_one = torch.cat([sub[t_sample] for sub in unconds])
        all_unconds.append(t_one)
    del(samples, ts, conds, unconds, maps1, maps2)
    gc.collect()
    torch.cuda.empty_cache()

    all_cali_data = []
    all_t = []
    all_cond = []
    all_uncond = []
    all_cali_t = []
    all_cache1 = []
    all_cache2 = []
    now_cache = 0
    for now_rt, sample_t in enumerate(all_samples):
        if now_rt not in args.interval_seq:
            idx = torch.randperm(sample_t.size(0))[:8]
        else:
            now_cache = args.interval_seq.index(now_rt)
            idx = torch.randperm(sample_t.size(0))[:32]

        all_cali_data.append(all_samples[now_rt][idx])
        all_t.append(all_ts[now_rt][idx])
        all_cond.append(all_conds[now_rt][idx])
        all_uncond.append(all_unconds[now_rt][idx])
        all_cali_t.append(torch.full_like(all_ts[now_rt][idx], now_rt-1).to(torch.int))
        all_cache1.append(all_maps1[now_cache][idx])
        all_cache2.append(all_maps2[now_cache][idx])
    all_cali_t[0] = torch.zeros_like(all_cali_t[0])
    del(all_samples, all_ts, all_conds, all_unconds, all_maps1, all_maps2)
    gc.collect()

    # combine the time 0 and time 1
    t_sample = torch.cat((all_cali_data[0], all_cali_data[1]))
    all_cali_data = [t_sample] + all_cali_data[2:]
    t_sample = torch.cat((all_t[0], all_t[1]))
    all_t = [t_sample] + all_t[2:]
    t_sample = torch.cat((all_cond[0], all_cond[1]))
    all_cond = [t_sample] + all_cond[2:]
    t_sample = torch.cat((all_uncond[0], all_uncond[1]))
    all_uncond = [t_sample] + all_uncond[2:]
    t_sample = torch.cat((all_cali_t[0], all_cali_t[1]))
    all_cali_t = [t_sample] + all_cali_t[2:]
    t_sample = torch.cat((all_cache1[0], all_cache1[1]))
    all_cache1 = [t_sample] + all_cache1[2:]
    t_sample = torch.cat((all_cache2[0], all_cache2[1]))
    all_cache2 = [t_sample] + all_cache2[2:]
    return all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2


def get_interval_seq(model, args, device):
    if args.dps_steps:
        logging.info("get dps steps......")
        batch_size = 8
        uc = model.get_learned_conditioning(batch_size * [""])
        prompts = args.list_prompts[:batch_size]
        c = model.get_learned_conditioning(prompts)
        shape = [args.C, args.H // args.f, args.W // args.f]
        start_code = None
        if args.plms:
            sampler = PLMSSampler(model)
        model.model.reset_no_cache(no_cache=True)
        hooks = []
        hooks.append(AttentionMap_add(model.model.diffusion_model.output_blocks[-2], interval_seq=range(args.ddim_steps+1), end_t=args.ddim_steps+1))

        with torch.no_grad():
            _, intermediates = sampler.sample(S=args.ddim_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=uc,
                                            eta=args.ddim_eta,
                                            x_T=start_code)
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
        interval_seq = list(range(1, args.ddim_steps+1, args.replicate_interval))
        interval_seq[0] = 0
        logging.info(interval_seq)
    return interval_seq


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

    parser.add_argument("--calib_num_samples",default=256, type=int)
    parser.add_argument("--batch_samples",default=4,)

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

    interval_seq = get_interval_seq(model=model, args=args, device=device)
    args.interval_seq = interval_seq
    all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2 = get_calibration(model=model, args=args, device=device)
    
    torch.save((interval_seq, all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2), \
                "./calibration/stable{}_cache{}_{}_{}.pth".format(args.ddim_steps, args.replicate_interval, benchmark, args.mode))

    logging.info("sample cali finish!")

