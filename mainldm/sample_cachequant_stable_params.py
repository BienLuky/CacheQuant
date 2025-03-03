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
from quant.utils import AttentionMap, seed_everything, Fisher 
from quant.quant_model import QModel
from quant.quant_block import Change_LDM_model_SpatialTransformer
from quant.set_quantize_params import set_act_quantize_params_cond, set_weight_quantize_params_cond
from quant.recon_Qmodel import recon_Qmodel, skip_LDM_Model
from quant.quant_layer import QuantModule

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

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

def block_train_w(q_unet, args, kwargs, cali_data, t, cond, uncond, cali_t, cache1, cache2):
    recon_qnn = recon_Qmodel(args, q_unet, kwargs)
    q_unet.block_count = 0
    '''weight'''
    kwargs['cali_data'] = (cali_data, t, cond, uncond, cache1, cache2)
    kwargs['cali_t'] = cali_t
    kwargs['cond'] = True
    recon_qnn.kwargs = kwargs
    recon_qnn.down_name = None
    del (cali_data, t, cond, uncond, cache1, cache2)
    gc.collect()
    q_unet.set_steps_state(is_mix_steps=True)
    q_unet = recon_qnn.recon()
    q_unet.set_steps_state(is_mix_steps=False)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-file",type=str,help="if specified, load prompts from this file",default="./coco/annotations/captions_val2014.json")
    parser.add_argument("--logdir",type=str,default=" ./mainldm/coco/image")
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
    parser.add_argument("--precision",type=str,help="evaluate at this precision",choices=["full", "autocast"],default="autocast")

    parser.add_argument("--n_samples",type=int,default=5000,)
    parser.add_argument("--n_batch",type=int,default=4,)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument("--replicate_interval", type=int, default=5)
    parser.add_argument("--sm_abit",type=int, default=8)
    parser.add_argument("--quant_act", action="store_true", default=True)
    parser.add_argument("--weight_bit",type=int,default=8)
    parser.add_argument("--act_bit",type=int,default=8)
    parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["qdiff"])
    parser.add_argument("--lr_w",type=float,default=5e-6)
    parser.add_argument("--lr_a", type=float, default=1e-6)
    parser.add_argument("--lr_z",type=float,default=1e-1)
    parser.add_argument("--lr_rw",type=float,default=1e-3)
    parser.add_argument("--split", action="store_true", default=True)
    parser.add_argument("--ptq", action="store_true", default=True)
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
    logging.info(args)
    logger.info("load calibration...")
    interval_seq, all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2 = \
            torch.load("./calibration/stable{}_cache{}_{}_{}.pth".format(args.ddim_steps, args.replicate_interval, benchmark, args.mode))
    logger.info("./calibration/stable{}_cache{}_{}_{}.pth".format(args.ddim_steps, args.replicate_interval, benchmark, args.mode))
    logger.info("load calibration down!")
    args.interval_seq = interval_seq
    ori_interval_seq = [i-1 for i in args.interval_seq]
    ori_interval_seq[0] = 0
    logger.info(f"The interval_seq: {args.interval_seq}")
    logger.info(f"The ori_interval_seq: {ori_interval_seq}")
    model = get_model()
    model.cuda()
    model.eval()

    (a_list, b_list) = torch.load(f"./error_dec/stable/pre_cacheerr_abCov_interval{args.replicate_interval}_{benchmark}_list.pth")
    model.model.diffusion_model.a_list = torch.stack(a_list)
    model.model.diffusion_model.b_list = torch.stack(b_list)
    model.model.diffusion_model.timesteps = args.ddim_steps+1

    if args.ptq:
        wq_params = {'n_bits': args.weight_bit, 'symmetric': False, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'symmetric': False, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 1.0, "num_timesteps": args.ddim_steps}
        q_unet = QModel(model.model.diffusion_model, args, wq_params=wq_params, aq_params=aq_params)
        q_unet.cuda()
        q_unet.eval()

        logger.info("Setting the first and the last layer to 8-bit")
        q_unet.set_first_last_layer_to_8bit()
        q_unet.set_quant_state(False, False)

        if args.split:
            q_unet.model.split_shortcut = True
        
        cali_data = [torch.cat([cali_data] * 2) for cali_data in all_cali_data]
        t = [torch.cat([t] * 2) for t in all_t]
        context = [torch.cat([all_uncond[i], all_cond[i]]) for i in range(len(all_cond))]

        cali_data = torch.cat(cali_data)
        t = torch.cat(t)
        context = torch.cat(context)
        idx = torch.randperm(len(cali_data))[:2]
        cali_data = cali_data[idx]
        t = t[idx]
        context = context[idx]

        set_weight_quantize_params_cond(q_unet, cali_data=(cali_data, t, context))
        set_act_quantize_params_cond(ori_interval_seq, q_unet, all_cali_data, all_t, all_cond, all_uncond, all_cache1, all_cache2, batch_size=4, cond_type="stable")
        
        torch.save((q_unet.model.output_blocks[-1][0].skip_connection.weight_quantizer.delta, q_unet.model.output_blocks[-1][0].skip_connection.weight_quantizer.zero_point), "./error_dec/stable/weight_quantizer_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].skip_connection.act_quantizer.delta, q_unet.model.output_blocks[-1][0].skip_connection.act_quantizer.zero_point), "./error_dec/stable/act_quantizer_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].skip_connection.org_weight, q_unet.model.output_blocks[-1][0].skip_connection.org_bias), "./error_dec/stable/weight_params.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[2].weight_quantizer.delta, q_unet.model.output_blocks[-1][0].in_layers[2].weight_quantizer.zero_point), "./error_dec/stable/weight_quantizer_norm_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[2].act_quantizer.delta, q_unet.model.output_blocks[-1][0].in_layers[2].act_quantizer.zero_point), "./error_dec/stable/act_quantizer_norm_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[2].org_weight, q_unet.model.output_blocks[-1][0].in_layers[2].org_bias), "./error_dec/stable/weight_norm_params.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[0].weight, q_unet.model.output_blocks[-1][0].in_layers[0].bias), "./error_dec/stable/groupnorm_norm_params.pth")

    logging.info(f"sampling quant int{args.weight_bit} params finish!")