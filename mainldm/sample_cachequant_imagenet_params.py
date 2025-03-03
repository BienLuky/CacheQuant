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
from imwatermark import WatermarkEncoder
from PIL import Image
from einops import rearrange
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from quant.utils import AttentionMap, seed_everything, Fisher 
from quant.quant_model import QModel
from quant.quant_block import Change_LDM_model_SpatialTransformer
from quant.set_quantize_params import set_act_quantize_params_cond, set_weight_quantize_params_cond, set_act_quantize_params_cond_ptq
from quant.recon_Qmodel import recon_Qmodel, skip_LDM_Model
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
    config = OmegaConf.load("./mainldm/configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "./models/ldm/cin256/model.ckpt")
    return model


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--sample_batch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument('--ddim_steps', type=int, default=250)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1234+9)

    parser.add_argument("--replicate_interval", type=int, default=20)
    parser.add_argument("--sm_abit",type=int, default=8)
    parser.add_argument("--quant_act", action="store_true", default=True)
    parser.add_argument("--weight_bit",type=int,default=8)
    parser.add_argument("--act_bit",type=int,default=8)
    parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["qdiff"])
    parser.add_argument("--lr_w",type=float,default=5e-1)
    parser.add_argument("--lr_a", type=float, default=0)
    parser.add_argument("--lr_z",type=float,default=0)
    parser.add_argument("--lr_rw",type=float,default=5e-2)
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
    logger.info("load calibration...")
    logger.info("./calibration/imageNet{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))
    interval_seq, all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2 = \
            torch.load("./calibration/imageNet{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))
    logger.info("load calibration down!")
    args.interval_seq = interval_seq
    logger.info(f"The interval_seq: {args.interval_seq}")
    model = get_model()

    (a_list, b_list) = torch.load(f"./error_dec/imagenet/pre_cacheerr_abCov_interval{args.replicate_interval}_list.pth")
    model.model.diffusion_model.a_list = torch.stack(a_list)
    model.model.diffusion_model.b_list = torch.stack(b_list)
    model.model.diffusion_model.timesteps = args.ddim_steps

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
        idx = torch.randperm(len(cali_data))[:32]
        cali_data = cali_data[idx]
        t = t[idx]
        context = context[idx]

        set_weight_quantize_params_cond(q_unet, cali_data=(cali_data, t, context))
        set_act_quantize_params_cond(args.interval_seq, q_unet, all_cali_data, all_t, all_cond, all_uncond, all_cache1, all_cache2)

        torch.save((q_unet.model.output_blocks[-1][0].skip_connection.weight_quantizer.delta, q_unet.model.output_blocks[-1][0].skip_connection.weight_quantizer.zero_point), "./error_dec/imagenet/weight_quantizer_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].skip_connection.act_quantizer.delta, q_unet.model.output_blocks[-1][0].skip_connection.act_quantizer.zero_point), "./error_dec/imagenet/act_quantizer_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].skip_connection.org_weight, q_unet.model.output_blocks[-1][0].skip_connection.org_bias), "./error_dec/imagenet/weight_params.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[2].weight_quantizer.delta, q_unet.model.output_blocks[-1][0].in_layers[2].weight_quantizer.zero_point), "./error_dec/imagenet/weight_quantizer_norm_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[2].act_quantizer.delta, q_unet.model.output_blocks[-1][0].in_layers[2].act_quantizer.zero_point), "./error_dec/imagenet/act_quantizer_norm_params_aftercacheadd.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[2].org_weight, q_unet.model.output_blocks[-1][0].in_layers[2].org_bias), "./error_dec/imagenet/weight_norm_params.pth")
        torch.save((q_unet.model.output_blocks[-1][0].in_layers[0].weight, q_unet.model.output_blocks[-1][0].in_layers[0].bias), "./error_dec/imagenet/groupnorm_norm_params.pth")

    logging.info(f"sampling quant int{args.weight_bit} params finish!")