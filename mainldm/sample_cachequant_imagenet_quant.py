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
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sample_batch', type=int, default=50)
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
    parser.add_argument("--lr_a", type=float, default=1e-4)
    parser.add_argument("--lr_z",type=float,default=1e-1)
    parser.add_argument("--lr_rw",type=float,default=1e-2)
    parser.add_argument("--split", action="store_true", default=True)
    parser.add_argument("--ptq", action="store_true", default=True)
    parser.add_argument("--dps_steps", action='store_true', default=True)
    parser.add_argument("--recon", action="store_true", default=False)

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
        idx = torch.randperm(len(cali_data))[:8]
        cali_data = cali_data[idx]
        t = t[idx]
        context = context[idx]

        set_weight_quantize_params_cond(q_unet, cali_data=(cali_data, t, context))
        set_act_quantize_params_cond(args.interval_seq, q_unet, all_cali_data, all_t, all_cond, all_uncond, all_cache1, all_cache2, cond_type="imagnet")

        pre_err_list = torch.load(f"./error_dec/imagenet/pre_quanterr_abCov_weight{args.weight_bit}_interval{args.replicate_interval}_list.pth")
        q_unet.model.output_blocks[-1][0].skip_connection.pre_err = pre_err_list
        pre_norm_err_list = torch.load(f"./error_dec/imagenet/pre_norm_quanterr_abCov_weight{args.weight_bit}_interval{args.replicate_interval}_list.pth")
        q_unet.model.output_blocks[-1][0].in_layers[2].pre_err = pre_norm_err_list

        q_unet.set_quant_state(True, True)
        setattr(model.model, 'diffusion_model', q_unet)

        '''block-wise training For other layers'''
        if args.recon:
            Change_LDM_model_SpatialTransformer(q_unet, aq_params)
            skip_model = skip_LDM_Model(q_unet, model_type="imagenet")
            q_unet = skip_model.set_skip()
            kwargs = dict(iters=3000,
                            act_quant=True, 
                            weight_quant=True, 
                            asym=True,
                            opt_mode='mse', 
                            lr_z=args.lr_z,
                            lr_a=args.lr_a,
                            lr_w=args.lr_w,
                            lr_rw=args.lr_rw,
                            p=2.0,
                            weight=0.01,
                            b_range=(20,2), 
                            warmup=0.2,
                            batch_size=args.batch_size,
                            batch_size1=32,
                            input_prob=1.0,
                            recon_w=True,
                            recon_a=True,
                            keep_gpu=False,
                            interval_seq=args.interval_seq,
                            )
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)

            all_cali_data = torch.cat(all_cali_data)
            all_t = torch.cat(all_t)
            all_cond = torch.cat(all_cond)
            all_uncond = torch.cat(all_uncond)
            all_cali_t = torch.cat(all_cali_t)
            all_cache1 = torch.cat(all_cache1)
            all_cache2 = torch.cat(all_cache2)
            idx = torch.randperm(len(all_cali_data))[:1024]
            cali_data = all_cali_data[idx].detach()
            t = all_t[idx].detach()
            cond = all_cond[idx].detach()
            uncond = all_uncond[idx].detach()
            cali_t = all_cali_t[idx].detach()
            cache1 = all_cache1[idx].detach()
            cache2 = all_cache2[idx].detach()
            del (all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2)
            gc.collect()
            q_unet.model.save_cache = False  
            block_train_w(q_unet, args, kwargs, cali_data, t, cond, uncond, cali_t, cache1, cache2)
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)
            q_unet.model.save_cache = True
            setattr(model.model, 'diffusion_model', q_unet)

    sampler = DDIMSampler(model, slow_steps=args.interval_seq)
    model.model.reset_no_cache(no_cache=False)
    model.model.diffusion_model.model.time = 0
    if args.ptq:
        sampler.quant_sample = True
    imglogdir = "./error_dec/imagenet/image"
    base_count = 0
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    logging.info("sampling...")
    seed_everything(args.seed)
    # model.first_stage_model.quantize.cpu()
    iterator = tqdm(range(1000), desc='DDIM Sampler')
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(args.sample_batch*[1000]).to(model.device)}
                )
            for i, class_num in enumerate(iterator):
                class_label = class_num
                xc = torch.tensor(args.sample_batch*[class_label])

                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                                conditioning=c,
                                                batch_size=args.sample_batch,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc, 
                                                eta=args.ddim_eta,
                                                replicate_interval=args.replicate_interval,
                                                nonuniform=args.nonuniform, pow=args.pow)

                # x_samples_ddim = model.decode_first_stage(samples_ddim.cpu())
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                # all_samples.append(x_samples_ddim)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                x_checked_image = x_samples_ddim
                # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                for x_sample in x_checked_image_torch:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(imglogdir, f"{base_count:05}.png"))
                    base_count += 1
                    if base_count == args.num_samples:
                        break
                if base_count == args.num_samples:
                    break
