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
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_trainer
from PIL import Image
from einops import rearrange
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from quant.utils import AttentionMap, seed_everything, Fisher 
from quant.quant_model import QModel
from quant.quant_block import Change_LDM_model_attnblock
from quant.set_quantize_params import set_act_quantize_params, set_weight_quantize_params
from quant.recon_Qmodel import recon_Qmodel, skip_LDM_Model
from quant.quant_layer import QuantModule
logger = logging.getLogger(__name__)


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )
@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model, slow_steps=model.interval_seq)
    ddim.quant_sample = True
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates
@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0
                    ):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False):
    log = dict()
    shape = [batch_size,
            model.model.diffusion_model.model.in_channels,
            model.model.diffusion_model.model.image_size,
            model.model.diffusion_model.model.image_size]

    # with model.ema_scope("Plotting"):
    with torch.no_grad():
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                            make_prog_row=True)
        elif dpm:
            logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
            sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)
        t1 = time.time()
        x_sample = model.decode_first_stage(sample)
    torch.cuda.empty_cache()
    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    # logger.info(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, dpm=False):

    tstart = time.time()
    n_saved = 0
    if model.cond_stage_model is None:
        all_images = []
        print(f"Running unconditional sampling for {n_samples} samples")
        with torch.no_grad():
            for _ in tqdm(range(n_samples // batch_size), desc="Sampling Batches (unconditional)"):
                logs = make_convolutional_sample(model, batch_size=batch_size,
                                                vanilla=vanilla, custom_steps=custom_steps,
                                                eta=eta, dpm=dpm)
                n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
                torch.cuda.empty_cache()

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


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

def block_train_w(q_unet, args, kwargs, cali_data, t, cali_t, cache):
    recon_qnn = recon_Qmodel(args, q_unet, kwargs)

    q_unet.block_count = 0
    '''weight'''
    kwargs['cali_data'] = (cali_data, t, cache)
    kwargs['cali_t'] = cali_t
    kwargs['cond'] = False
    recon_qnn.kwargs = kwargs
    recon_qnn.down_name = None
    del (cali_data, t, cache)
    gc.collect()
    q_unet.set_steps_state(is_mix_steps=True)
    q_unet = recon_qnn.recon()
    q_unet.set_steps_state(is_mix_steps=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sample_batch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument('--ddim_steps', type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1234+9)

    parser.add_argument("--replicate_interval", type=int, default=2)
    parser.add_argument("--sm_abit",type=int, default=8)
    parser.add_argument("--quant_act", action="store_true", default=True)
    parser.add_argument("--weight_bit",type=int, default=8)
    parser.add_argument("--act_bit",type=int,default=8)
    parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["qdiff"])
    parser.add_argument("--lr_w",type=float,default=5e-7)
    parser.add_argument("--lr_a", type=float, default=1e-7)
    parser.add_argument("--lr_z",type=float,default=0)
    parser.add_argument("--lr_rw",type=float,default=1e-4)
    parser.add_argument("--split", action="store_true", default=True)
    parser.add_argument("--ptq", action="store_true", default=True)
    parser.add_argument("--dps_steps", action='store_true', default=False)
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
    interval_seq, all_cali_data, all_t, all_cali_t, all_cache = \
            torch.load("./calibration/bedroom{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))
    logger.info("./calibration/bedroom{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))
    logger.info("load calibration down!")
    args.interval_seq = interval_seq
    logger.info(f"The interval_seq: {args.interval_seq}")
    model = get_model()

    (a_list, b_list) = torch.load(f"./error_dec/bedroom/pre_cacheerr_abCov_interval{args.replicate_interval}_list.pth")
    model.model.diffusion_model.a_list = torch.stack(a_list)
    model.model.diffusion_model.b_list = torch.stack(b_list)
    model.model.diffusion_model.timesteps = args.ddim_steps

    if args.ptq:
        wq_params = {'n_bits': args.weight_bit, 'symmetric': False, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'symmetric': False, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 0.5, "num_timesteps": args.ddim_steps}
        q_unet = QModel(model.model.diffusion_model, args, wq_params=wq_params, aq_params=aq_params)
        q_unet.cuda()
        q_unet.eval()

        logger.info("Setting the first and the last layer to 8-bit")
        q_unet.set_first_last_layer_to_8bit()
        q_unet.set_quant_state(False, False)

        if args.split:
            q_unet.model.split_shortcut = True

        cali_data = torch.cat(all_cali_data)
        t = torch.cat(all_t)
        idx = torch.randperm(len(cali_data))[:32]
        cali_data = cali_data[idx]
        t = t[idx]

        set_weight_quantize_params(q_unet, cali_data=(cali_data, t))
        set_act_quantize_params(args.interval_seq, q_unet, all_cali_data, all_t, all_cache)

        pre_err_list = torch.load(f"./error_dec/bedroom/pre_quanterr_abCov_weight{args.weight_bit}_interval{args.replicate_interval}_list.pth")
        q_unet.model.output_blocks[-1][0].skip_connection.pre_err = pre_err_list
        pre_norm_err_list = torch.load(f"./error_dec/bedroom/pre_norm_quanterr_abCov_weight{args.weight_bit}_interval{args.replicate_interval}_list.pth")
        q_unet.model.output_blocks[-1][0].in_layers[2].pre_err = pre_norm_err_list

        q_unet.set_quant_state(True, True)
        setattr(model.model, 'diffusion_model', q_unet)

        if args.recon:
            Change_LDM_model_attnblock(q_unet, aq_params)
            skip_model = skip_LDM_Model(q_unet, model_type="bedroom")
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
                            lr_rw=args.lr_rw,
                            p=2.0,
                            weight=0.01,
                            b_range=(20,2), 
                            warmup=0.2,
                            batch_size=args.batch_size,
                            batch_size1=32,
                            input_prob=0.5,
                            recon_w=True,
                            recon_a=True,
                            keep_gpu=False,
                            interval_seq=args.interval_seq,
                            )
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)

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
            q_unet.model.save_cache = False  
            block_train_w(q_unet, args, kwargs, cali_data, t, cali_t, cache)
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)
            q_unet.model.save_cache = True
            setattr(model.model, 'diffusion_model', q_unet)
            model.model.reset_no_cache(no_cache=False)

    model.interval_seq = args.interval_seq
    model.model.reset_no_cache(no_cache=False)
    model.model.diffusion_model.model.time = 0
    imglogdir = "./error_dec/bedroom/image"

    logging.info("sampling...")
    model.interval_seq = args.interval_seq
    logging.info(model.interval_seq)
    seed_everything(args.seed)
    run(model, imglogdir, eta=args.ddim_eta, n_samples=args.num_samples, custom_steps=args.ddim_steps, batch_size=args.sample_batch)
