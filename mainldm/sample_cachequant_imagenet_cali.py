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


def get_calibration(model, args, device):
    logging.info("sample cali start......")

    uc = model.get_learned_conditioning(
        {model.cond_stage_key: torch.tensor(args.calib_num_samples*[1000]).to(model.device)}
        )
    xc = torch.randint(0, args.num_classes, (args.calib_num_samples,)).to(model.device)
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    shape = [3, 64, 64]
    sampler = DDIMSampler(model, slow_steps=args.interval_seq)
    model.model.reset_no_cache(no_cache=True)
    hooks = []
    hooks.append(AttentionMap_add(model.model.diffusion_model.output_blocks[-2], interval_seq=args.interval_seq, end_t=args.ddim_steps))

    maps1 = []
    maps2 = []
    samples = []
    ts = []
    conds = []
    unconds = []

    with torch.no_grad():
        for i in tqdm(range(int(args.calib_num_samples/args.calib_batch)), desc="Generating image samples for cali-data"):
            _, intermediates = sampler.sample(S=args.ddim_steps,
                                            conditioning=c[i*args.calib_batch:(i+1)*args.calib_batch],
                                            batch_size=args.calib_batch,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=uc[i*args.calib_batch:(i+1)*args.calib_batch],
                                            eta=args.ddim_eta,
                                            )
            import ldm.globalvar as globalvar   
            input_list = globalvar.getInputList()

            maps1.append([sample[:args.calib_batch].cpu() for sample in hooks[0].out])
            maps2.append([sample[args.calib_batch:].cpu() for sample in hooks[0].out])
            samples.append([sample[0][:args.calib_batch].cpu() for sample in input_list])                                    
            ts.append([sample[1][:args.calib_batch].cpu() for sample in input_list])
            conds.append([sample[2][args.calib_batch:].cpu() for sample in input_list])
            unconds.append([sample[2][:args.calib_batch].cpu() for sample in input_list])
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

    for t_sample in range(args.ddim_steps):
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
            # idx = torch.randperm(sample_t.size(0))[:32]
        else:
            now_cache = args.interval_seq.index(now_rt)
            idx = torch.randperm(sample_t.size(0))[:32] 
            # idx = torch.randperm(sample_t.size(0))[:64] 

        all_cali_data.append(all_samples[now_rt][idx])
        all_t.append(all_ts[now_rt][idx])
        all_cond.append(all_conds[now_rt][idx])
        all_uncond.append(all_unconds[now_rt][idx])
        all_cali_t.append(torch.full_like(all_ts[now_rt][idx], now_rt).to(torch.int))
        all_cache1.append(all_maps1[now_cache][idx])
        all_cache2.append(all_maps2[now_cache][idx])
    del(all_samples, all_ts, all_conds, all_unconds, all_maps1, all_maps2)
    gc.collect()

    return all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2


def get_interval_seq(model, args, device):
    if args.dps_steps:
        logging.info("get my steps......")
        batch_size = 8
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

        start_time = time.time()
        fisher = Fisher(samples=feature_maps, class_num=groups_num)
        # interval_seq = fisher.feature_to_interval_seq()
        interval_seq = fisher.feature_to_interval_seq_optimal(args.replicate_interval)
        logging.info(interval_seq)
        end_time = time.time()
        cal_time = (end_time - start_time)
        logging.info(f'DPS time: {cal_time}')
        for hook in hooks:
            hook.remove()
    else:
        logging.info("get uniform steps......")
        interval_seq = list(range(0, args.ddim_steps, args.replicate_interval))
        logging.info(interval_seq)
    return interval_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--calib_num_samples', type=int, default=64) #64 #256
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

    interval_seq = get_interval_seq(model=model, args=args, device=device)
    args.interval_seq = interval_seq
    all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2 = get_calibration(model=model, args=args, device=device)
    
    torch.save((interval_seq, all_cali_data, all_t, all_cond, all_uncond, all_cali_t, all_cache1, all_cache2), \
                "./calibration/imageNet{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))

    logging.info("sample cali finish!")