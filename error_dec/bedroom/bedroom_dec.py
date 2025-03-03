import argparse, os, glob, datetime, yaml, sys, gc
sys.path.append("./error_dec/bedroom")
sys.path.append(".")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def calculate_abCov(Y, X):
    Y_all = Y.permute(1, 0, 2, 3)
    X_all = X.permute(1, 0, 2, 3)
    a = torch.zeros(Y_all.size(0))
    b = torch.zeros(Y_all.size(0))
    for i in range(len(Y_all)):
        Y = Y_all[i]
        X = X_all[i]
        X_flat = X.contiguous().view(-1)
        Y_flat = Y.contiguous().view(-1)
        X_mean = torch.mean(X_flat)
        Y_mean = torch.mean(Y_flat)

        cov_XY = torch.mean((X_flat - X_mean) * (Y_flat - Y_mean))
        var_X = torch.var(X_flat)

        a[i] = cov_XY / var_X
        b[i] = Y_mean - a[i] * X_mean
    return a, b


def cache_pred_cacheerr_abCov_statistic(interval, mode):
    all_cache = torch.load("./calibration/feature_maps.pt")
    interval_seq, all_cali_data, all_t, all_cali_t, _ = \
            torch.load("./calibration/bedroom100_cache{}_{}.pth".format(interval, mode))
    del (all_cali_data, all_t, all_cali_t)
    print(list(interval_seq))
    Cova_list = []
    Covb_list = []
    with torch.no_grad():
        for i in range(len(all_cache)):
            if i in interval_seq:
                cache_c = all_cache[i]
                a = torch.ones(cache_c.size(1))
                b = torch.zeros(cache_c.size(1))
                Cova_list.append(a)
                Covb_list.append(b)
            else:
                cache_r = all_cache[i]
                a, b = calculate_abCov(cache_r, cache_c)
                Cova_list.append(a)
                Covb_list.append(b)

    Cova_list = [a.cuda() for a in Cova_list]
    Covb_list = [b.cuda() for b in Covb_list]
    torch.save((Cova_list, Covb_list), f"./error_dec/bedroom/pre_cacheerr_abCov_interval{interval}_list.pth")


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    return GroupNorm32(16, channels)


def cache_pred_quanterr_abCov_statistic(weight_bit, norm, interval, mode):
    from quant.quant_layer import UniformAffineQuantizer 
    import torch.nn.functional as F
    in_layers = nn.Sequential(
        normalization(224),
        nn.SiLU(),
    )
    all_cache = torch.load("./calibration/feature_maps.pt")
    all_cache_after_cacheadd = []
    (a_list, b_list) = torch.load(f"./error_dec/bedroom/pre_cacheerr_abCov_interval{interval}_list.pth")
    interval_seq, all_cali_data, all_t, all_cali_t, _ = \
            torch.load("./calibration/bedroom100_cache{}_{}.pth".format(interval, mode))
    del (all_cali_data, all_t, all_cali_t)
    print(list(interval_seq))
    with torch.no_grad():
        for i in range(len(all_cache)):
            if i in interval_seq:
                cache_c = all_cache[i]
                all_cache_after_cacheadd.append(cache_c)
            else:
                a = a_list[i].contiguous().view(1, a_list[i].size(0), 1, 1).cpu()
                b = b_list[i].contiguous().view(1, b_list[i].size(0), 1, 1).cpu()
                cache_after_cacheadd = a * cache_c + b
                all_cache_after_cacheadd.append(cache_after_cacheadd)
    if norm:
        save_path = f"./error_dec/bedroom/pre_norm_quanterr_abCov_weight{weight_bit}_interval{interval}_list.pth"
        group_weight, group_bias = torch.load("./error_dec/bedroom/groupnorm_norm_params.pth")
        in_layers[0].weight = torch.nn.Parameter(group_weight[:224].cpu())
        in_layers[0].bias = torch.nn.Parameter(group_bias[:224].cpu())
        with torch.no_grad():
            all_cache_after_cacheadd = [in_layers(cache) for cache in all_cache_after_cacheadd]
        act_delta, act_zero_point = torch.load("./error_dec/bedroom/act_quantizer_norm_params_aftercacheadd.pth", map_location=torch.device('cpu'))
        weight_delta, weight_zero_point = torch.load("./error_dec/bedroom/weight_quantizer_norm_params_aftercacheadd.pth", map_location=torch.device('cpu'))
        weight, bias = torch.load("./error_dec/bedroom/weight_norm_params.pth", map_location=torch.device('cpu'))
        fwd_kwargs = dict(stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    else:
        save_path = f"./error_dec/bedroom/pre_quanterr_abCov_weight{weight_bit}_interval{interval}_list.pth"
        act_delta, act_zero_point = torch.load("./error_dec/bedroom/act_quantizer_params_aftercacheadd.pth", map_location=torch.device('cpu'))
        weight_delta, weight_zero_point = torch.load("./error_dec/bedroom/weight_quantizer_params_aftercacheadd.pth", map_location=torch.device('cpu'))
        weight, bias = torch.load("./error_dec/bedroom/weight_params.pth", map_location=torch.device('cpu'))
        fwd_kwargs = dict(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    
    fwd_func = F.conv2d
    wq_params = {'n_bits': weight_bit, 'symmetric': False, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': 8, 'symmetric': False, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True, "prob": 1.0, "num_timesteps": 100}
    act_quantizer = UniformAffineQuantizer(**aq_params)
    weight_quantizer = UniformAffineQuantizer(**wq_params)
    act_quantizer.inited = True
    weight_quantizer.inited = True
    act_quantizer.delta = act_delta
    act_quantizer.zero_point = act_zero_point
    weight_quantizer.delta = weight_delta
    weight_quantizer.zero_point = weight_zero_point
    weight = weight[:, :224, ...]
    q_weight = weight_quantizer(weight)

    Cova_list = []
    Covb_list = []
    with torch.no_grad():
        for i in range(len(all_cache)):
            cache_c = all_cache_after_cacheadd[i]
            cache_r = all_cache[i]
            act_quantizer.time = i
            q_cache_c = act_quantizer(cache_c)

            # WrXr = fwd_func(cache_r, weight, bias, **fwd_kwargs)
            WrXc = fwd_func(cache_c, weight, bias, **fwd_kwargs)
            WqXcq = fwd_func(q_cache_c, q_weight, bias, **fwd_kwargs)

            a, b = calculate_abCov(WrXc, WqXcq)

            Cova_list.append(a)
            Covb_list.append(b)

    Cova_list = [a.cuda() for a in Cova_list]
    Covb_list = [b.cuda() for b in Covb_list]
    torch.save((Cova_list, Covb_list), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234+9, type=int)
    parser.add_argument('--error', default='quant', type=str)
    parser.add_argument('--weight_bit', default=8, type=int)
    parser.add_argument("--replicate_interval", type=int, default=2)
    parser.add_argument("--dps_steps", action='store_true', default=False)
    args = parser.parse_args()
    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"

    from quant.utils import seed_everything
    seed_everything(args.seed)

    if args.error == 'cache':
        cache_pred_cacheerr_abCov_statistic(interval=args.replicate_interval, mode=args.mode)
    elif args.error == 'quant':
        cache_pred_quanterr_abCov_statistic(weight_bit=args.weight_bit, norm=False, interval=args.replicate_interval, mode=args.mode)
        cache_pred_quanterr_abCov_statistic(weight_bit=args.weight_bit, norm=True, interval=args.replicate_interval, mode=args.mode)