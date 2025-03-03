import logging
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
logger = logging.getLogger(__name__)


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def smooth_value(x: torch.Tensor):
    max_value = torch.max(x).cpu().item()
    min_value = torch.min(x).cpu().item()
    q1 = np.percentile(x.cpu().numpy(), 25)
    q2 = np.percentile(x.cpu().numpy(), 50)
    q3 = np.percentile(x.cpu().numpy(), 75)
    return min_value, q1, q2, q3, max_value


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for floor operation.
    """
    return (x.floor() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False, prob: float = 1.0, num_timesteps: int = 100):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        # self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.running_stat = False
        self.always_zero = always_zero
        if self.leaf_param:
            self.x_min, self.x_max = None, None
        # """for activation quantization"""
            self.running_min = None
            self.running_max = None
            self.running_scale = None
            self.running_zero = None

        """mse params"""
        # self.scale_method = scale_method
        self.one_side_dist = None
        self.num = 100
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        """do like dropout"""
        self.prob = prob
        self.is_training = False
        """DM para"""
        self.timesteps = num_timesteps
        self.time = 0
        self.t = None
        self.is_mix_steps = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited
        
    def set_time(self, time: int = 0):  # inited manually
        self.time = time

    def set_t(self, t):
        self.t = t

    def set_steps_state(self, is_mix_steps: bool = True):
        self.is_mix_steps = is_mix_steps

    def update_quantize_range(self, x_scale, x_zero):
        if self.running_scale is None:
            self.running_scale = x_scale
            self.running_zero = x_zero
        self.running_scale = 0.1 * x_scale + 0.9 * self.running_scale
        self.running_zero = 0.1 * x_zero + 0.9 * self.running_zero
        return self.running_scale, self.running_zero

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    # @profile
    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def get_scale_zeropoint(self, x):
        scale, zero_point = self.init_quantization_scale_2(x, self.channel_wise)
        return scale, zero_point

    def init_quantization_scale_channel(self, x: torch.Tensor):
        with torch.no_grad():
            scale, zero_point = self.get_scale_zeropoint(x)
            if self.leaf_param:
                scale, zero_point = self.update_quantize_range(scale, zero_point)
        return scale, zero_point

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.inited)
        assert(x.size(0) == len(self.t))

        x_min, _ = torch.min(x.reshape(len(self.t), -1), dim=1)
        x_max, _ = torch.max(x.reshape(len(self.t), -1), dim=1)
        self.x_min[self.t] = self.x_min[self.t] * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max[self.t] = self.x_max[self.t] * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            delta = torch.max(self.x_min[self.t].abs(), self.x_max[self.t].abs()) / self.n_levels
        else:
            delta = (self.x_max[self.t] - self.x_min[self.t]) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max[self.t] / (self.n_levels - 1)
        
        delta = torch.clamp(delta, min=1e-8)
        # if not self.sym:
        #     self.zero_point[self.t] = (-self.x_min[self.t] / delta).round() if not (self.sym or self.always_zero) else 0
        batch_zeropoint = torch.nn.Parameter((-self.x_min[self.t] / delta).round())
        batch_delta = torch.nn.Parameter(delta) 
        with torch.no_grad():
            self.zero_point[self.t] = batch_zeropoint.clone().detach()
            self.delta[self.t] = batch_delta.clone().detach()

    def act_dynamic_update(self, x: torch.Tensor):
        assert(self.inited)

        self.x_min[self.time] = x.data.min()
        self.x_max[self.time] = x.data.max()

        if self.sym:
            delta = torch.max(self.x_min[self.time].abs(), self.x_max[self.time].abs()) / self.n_levels
        else:
            delta = (self.x_max[self.time] - self.x_min[self.time]) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max[self.time] / (self.n_levels - 1)
        
        delta = torch.clamp(delta, min=1e-8)
        # if not self.sym:
        #     self.zero_point[self.time] = (-self.x_min[self.time] / delta).round() if not (self.sym or self.always_zero) else 0
        batch_zeropoint = torch.nn.Parameter((-self.x_min[self.time] / delta).round())
        batch_delta = torch.nn.Parameter(delta) 
        with torch.no_grad():
            self.zero_point[self.time] = batch_zeropoint.clone().detach()
            self.delta[self.time] = batch_delta.clone().detach()

    def init_quantization_scale_1(
        self, x_clone: torch.Tensor, channel_wise: bool = False
    ):
        if channel_wise:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                if self.scale_method != 'channel_time' and self.channel_wise is False:
                    delta, zero_point = self.init_quantization_scale_1(x, self.channel_wise)
                elif self.scale_method != 'channel_time' and self.channel_wise:
                    delta, zero_point = self.init_quantization_scale_for_activation(x, self.channel_wise)
                else:
                    raise NotImplementedError
                    
                if self.delta == None:
                    self.delta = torch.nn.Parameter(torch.stack([delta] * self.timesteps, dim=0))
                    self.zero_point = torch.nn.Parameter(torch.stack([zero_point] * self.timesteps, dim=0))
                else:
                    self.delta[self.time] = delta
                    self.zero_point[self.time] = zero_point
            else:
                delta, zero_point = self.init_quantization_scale_1(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                self.zero_point = torch.nn.Parameter(zero_point)
        # start quantization
        if self.leaf_param:
            if self.is_mix_steps:
                if len(x.shape) == 3:
                    assert x.size(0) % len(self.t)==0
                    n_heads = x.size(0) // len(self.t)
                else:
                    assert x.size(0) == len(self.t)
                if self.channel_wise:
                    if len(x.shape) == 3:
                        cur_delta = self.delta[self.t].squeeze(1)
                        zero_point = self.zero_point[self.t].squeeze(1)
                        cur_delta = torch.repeat_interleave(cur_delta, repeats=n_heads, dim=0)
                        zero_point = torch.repeat_interleave(zero_point, repeats=n_heads, dim=0)
                    else:
                        cur_delta = self.delta[self.t].squeeze(1)
                        zero_point = self.zero_point[self.t].squeeze(1)
                else:
                    cur_delta = self.delta[self.t]
                    zero_point = self.zero_point[self.t]
                    for dim in range(x.dim()-1):
                        cur_delta = cur_delta.unsqueeze(1)
                        zero_point = zero_point.unsqueeze(1)
                    if len(x.shape) == 3:
                        cur_delta = torch.repeat_interleave(cur_delta, repeats=n_heads, dim=0)
                        zero_point = torch.repeat_interleave(zero_point, repeats=n_heads, dim=0)

                x_int = round_ste(x / cur_delta) + round_ste(zero_point)
                if self.sym:
                    x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
                else:
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant = (x_quant - round_ste(zero_point)) * cur_delta
            else:
                x_int = round_ste(x / self.delta[self.time]) + round_ste(self.zero_point[self.time])
                if self.sym:
                    x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
                else:
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant = (x_quant - round_ste(self.zero_point[self.time])) * self.delta[self.time]
        else:
            x_int = round_ste(x / self.delta) + round_ste(self.zero_point)
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - round_ste(self.zero_point)) * self.delta

        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def init_quantization_scale_2(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale_2(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)

        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if x_max == x_min:
                    delta = torch.tensor(x_max).type_as(x)
                    zero_point = torch.tensor(0 / delta).type_as(x)
                    return delta, zero_point
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                zero_point = torch.tensor(zero_point).type_as(x)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                if x_max == x_min:
                    delta = torch.tensor(x_max).type_as(x)
                    zero_point = torch.tensor(0 / delta).type_as(x)
                    return delta, zero_point
                best_score = 1e+10
                for i in range(0,80,2):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1) \
                            if not self.always_zero else new_max / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round() if not self.always_zero else 0
            else:
                raise NotImplementedError
        return delta, zero_point

    def init_quantization_scale_for_activation(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if len(x.shape) == 4:
                n_channels = x_clone.shape[1]
            elif len(x.shape) == 3: 
                n_channels = x_clone.shape[1] # channel wise quantization
            else:
                n_channels = x_clone.shape[1]

            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale_for_activation(x_clone[:, c, :], channel_wise=False)
                elif len(x.shape) == 4:
                    delta[c], zero_point[c] = self.init_quantization_scale_for_activation(x_clone[:,c,:,:], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale_for_activation(x_clone[:, c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(1, -1, 1, 1)
                zero_point = zero_point.view(1, -1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, -1, 1)
                zero_point = zero_point.view(1, -1, 1)
            else:
                raise NotImplementedError
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if "max" in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)

                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x.max().item(), x.min().item()))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)
                zero_point = torch.tensor(zero_point).type_as(x)
            else:
                x_clone = x.clone().detach()
                x_max = x_clone.max()
                x_min = x_clone.min()
                best_score = 1e+10
                self.x_min = x_min
                self.x_max = x_max
                # RepQ method
                for pct in [0.999, 0.9999, 0.99999]:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)   
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    # score = new_lp_loss(x_clone.view(x_clone.shape[0], -1), x_q.view(x_q.shape[0], -1))

                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        delta = torch.clamp(delta, min=1e-8)  # TODO: Added, examine effect
                        zero_point = (- new_min / delta).round()
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff'):
        super(QuantModule, self).__init__()
        # self.module = org_module
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params

        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
            self.module_str = 'conv2d'
            self.weight = org_module.weight
            self.org_weight = org_module.weight.data.clone()
            if org_module.bias is not None:
                self.bias = org_module.bias
                self.org_bias = org_module.bias.data.clone()
            else:
                self.bias = None
                self.org_bias = None
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
            self.module_str = 'conv1d'
            self.weight = org_module.weight
            self.org_weight = org_module.weight.data.clone()
            if org_module.bias is not None:
                self.bias = org_module.bias
                self.org_bias = org_module.bias.data.clone()
            else:
                self.bias = None
                self.org_bias = None
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            self.module_str = 'linear'
            self.weight = org_module.weight
            self.org_weight = org_module.weight.data.clone()
            if org_module.bias is not None:
                self.bias = org_module.bias
                self.org_bias = org_module.bias.data.clone()
            else:
                self.bias = None
                self.org_bias = None
        else:
            raise ValueError(
                "Unexpect quantization module"
            )

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.can_recon = True
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant
        self.disable_wei_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)
            self.timesteps = self.act_quantizer.timesteps
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.extra_repr = org_module.extra_repr
        self.skip_state = None
        self.skip_start = False
        self.skip_end = False
        self.time = 0

    def forward(self, input: torch.Tensor, split: int = 0):
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()

        if not self.disable_act_quant and self.use_act_quant:
            if self.split != 0:
                if self.act_quant_mode == 'qdiff':
                    input_0 = self.act_quantizer(input[:, :self.split, ...])
                    input_1 = self.act_quantizer_0(input[:, self.split:, ...])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                if self.act_quant_mode == 'qdiff':
                    input = self.act_quantizer(input)

        if not self.disable_wei_quant and self.use_weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if hasattr(self, "pre_err"):
            a = self.pre_err[0][self.act_quantizer.time].contiguous().view(self.pre_err[0][self.act_quantizer.time].size(0), 1, 1, 1)
            b = self.pre_err[1][self.act_quantizer.time]
            out1 = self.fwd_func(input_0, a * weight_0, b, **self.fwd_kwargs)
            out2 = self.fwd_func(input_1, weight_1, bias, **self.fwd_kwargs)
            out = out1 + out2
        else:
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_skip_state(self, skip_state):
        self.skip_state = skip_state

    def set_time(self, time: int = 0):  
        self.time = time

    def set_t(self, t):
        for name, module in self.named_modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    module.set_t(t)

    def set_split(self):
        self.weight_quantizer_0 = UniformAffineQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = UniformAffineQuantizer(**self.act_quant_params)
