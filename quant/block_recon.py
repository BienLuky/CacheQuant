import numpy as np
import torch
import random, gc
from quant.quant_layer import QuantModule, lp_loss
from quant.quant_model import Quant_Model
from quant.quant_block import BaseQuantBlock, QuantAttnBlock, QuantQKMatMul, QuantSMVMatMul, QuantAttentionBlock, QuantBasicTransformerBlock
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.data_utils import save_inp_oup_data
from quant.data_utils_cond import save_inp_oup_data_cond
import torch.nn.functional as F
import matplotlib.pyplot as plt

def block_reconstruction(model: Quant_Model, block: BaseQuantBlock, cali_data: torch.Tensor, cali_t: torch.Tensor, branch: int = 2, interval_seq: list = [],
                         batch_size: int = 32, batch_size1: int = 1024, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, b_range: tuple = (20, 2), p: float = 2.0,
                         warmup: float = 0.0, act_quant: bool = False, weight_quant: bool = False, lr_a: float = 4e-5, lr_w=1e-2, lr_z=1e-1,
                         input_prob: float = 1.0, keep_gpu: bool = True, 
                         recon_w: bool = False, recon_a: bool = False, 
                         cond: bool = False,
                         recon_rw: bool = False, lr_rw=1e-6,
                         ):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    """

    if block.skip_state == 2:
        recon_rw = True
        input_prob = 1.0
    else:
        return 0
    '''set state'''                                    
    block.set_quant_state(weight_quant, act_quant)
    round_mode = 'learned_hard_sigmoid'
    '''set quantizer'''
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para, zero_para, rw_para = [], [], [], []
    for module in block.modules():
        '''weight'''
        if isinstance(module, QuantModule):
            if module.split == 0:
                if isinstance(module.weight_quantizer, AdaRoundQuantizer)==False:
                    module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                weight_tensor=module.org_weight.data)
                if recon_w:                                        
                    module.weight_quantizer.soft_targets = True
                    w_para += [module.weight_quantizer.alpha]
                    zero_para += [module.weight_quantizer.zero_point]
                    # module.weight_quantizer.delta = torch.nn.Parameter(torch.tensor(module.weight_quantizer.delta))
                    # w_para += [module.weight_quantizer.delta]
                    # zero_para += [module.weight_quantizer.zero_point]
            else :
                if isinstance(module.weight_quantizer, AdaRoundQuantizer)==False:
                    module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                weight_tensor=module.org_weight.data[:, :module.split, ...])
                if isinstance(module.weight_quantizer_0, AdaRoundQuantizer)==False:
                    module.weight_quantizer_0 = AdaRoundQuantizer(uaq=module.weight_quantizer_0, round_mode=round_mode,
                                                                weight_tensor=module.org_weight.data[:, module.split:, ...])
                if recon_w: 
                    module.weight_quantizer.soft_targets = True
                    module.weight_quantizer_0.soft_targets = True
                    w_para += [module.weight_quantizer.alpha]
                    w_para += [module.weight_quantizer_0.alpha]
                    zero_para += [module.weight_quantizer.zero_point]
                    zero_para += [module.weight_quantizer_0.zero_point]
                    # module.weight_quantizer.delta = torch.nn.Parameter(torch.tensor(module.weight_quantizer.delta))
                    # module.weight_quantizer_0.delta = torch.nn.Parameter(torch.tensor(module.weight_quantizer_0.delta))
                    # w_para += [module.weight_quantizer.delta]
                    # w_para += [module.weight_quantizer_0.delta]
                    # zero_para += [module.weight_quantizer.zero_point]
                    # zero_para += [module.weight_quantizer_0.zero_point]
            if recon_rw:
                module.weight = torch.nn.Parameter(torch.tensor(module.weight))
                module.weight.requires_grad = True
                w_para += [module.weight]
                if module.bias is not None:
                    module.bias = torch.nn.Parameter(torch.tensor(module.bias))
                    module.bias.requires_grad = True
                    w_para += [module.bias]
                module.weight_quantizer.delta.requires_grad = True

        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if act_quant and isinstance(module, QuantAttnBlock):
                if recon_a:
                    a_para += [module.act_quantizer_q.delta]
                    a_para += [module.act_quantizer_k.delta]
                    a_para += [module.act_quantizer_v.delta]
                    a_para += [module.act_quantizer_w.delta]
                    zero_para += [module.act_quantizer_q.zero_point]
                    zero_para += [module.act_quantizer_k.zero_point]
                    zero_para += [module.act_quantizer_v.zero_point]
                    zero_para += [module.act_quantizer_w.zero_point]
                    module.act_quantizer_q.is_training = True
                    module.act_quantizer_k.is_training = True
                    module.act_quantizer_v.is_training = True
                    module.act_quantizer_w.is_training = True
            if act_quant and isinstance(module, QuantAttentionBlock):
                if recon_a:
                    a_para += [module.attention.qkv_matmul.act_quantizer_q.delta]
                    a_para += [module.attention.qkv_matmul.act_quantizer_k.delta]
                    a_para += [module.attention.smv_matmul.act_quantizer_v.delta]
                    a_para += [module.attention.smv_matmul.act_quantizer_w.delta]
                    zero_para += [module.attention.qkv_matmul.act_quantizer_q.zero_point]
                    zero_para += [module.attention.qkv_matmul.act_quantizer_k.zero_point]
                    zero_para += [module.attention.smv_matmul.act_quantizer_v.zero_point]
                    zero_para += [module.attention.smv_matmul.act_quantizer_w.zero_point]
                    module.attention.qkv_matmul.act_quantizer_q.is_training = True
                    module.attention.qkv_matmul.act_quantizer_k.is_training = True
                    module.attention.smv_matmul.act_quantizer_v.is_training = True
                    module.attention.smv_matmul.act_quantizer_w.is_training = True
            if act_quant and isinstance(module, QuantBasicTransformerBlock):    
                if recon_a:
                    a_para += [module.attn1.act_quantizer_q.delta]
                    a_para += [module.attn1.act_quantizer_k.delta]
                    a_para += [module.attn1.act_quantizer_v.delta]
                    a_para += [module.attn1.act_quantizer_w.delta]
                    a_para += [module.attn2.act_quantizer_q.delta]
                    a_para += [module.attn2.act_quantizer_k.delta]
                    a_para += [module.attn2.act_quantizer_v.delta]
                    a_para += [module.attn2.act_quantizer_w.delta]
                    zero_para += [module.attn1.act_quantizer_q.zero_point]
                    zero_para += [module.attn1.act_quantizer_k.zero_point]
                    zero_para += [module.attn1.act_quantizer_v.zero_point]
                    zero_para += [module.attn1.act_quantizer_w.zero_point]
                    zero_para += [module.attn2.act_quantizer_q.zero_point]
                    zero_para += [module.attn2.act_quantizer_k.zero_point]
                    zero_para += [module.attn2.act_quantizer_v.zero_point]
                    zero_para += [module.attn2.act_quantizer_w.zero_point]
                    module.attn1.act_quantizer_q.is_training = True
                    module.attn1.act_quantizer_k.is_training = True
                    module.attn1.act_quantizer_v.is_training = True
                    module.attn1.act_quantizer_w.is_training = True
                    module.attn2.act_quantizer_q.is_training = True
                    module.attn2.act_quantizer_k.is_training = True
                    module.attn2.act_quantizer_v.is_training = True
                    module.attn2.act_quantizer_w.is_training = True 
            if act_quant and module.act_quantizer.delta is not None:
                if module.split == 0:
                    if recon_a:
                        a_para += [module.act_quantizer.delta]
                        zero_para += [module.act_quantizer.zero_point]
                        module.act_quantizer.is_training = True
                else:
                    if recon_a:
                        a_para += [module.act_quantizer.delta]
                        a_para += [module.act_quantizer_0.delta]
                        zero_para += [module.act_quantizer.zero_point]
                        zero_para += [module.act_quantizer_0.zero_point]
                        module.act_quantizer.is_training = True
                        module.act_quantizer_0.is_training = True
    
    w_opt, a_opt, zero_opt, rw_opt = None, None, None, None
    a_scheduler, zero_scheduler, w_scheduler, rw_scheduler = None, None, None, None
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=lr_w)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_opt, T_max=iters, eta_min=0.)
    if len(a_para) != 0:
        avg_a_scales = torch.mean(torch.tensor([torch.mean(a_scale) for a_scale in a_para])).item()
        a_opt = torch.optim.Adam(a_para, lr=lr_a*avg_a_scales)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)
    if len(zero_para) != 0:
        zero_opt = torch.optim.Adam(zero_para, lr=lr_z)
        zero_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(zero_opt, T_max=iters, eta_min=0.)
    if len(rw_para) != 0:
        rw_opt = torch.optim.Adam(rw_para, lr=lr_rw)
        rw_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rw_opt, T_max=iters, eta_min=0.)

    # loss_mode = 'relaxation'
    loss_mode = 'none'
    rec_loss = opt_mode
    loss_func = torch.nn.MSELoss()

    '''get input and set scale'''
    # cut cali groups
    batch_cali = 64
    batch_cali_data = []
    batch_cali_t = []
    for i in range(int(cali_t.size(0)/batch_cali)):
        batch_cali_data.append([_[i * batch_cali : (i + 1) * batch_cali] for _ in cali_data])
        batch_cali_t.append(cali_t[i * batch_cali : (i + 1) * batch_cali])
    del (cali_data)
    gc.collect()
    # get hooks
    all_cached_inps_x = []
    all_cached_inps_time = []
    all_cached_syms_x = []
    all_cached_syms_time = []
    all_cached_outs = []
    all_cali_t = []
    for i in  range(len(batch_cali_data)):
        cali_data = batch_cali_data[i]
        cali_t_1 = batch_cali_t[i]
        if not cond:
            Resblock, cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, cali_t_1, branch, asym, act_quant=act_quant, weight_quant=weight_quant, 
                                                                    batch_size=batch_size1, input_prob=True, keep_gpu=keep_gpu)
        else:
            Resblock, cached_inps, cached_outs = save_inp_oup_data_cond(model, block, cali_data, cali_t_1, branch, asym, act_quant=act_quant, weight_quant=weight_quant, 
                                                                    batch_size=batch_size1, input_prob=True, keep_gpu=keep_gpu)
        batch_cali_data[i] = None
        batch_cali_t[i] = None
        del (cali_data)
        gc.collect()
        if not cond:
            all_cali_t.append(cali_t_1)
        else:
            all_cali_t_1 = []
            for j in range(int(batch_cali/batch_size1)):
                all_cali_t_1.append(torch.cat([cali_t_1[j * batch_size1 : (j + 1) * batch_size1]] * 2))
            all_cali_t.append(torch.cat(all_cali_t_1))
        if Resblock:
            all_cached_inps_x.append(cached_inps[0][0])
            all_cached_inps_time.append(cached_inps[0][1])
            all_cached_syms_x.append(cached_inps[1][0])
            all_cached_syms_time.append(cached_inps[1][1])
        else:
            all_cached_inps_x.append(cached_inps[0]) 
            all_cached_syms_x.append(cached_inps[1]) 
        all_cached_outs.append(cached_outs)
    del (batch_cali_data, cached_inps, cached_outs)
    gc.collect()
    cached_outs = torch.cat(all_cached_outs)
    del (all_cached_outs)
    gc.collect()
    if Resblock:
        cached_inps = [[torch.cat(all_cached_inps_x), torch.cat(all_cached_inps_time)]]
        del (all_cached_inps_x, all_cached_inps_time)
        gc.collect()
        cached_inps.append([torch.cat(all_cached_syms_x), torch.cat(all_cached_syms_time)])
        del (all_cached_syms_x, all_cached_syms_time)
        gc.collect()
    else:
        cached_inps = [torch.cat(all_cached_inps_x), torch.cat(all_cached_syms_x)]
        del (all_cached_inps_x, all_cached_syms_x)
        gc.collect()
    cali_t = torch.cat(all_cali_t)

    if block.skip_state == 1:
        indices = [index for index, element in enumerate(cali_t) if element in interval_seq]
        cached_outs = cached_outs[indices]
        cali_t = cali_t[indices]
        if Resblock:
            cached_inps = [[cached_inps[0][0][indices], cached_inps[0][1][indices]], [cached_inps[1][0][indices], cached_inps[1][1][indices]]]
        else:
            cached_inps = [cached_inps[0][indices], cached_inps[1][indices]]

    device = 'cuda'
    sz = cached_outs.size(0)
    model.block_count = model.block_count + 1
    out_loss_list = []
    for num_iter in range(iters):
        idx = torch.randperm(sz)[:batch_size]
        t = cali_t[idx].to(device)
        block.set_t(t=t)
        cur_out = cached_outs[idx].to(device)
        if Resblock:
            cur_inp, cur_sym = cached_inps[0][0][idx].to(device), cached_inps[1][0][idx].to(device)
            temb_cur_inp, temb_cur_sym = cached_inps[0][1][idx].to(device), cached_inps[1][1][idx].to(device)
            if input_prob <= 1.0:
                num_batch = cur_inp.size(0)
                sym_index = int(num_batch * (1-input_prob))
                if sym_index > 0:
                    indices = torch.randperm(num_batch)[:sym_index]
                    cur_inp[indices] = cur_sym[indices]
                    temb_cur_inp[indices] = temb_cur_sym[indices]
        else:
            cur_inp, cur_sym = cached_inps[0][idx].to(device), cached_inps[1][idx].to(device)
            if input_prob <= 1.0:
                num_batch = cur_inp.size(0)
                sym_index = int(num_batch * (1-input_prob))
                if sym_index > 0:
                    indices = torch.randperm(num_batch)[:sym_index]
                    cur_inp[indices] = cur_sym[indices]

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        if zero_opt:
            zero_opt.zero_grad()
        if rw_opt:
            rw_opt.zero_grad()

        if Resblock:
            out_quant = block(cur_inp, temb_cur_inp)
        else:
            out_quant = block(cur_inp)

        loss = loss_func(out_quant, cur_out)
        if len(w_para) != 0 or len(a_para) != 0:
            loss.backward()#retain_graph=True
        out_loss_list.append(loss.cpu().detach().numpy())

        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
        if zero_opt:
            zero_opt.step()
        if rw_opt:
            rw_opt.step()
        if w_scheduler:
            w_scheduler.step()
        if a_scheduler:
            a_scheduler.step()
        if zero_scheduler:
            zero_scheduler.step()
        if rw_scheduler:
            rw_scheduler.step()

        if len(a_para) != 0:
            new_a_lr = a_scheduler.optimizer.param_groups[0]['lr']
            a_opt = torch.optim.Adam(a_para, lr=new_a_lr)
        if len(zero_para) != 0:
            new_a_lr = zero_scheduler.optimizer.param_groups[0]['lr']
            zero_opt = torch.optim.Adam(zero_para, lr=new_a_lr)
    torch.cuda.empty_cache()

    for module in block.modules():
        if isinstance(module, QuantModule):
            '''weight '''
            if module.split == 0:
                module.weight_quantizer.soft_targets = False
                module.act_quantizer.is_training = False
            else:
                module.weight_quantizer.soft_targets = False
                module.weight_quantizer_0.soft_targets = False
                module.act_quantizer.is_training = False
                module.act_quantizer_0.is_training = False
        if isinstance(module, QuantAttnBlock):
            module.act_quantizer_q.is_training = False
            module.act_quantizer_k.is_training = False
            module.act_quantizer_v.is_training = False
            module.act_quantizer_w.is_training = False
        if isinstance(module, QuantAttentionBlock):
            module.attention.qkv_matmul.act_quantizer_q.is_training = False
            module.attention.qkv_matmul.act_quantizer_k.is_training = False
            module.attention.smv_matmul.act_quantizer_v.is_training = False
            module.attention.smv_matmul.act_quantizer_w.is_training = False
        if isinstance(module, QuantBasicTransformerBlock):
            module.attn1.act_quantizer_q.is_training = False
            module.attn1.act_quantizer_k.is_training = False
            module.attn1.act_quantizer_v.is_training = False
            module.attn1.act_quantizer_w.is_training = False
            module.attn2.act_quantizer_q.is_training = False
            module.attn2.act_quantizer_k.is_training = False
            module.attn2.act_quantizer_v.is_training = False
            module.attn2.act_quantizer_w.is_training = False 

class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.iters = max_count

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    if module.split == 0:
                        round_vals = module.weight_quantizer.get_soft_targets()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                    else:
                        round_vals1 = module.weight_quantizer.get_soft_targets()
                        round_loss += self.weight * (1 - ((round_vals1 - .5).abs() * 2).pow(b)).sum()
                        round_vals2 = module.weight_quantizer_0.get_soft_targets()
                        round_loss += self.weight * (1 - ((round_vals2 - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        # if self.count%50 == 0 or self.count == 1:
        # # if self.count == self.iters or self.count == 1:
        #     print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
        #           float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        # print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
        #         float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return rec_loss, round_loss
        # return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            # return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
