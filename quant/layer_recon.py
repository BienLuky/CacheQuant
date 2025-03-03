import torch
import random, gc
from quant.quant_layer import QuantModule, StraightThrough, lp_loss
from quant.quant_model import Quant_Model
from quant.block_recon import LinearTempDecay
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.data_utils import save_inp_oup_data
from quant.data_utils_cond import save_inp_oup_data_cond
import torch.nn.functional as F
import matplotlib.pyplot as plt

def layer_reconstruction(model: Quant_Model, layer: QuantModule, cali_data: torch.Tensor, cali_t: torch.Tensor, branch: int = 2, interval_seq: list = [],
                         batch_size: int = 32, batch_size1: int = 1024, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, b_range: tuple = (20, 2), p: float = 2.0,
                         warmup: float = 0.0, act_quant: bool = False, weight_quant: bool = False, lr_a: float = 4e-5, lr_w=1e-2, lr_z=1e-1,
                         input_prob: float = 1.0, keep_gpu: bool = True, 
                         recon_w: bool = False, recon_a: bool = False, 
                         cond: bool = False,
                         recon_rw: bool = False, lr_rw=1e-6,
                         ):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
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
    :param keep_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """

    if layer.skip_state == 2:
        recon_rw = True
        input_prob = 1.0
    else:
        return 0
    '''set state'''                                    
    layer.set_quant_state(weight_quant, act_quant)
    round_mode = 'learned_hard_sigmoid'

    '''set quantizer'''
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para, zero_para, rw_para = [], [], [], []
    '''weight'''
    if layer.split == 0:
        if isinstance(layer.weight_quantizer, AdaRoundQuantizer)==False:
            layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                                        weight_tensor=layer.org_weight.data)
        if recon_w:                                        
            layer.weight_quantizer.soft_targets = True
            w_para += [layer.weight_quantizer.alpha]
            zero_para += [layer.weight_quantizer.zero_point]
            # layer.weight_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.weight_quantizer.delta))
            # w_para += [layer.weight_quantizer.delta]
            # zero_para += [layer.weight_quantizer.zero_point]
    else :
        if isinstance(layer.weight_quantizer, AdaRoundQuantizer)==False:
            layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                                        weight_tensor=layer.org_weight.data[:, :layer.split, ...])
        if isinstance(layer.weight_quantizer_0, AdaRoundQuantizer)==False:
            layer.weight_quantizer_0 = AdaRoundQuantizer(uaq=layer.weight_quantizer_0, round_mode=round_mode,
                                                        weight_tensor=layer.org_weight.data[:, layer.split:, ...])
        if recon_w: 
            layer.weight_quantizer.soft_targets = True
            layer.weight_quantizer_0.soft_targets = True
            w_para += [layer.weight_quantizer.alpha]
            w_para += [layer.weight_quantizer_0.alpha]
            zero_para += [layer.weight_quantizer.zero_point]
            zero_para += [layer.weight_quantizer_0.zero_point]
            # layer.weight_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.weight_quantizer.delta))
            # layer.weight_quantizer_0.delta = torch.nn.Parameter(torch.tensor(layer.weight_quantizer_0.delta))
            # w_para += [layer.weight_quantizer.delta]
            # w_para += [layer.weight_quantizer_0.delta]
            # zero_para += [layer.weight_quantizer.zero_point]
            # zero_para += [layer.weight_quantizer_0.zero_point]
    if recon_rw:
        layer.weight = torch.nn.Parameter(torch.tensor(layer.weight))
        layer.weight.requires_grad = True
        layer.bias = torch.nn.Parameter(torch.tensor(layer.bias))
        layer.bias.requires_grad = True
        rw_para += [layer.weight]
        rw_para += [layer.bias]
        layer.weight_quantizer.delta.requires_grad = True
    '''activation'''
    if act_quant and layer.act_quantizer.delta is not None:
        if layer.split == 0:
            layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
            layer.act_quantizer.zero_point = torch.nn.Parameter(torch.tensor(layer.act_quantizer.zero_point))
            if recon_a:
                a_para += [layer.act_quantizer.delta]
                zero_para += [layer.act_quantizer.zero_point]
                layer.act_quantizer.is_training = True
        else:
            layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
            layer.act_quantizer_0.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer_0.delta))
            layer.act_quantizer.zero_point = torch.nn.Parameter(torch.tensor(layer.act_quantizer.zero_point))
            layer.act_quantizer_0.zero_point = torch.nn.Parameter(torch.tensor(layer.act_quantizer_0.zero_point))
            if recon_a:
                a_para += [layer.act_quantizer.delta]
                a_para += [layer.act_quantizer_0.delta]
                zero_para += [layer.act_quantizer.zero_point]
                zero_para += [layer.act_quantizer_0.zero_point]
                layer.act_quantizer.is_training = True
                layer.act_quantizer_0.is_training = True

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
    all_cached_syms_x = []
    all_cached_outs = []
    all_cali_t = []
    for i in range(len(batch_cali_data)):
        cali_data = batch_cali_data[i]
        cali_t_1 = batch_cali_t[i]
        if not cond:
            Resblock, cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, cali_t_1, branch, asym, act_quant=act_quant, weight_quant=weight_quant, 
                                                                    batch_size=batch_size1, input_prob=True, keep_gpu=keep_gpu)
        else:
            Resblock, cached_inps, cached_outs = save_inp_oup_data_cond(model, layer, cali_data, cali_t_1, branch, asym, act_quant=act_quant, weight_quant=weight_quant, 
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
        all_cached_inps_x.append(cached_inps[0])
        all_cached_syms_x.append(cached_inps[1])
        all_cached_outs.append(cached_outs)

    del (batch_cali_data, cached_inps, cached_outs)
    gc.collect()
    cached_outs = torch.cat(all_cached_outs)
    del (all_cached_outs)
    gc.collect()
    cached_inps = [torch.cat(all_cached_inps_x), torch.cat(all_cached_syms_x)]
    del (all_cached_inps_x, all_cached_syms_x)
    gc.collect()
    cali_t = torch.cat(all_cali_t)

    if layer.skip_state == 1:
        indices = [index for index, element in enumerate(cali_t) if element in interval_seq]
        cached_outs = cached_outs[indices]
        cali_t = cali_t[indices]
        cached_inps = [cached_inps[0][indices], cached_inps[1][indices]]

    device = 'cuda'
    sz = cached_outs.size(0)
    model.block_count = model.block_count + 1
    out_loss_list = []
    for num_iter in range(iters):
        idx = torch.randperm(sz)[:batch_size]
        t = cali_t[idx].to(device)
        layer.set_t(t=t)
        cur_out = cached_outs[idx].to(device)
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

        out_quant = layer(cur_inp)

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

    if layer.split == 0:
        layer.weight_quantizer.soft_targets = False
        layer.act_quantizer.is_training = False
    else:
        layer.weight_quantizer.soft_targets = False
        layer.weight_quantizer_0.soft_targets = False
        layer.act_quantizer.is_training = False
        layer.act_quantizer_0.is_training = False

class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
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
            # round_vals = self.layer.weight_quantizer.get_soft_targets()
            # round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            if self.layer.split == 0:
                round_vals = self.layer.weight_quantizer.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                round_vals1 = self.layer.weight_quantizer.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals1 - .5).abs() * 2).pow(b)).sum()
                round_vals2 = self.layer.weight_quantizer_0.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals2 - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        # if self.count%50 == 0 or self.count == 1:
        # # if self.count == self.iters or self.count == 1:
            # print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
            #       float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        # print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
        #         float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return rec_loss, round_loss
        # return total_loss

