import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import random

from quant.quant_block import get_specials, BaseQuantBlock
from quant.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock, QuantResBlock
from quant.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from ldm.modules.attention import BasicTransformerBlock


class Quant_Model(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.block_count = 0
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): 
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            # print(type(child_module))
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def set_skip_state(self):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_skip_state(m.skip_state)

    def set_act_quantize_init(self, act_init: bool = False):
        for m in self.model.modules():
            # if isinstance(m, (QuantModule, BaseQuantBlock, QuantAttnBlock)):
            if isinstance(m, QuantModule):
                if m.split == 0:
                    m.act_quantizer.set_inited(act_init)
                    m.act_quantizer.running_min = None
                    m.act_quantizer.running_scale = None
                else:
                    m.act_quantizer.set_inited(act_init)
                    m.act_quantizer.running_min = None
                    m.act_quantizer.running_scale = None
                    m.act_quantizer_0.set_inited(act_init)
                    m.act_quantizer_0.running_min = None
                    m.act_quantizer_0.running_scale = None
            if isinstance(m, QuantAttnBlock):
                m.act_quantizer_k.set_inited(act_init)
                m.act_quantizer_k.running_min = None
                m.act_quantizer_k.running_scale = None
                m.act_quantizer_q.set_inited(act_init)
                m.act_quantizer_q.running_min = None
                m.act_quantizer_q.running_scale = None
                m.act_quantizer_v.set_inited(act_init)
                m.act_quantizer_v.running_min = None
                m.act_quantizer_v.running_scale = None
                m.act_quantizer_w.set_inited(act_init)
                m.act_quantizer_w.running_min = None
                m.act_quantizer_w.running_scale = None
            if isinstance(m, QuantSMVMatMul):
                m.act_quantizer_v.set_inited(act_init)
                m.act_quantizer_v.running_min = None
                m.act_quantizer_v.running_scale = None
                m.act_quantizer_w.set_inited(act_init)
                m.act_quantizer_w.running_min = None
                m.act_quantizer_w.running_scale = None
            if isinstance(m, QuantQKMatMul):
                m.act_quantizer_k.set_inited(act_init)
                m.act_quantizer_k.running_min = None
                m.act_quantizer_k.running_scale = None
                m.act_quantizer_q.set_inited(act_init)
                m.act_quantizer_q.running_min = None
                m.act_quantizer_q.running_scale = None
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.act_quantizer_q.set_inited(act_init)
                m.attn1.act_quantizer_q.running_min = None
                m.attn1.act_quantizer_q.running_scale = None
                m.attn1.act_quantizer_k.set_inited(act_init)
                m.attn1.act_quantizer_k.running_min = None
                m.attn1.act_quantizer_k.running_scale = None
                m.attn1.act_quantizer_v.set_inited(act_init)
                m.attn1.act_quantizer_v.running_min = None
                m.attn1.act_quantizer_v.running_scale = None
                m.attn1.act_quantizer_w.set_inited(act_init)
                m.attn1.act_quantizer_w.running_min = None
                m.attn1.act_quantizer_w.running_scale = None
                m.attn2.act_quantizer_q.set_inited(act_init)
                m.attn2.act_quantizer_q.running_min = None
                m.attn2.act_quantizer_q.running_scale = None
                m.attn2.act_quantizer_k.set_inited(act_init)
                m.attn2.act_quantizer_k.running_min = None
                m.attn2.act_quantizer_k.running_scale = None
                m.attn2.act_quantizer_v.set_inited(act_init)
                m.attn2.act_quantizer_v.running_min = None
                m.attn2.act_quantizer_v.running_scale = None
                m.attn2.act_quantizer_w.set_inited(act_init)
                m.attn2.act_quantizer_w.running_min = None
                m.attn2.act_quantizer_w.running_scale = None

    def set_weight_quantize_init(self, weight_init: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                if m.split == 0:
                    m.weight_quantizer.set_inited(weight_init)
                else:
                    m.weight_quantizer.set_inited(weight_init)
                    m.weight_quantizer_0.set_inited(weight_init)

    def set_time(self, time: int = 0):
        for name, module in self.model.named_modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    module.set_time(time)
            if isinstance(module, QuantModule):
                module.set_time(time)

    def set_t(self, t):
        for name, module in self.model.named_modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    module.set_t(t)

    def set_steps_state(self, is_mix_steps: bool = True):
        for name, module in self.model.named_modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    module.set_steps_state(is_mix_steps)

    def forward(self, x, timesteps=None, context=None, prv_f=None, branch=None):
        return self.model(x, timesteps, context, prv_f=prv_f, branch=branch)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt
                
    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for name, module in self.model.named_modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[2].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        a_list[2].bitwidth_refactor(8)
        a_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

    def set_cosine_embedding_layer_to_32bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(32)
        a_list[0].bitwidth_refactor(32)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

def QModel(model, args, wq_params, aq_params):
    model.eval()
    q_unet = Quant_Model(model=model, weight_quant_params=wq_params, act_quant_params=aq_params, sm_abit=args.sm_abit)
    q_unet.cuda()
    q_unet.eval()
    return q_unet
