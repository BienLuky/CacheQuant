import logging
import torch
from quant.quant_layer import QuantModule
from quant.quant_block import BaseQuantBlock, QuantAttnBlock, QuantSMVMatMul, QuantQKMatMul, QuantBasicTransformerBlock
from quant.quant_model import Quant_Model
from typing import Union
from tqdm import tqdm
logger = logging.getLogger(__name__)


def set_act_quantize_params(
    interval_seq, 
    module: Union[Quant_Model, QuantModule, BaseQuantBlock],
    all_cali_data, 
    all_t,
    all_cache,
    batch_size: int = 256,
):
    logger.info("set_act_quantize_params")
    module.set_quant_state(False, True)

    """set or init step size and zero point in the activation quantizer"""
    module.set_act_quantize_init(act_init=False)
    with torch.no_grad():
        for time in tqdm(range(len(all_cali_data)), desc="Init scale_a"):
            batch_size = min(batch_size, all_cali_data[time].size(0))
            module.set_time(time)
            if time in interval_seq:
                for i in range(int(all_cali_data[time].size(0) / batch_size)):
                    cali_data = all_cali_data[time][i * batch_size : (i + 1) * batch_size]
                    t = all_t[time][i * batch_size : (i + 1) * batch_size]
                    module.model.time = time
                    module(cali_data.cuda(), t.cuda())
            else:
                for i in range(int(all_cali_data[time].size(0) / batch_size)):
                    cali_data = all_cali_data[time][i * batch_size : (i + 1) * batch_size]
                    t = all_t[time][i * batch_size : (i + 1) * batch_size]
                    cache = all_cache[time][i * batch_size : (i + 1) * batch_size]
                    module.model.time = time
                    module(cali_data.cuda(), t.cuda(), prv_f=cache.cuda())
            module.set_act_quantize_init(act_init=False)

    torch.cuda.empty_cache()
    module.set_act_quantize_init(act_init=True)


def set_weight_quantize_params(module, cali_data):
    logger.info("set_weight_quantize_params")

    module.set_quant_state(True, False)
    module.set_weight_quantize_init(weight_init=False)

    batch_size = 8
    with torch.no_grad():
        module(*[_[:batch_size].cuda() for _ in cali_data])
    torch.cuda.empty_cache()
    module.set_weight_quantize_init(weight_init=True)


def set_act_quantize_params_cond(
    interval_seq, 
    module: Union[Quant_Model, QuantModule, BaseQuantBlock],
    all_cali_data, 
    all_t,
    all_cond,
    all_uncond,
    all_cache1,
    all_cache2,
    batch_size: int = 16, #16
    cond_type: str = "imagenet",
):
    logger.info("set_act_quantize_params")
    module.set_quant_state(False, True)
    """set or init step size and zero point in the activation quantizer"""
    module.set_act_quantize_init(act_init=False)
    with torch.no_grad():
        for time in tqdm(range(len(all_cali_data)), desc="Init scale_a"):
            batch_size = min(batch_size, all_cali_data[time].size(0))
            module.set_time(time)
            if time in interval_seq:
                for i in range(int(all_cali_data[time].size(0) / batch_size)):
                # for i in range(int(batch_size / batch_size)):
                    cali_data = torch.cat([all_cali_data[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    t = torch.cat([all_t[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    context = torch.cat([all_uncond[time][i * batch_size : (i + 1) * batch_size], all_cond[time][i * batch_size : (i + 1) * batch_size]])
                    module.model.time = time
                    module(
                        cali_data.cuda(), t.cuda(), context.cuda()
                    )
            else:
                for i in range(int(all_cali_data[time].size(0) / batch_size)):
                    cali_data = torch.cat([all_cali_data[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    t = torch.cat([all_t[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    context = torch.cat([all_uncond[time][i * batch_size : (i + 1) * batch_size], all_cond[time][i * batch_size : (i + 1) * batch_size]])
                    cache = torch.cat([all_cache1[time][i * batch_size : (i + 1) * batch_size], all_cache2[time][i * batch_size : (i + 1) * batch_size]])
                    if cond_type == "stable":
                        module.model.time = time + 1
                    else:
                        module.model.time = time
                    module(
                        cali_data.cuda(), t.cuda(), context.cuda(), prv_f=cache.cuda()
                    )
            module.set_act_quantize_init(act_init=False)

    torch.cuda.empty_cache()
    module.set_act_quantize_init(act_init=True)


def set_weight_quantize_params_cond(module, cali_data):
    logger.info("set_weight_quantize_params")

    module.set_quant_state(True, False)
    module.set_weight_quantize_init(weight_init=False)

    batch_size = 32
    with torch.no_grad():
        module(*[_[:batch_size].cuda() for _ in cali_data])
    torch.cuda.empty_cache()
    module.set_weight_quantize_init(weight_init=True)


def set_act_quantize_params_cond_ptq(
    interval_seq, 
    module: Union[Quant_Model, QuantModule, BaseQuantBlock],
    all_cali_data, 
    all_t,
    all_cond,
    all_uncond,
    all_cache1,
    all_cache2,
    batch_size: int = 16,
):
    logger.info("cali_act_quantize_params")
    module.set_quant_state(True, True)
    module.set_ptq_cali(True)
    """set or init step size and zero point in the activation quantizer"""
    with torch.no_grad():
        for time in tqdm(range(len(all_cali_data)), desc="Init scale_a"):
            batch_size = min(batch_size, all_cali_data[time].size(0))
            module.set_time(time)
            if time in interval_seq:
                for i in range(int(all_cali_data[time].size(0) / batch_size)):
                    cali_data = torch.cat([all_cali_data[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    t = torch.cat([all_t[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    context = torch.cat([all_uncond[time][i * batch_size : (i + 1) * batch_size], all_cond[time][i * batch_size : (i + 1) * batch_size]])
                    module(
                        cali_data.cuda(), t.cuda(), context.cuda()
                    )
            else:
                for i in range(int(all_cali_data[time].size(0) / batch_size)):
                    cali_data = torch.cat([all_cali_data[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    t = torch.cat([all_t[time][i * batch_size : (i + 1) * batch_size]] * 2)
                    context = torch.cat([all_uncond[time][i * batch_size : (i + 1) * batch_size], all_cond[time][i * batch_size : (i + 1) * batch_size]])
                    cache = torch.cat([all_cache1[time][i * batch_size : (i + 1) * batch_size], all_cache2[time][i * batch_size : (i + 1) * batch_size]])
                    module(
                        cali_data.cuda(), t.cuda(), context.cuda(), prv_f=cache.cuda()
                    )
    module.set_ptq_cali(False)
    torch.cuda.empty_cache()
