import torch
import torch.nn.functional as F
from quant.quant_layer import QuantModule, Union
from quant.quant_model import Quant_Model
from quant.quant_block import BaseQuantBlock

def save_inp_oup_data_cond(model: Quant_Model, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor, t: torch.Tensor, branch: int = 2,
                      asym: bool = False, act_quant: bool = False, weight_quant: bool = False, 
                      batch_size: int = 32, input_prob: bool = False, keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param weight_quant: use weight_quant quantization
    :param act_quant: use act_quant quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, input_prob=input_prob, 
                                    act_quant=act_quant, weight_quant=weight_quant, branch=branch)
    cached_batches = []
    temb_cached_batches = []
    Resblock = False

    for i in range(int(cali_data[0].size(0) / batch_size)):
        batch = [_[i * batch_size : (i + 1) * batch_size] for _ in cali_data]
        x_in = torch.cat([batch[0]] * 2)
        t_in = torch.cat([batch[1]] * 2)
        c_in = torch.cat([batch[3], batch[2]])
        cache_in = torch.cat([batch[4], batch[5]])
        cali_t = torch.cat([t[i * batch_size : (i + 1) * batch_size]] * 2)
        if input_prob:
            resblock, cur_inp, cur_out, cur_sym = get_inp_out(
                [x_in, t_in, c_in, cache_in], cali_t
            )
            Resblock = resblock
            if Resblock:
                cached_batches.append((cur_inp[0].cpu(), cur_out.cpu(), cur_sym[0].cpu()))
                temb_cached_batches.append((cur_inp[1].cpu(), cur_sym[1].cpu()))
            else:
                cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_sym.cpu()))
        else:
            resblock, cur_inp, cur_out = get_inp_out(
                [x_in, t_in, c_in, cache_in], cali_t
            )
            Resblock = resblock
            if Resblock:
                cached_batches.append((cur_inp[0].cpu(), cur_out.cpu()))
                temb_cached_batches.append((cur_inp[1].cpu()))
            else:
                cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    if Resblock:
        temb_cached_inps = torch.cat([x[0] for x in temb_cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[2] for x in cached_batches])
        if Resblock:
            temb_cached_sym = torch.cat([x[1] for x in temb_cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        if Resblock:
            temb_cached_inps = temb_cached_inps.to(device)
        cached_outs = cached_outs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
            if Resblock:
                temb_cached_sym = temb_cached_sym.to(device)
    if input_prob:
        if Resblock:
            return Resblock, ([cached_inps, temb_cached_inps], [cached_sym, temb_cached_sym]), cached_outs
        else:
            return Resblock, (cached_inps, cached_sym), cached_outs
    if Resblock:
        return Resblock, ([cached_inps, temb_cached_inps]), cached_outs
    else:
        return Resblock, (cached_inps,), cached_outs


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(
        self,
        model: Quant_Model,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        input_prob: bool = False, 
        act_quant: bool = False, 
        weight_quant: bool = False, 
        asym: bool = False,
        branch: int = 32,
    ):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.weight_quant = weight_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)
        self.input_prob = input_prob
        self.branch = branch

    def __call__(self, model_input, t):
        self.model.eval()
        self.model.set_quant_state(False, False)
        self.model.set_t(t)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(*[_.to(self.device) for _ in model_input[:-1]], prv_f=None, branch=self.branch)
            except StopForwardException:
                pass
            if self.input_prob:
                # input_sym = self.data_saver.input_store[0].detach() 
                if len(self.data_saver.input_store) == 1:
                    input_sym = self.data_saver.input_store[0].detach() 
                else:
                    input_sym = (self.data_saver.input_store[0].detach(), self.data_saver.input_store[1].detach())
            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=self.weight_quant, act_quant=self.act_quant)
                try:
                    if self.layer.skip_state != 2:
                        _ = self.model(*[_.to(self.device) for _ in model_input[:-1]], prv_f=None, branch=self.branch)
                    else:
                        _ = self.model(*[_.to(self.device) for _ in model_input[:-1]], prv_f=model_input[-1].to(self.device), branch=self.branch)
                except StopForwardException:
                    pass

            self.data_saver.store_output = True
        handle.remove()

        resblock = False
        if len(self.data_saver.input_store) == 1:
            input_store = self.data_saver.input_store[0].detach() 
        else:
            input_store = (self.data_saver.input_store[0].detach(), self.data_saver.input_store[1].detach())
            resblock = True

        if self.input_prob:       
            return (
                resblock,
                input_store,
                self.data_saver.output_store.detach(),
                input_sym,
            )
        return (
            resblock,
            input_store,
            self.data_saver.output_store.detach(),
        )


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(
        self,
        model: Quant_Model,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        act_quant: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(
                    F.log_softmax(out_q, dim=1),
                    F.softmax(out_fp, dim=1),
                    reduction="batchmean",
                )
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(
    model: QuantModule,
    layer: Union[QuantModule, BaseQuantBlock],
    act_quant: bool = False,
):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break

class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        # self.ac_output = []

    def hook_fn(self, module, input, output):
        self.out = output
        self.feature = input
        # self.ac_output.append(output)

    def remove(self):
        self.hook.remove()