import torch.nn as nn
from quant.quant_layer import QuantModule
from quant.quant_block import BaseQuantBlock, QuantResnetBlock, QuantAttnBlock
from quant.block_recon import block_reconstruction
from quant.layer_recon import layer_reconstruction
import logging
logger = logging.getLogger(__name__)


class recon_Qmodel():
    def __init__(self, args, qnn, kwargs, atten_layer = '1'):
        self.args = args
        self.model = qnn
        self.kwargs = kwargs
        self.down_name = None
        self.atten_layer = atten_layer

    def recon_model(self, module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in module.named_children():
            if self.down_name == None and name == 'down':
                self.down_name = 'down'
            if self.down_name == 'down' and name == self.atten_layer and isinstance(module, BaseQuantBlock) == 0:
                logger.info('reconstruction for down 1 modulelist')
                block_reconstruction(self.model, module.block[0], **self.kwargs)
                block_reconstruction(self.model, module.attn[0], **self.kwargs)
                block_reconstruction(self.model, module.block[1], **self.kwargs)
                block_reconstruction(self.model, module.attn[1], **self.kwargs)
                layer_reconstruction(self.model, module.downsample.conv, **self.kwargs)
                self.down_name = 'over'
            elif isinstance(module, QuantModule):
                if module.can_recon == False:
                    continue
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(self.model, module, **self.kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.can_recon == False:
                    continue
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(name))
                    block_reconstruction(self.model, module, **self.kwargs)
            elif name == 'up':
                self.recon_up_model(module)
            else:
                self.recon_model(module)

    def recon_up_model(self, module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for up_name, up_module in reversed(list(module.named_children())):
            if up_name == self.atten_layer:
                logger.info('reconstruction for up 1 modulelist')
                block_reconstruction(self.model, up_module.block[0], **self.kwargs)
                block_reconstruction(self.model, up_module.attn[0], **self.kwargs)
                block_reconstruction(self.model, up_module.block[1], **self.kwargs)
                block_reconstruction(self.model, up_module.attn[1], **self.kwargs)
                block_reconstruction(self.model, up_module.block[2], **self.kwargs)
                block_reconstruction(self.model, up_module.attn[2], **self.kwargs) 
                layer_reconstruction(self.model, up_module.upsample.conv, **self.kwargs)
            elif isinstance(up_module, QuantModule):
                if up_module.can_recon == False:
                    continue
                if up_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(up_name))
                    layer_reconstruction(self.model, up_module, **self.kwargs)
            elif isinstance(up_module, BaseQuantBlock):
                if up_module.can_recon == False:
                    continue
                if up_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(up_name))
                    block_reconstruction(self.model, up_module, **self.kwargs)
            else:
                self.recon_model(up_module)

    def recon(self):
        self.recon_model(self.model)
        return self.model


class recon_layer_Qmodel():
    def __init__(self, args, qnn, kwargs, atten_layer = '1'):
        self.args = args
        self.model = qnn
        self.kwargs = kwargs
        self.down_name = None
        self.atten_layer = atten_layer

    def recon_model(self, module: nn.Module):
        """
        layer reconstruction. 
        """
        for name, module in module.named_children():
            if self.down_name == None and name == 'down':
                self.down_name = 'down'
            if self.down_name == 'down' and name == self.atten_layer and isinstance(module, BaseQuantBlock) == 0:
                logger.info("reconstruction for down 1 modulelist")
                self.recon_block(module.block[0])
                self.recon_block(module.attn[0])
                self.recon_block(module.block[1])
                self.recon_block(module.attn[1])
                layer_reconstruction(self.model, module.downsample.conv, **self.kwargs)
                self.down_name = 'over'
            elif isinstance(module, QuantModule):
                if module.can_recon == False:
                    continue
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(self.model, module, **self.kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.can_recon == False:
                    continue
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(name))
                    self.recon_block(module)
            elif name == 'up':
                self.recon_up_model(module)
            else:
                self.recon_model(module)

    def recon_up_model(self, module: nn.Module):
        """
        layer reconstruction. 
        """
        for up_name, up_module in reversed(list(module.named_children())):
            if up_name == self.atten_layer:
                logger.info('reconstruction for up 1 modulelist')
                self.recon_block(up_module.block[0])
                self.recon_block(up_module.attn[0])
                self.recon_block(up_module.block[1])
                self.recon_block(up_module.attn[1])
                self.recon_block(up_module.block[2])
                self.recon_block(up_module.attn[2]) 
                layer_reconstruction(self.model, up_module.upsample.conv, **self.kwargs)

            elif isinstance(up_module, QuantModule):
                if up_module.can_recon == False:
                    continue
                if up_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(up_name))
                    layer_reconstruction(self.model, up_module, **self.kwargs)
            elif isinstance(up_module, BaseQuantBlock):
                if up_module.can_recon == False:
                    continue
                if up_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(up_name))
                    self.recon_block(up_module)
            else:
                self.recon_model(up_module)

    def recon_block(self, block: nn.Module):
        """
        layer reconstruction. 
        """
        if isinstance(block, QuantResnetBlock):
            self.recon_QuantResnetBlock_block(block)
        elif isinstance(block, QuantAttnBlock):
            self.recon_QuantAttnBlock_block(block)

    def recon_QuantResnetBlock_block(self, module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                if module.can_recon == False:
                    continue
                if module.ignore_reconstruction is True:
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(self.model, module, **self.kwargs)
            else:
                self.recon_QuantResnetBlock_block(module)

    def recon_QuantAttnBlock_block(self, module: nn.Module):
        layer_reconstruction(self.model, module.q, **self.kwargs)
        layer_reconstruction(self.model, module.k, **self.kwargs)
        layer_reconstruction(self.model, module.v, **self.kwargs)
        layer_reconstruction(self.model, module.proj_out, **self.kwargs)

    def recon(self):
        self.recon_model(self.model)
        return self.model


class skip_LDM_Model():
    def __init__(self, qnn, model_type="imagenet"):
        self.model = qnn
        self.skip = 0
        self.model_type = model_type

    def set_skip_model(self, model: nn.Module):
        for name, module in model.named_modules():
                
            if self.model_type in ["imagenet", "stable", "bedroom"]: #
                if name == "model.input_blocks.1.0":
                    self.skip = 1
                elif name == "model.output_blocks.11.0":
                    self.skip = 2
            elif self.model_type == "church":
                if name == "model.input_blocks.1.0":
                    self.skip = 1
                elif name == "model.output_blocks.14.0":
                    self.skip = 2
            else:
                raise ValueError("Undefind model type")
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.skip_state = self.skip

    def set_skip(self):
        self.set_skip_model(self.model)
        self.model.set_skip_state()
        return self.model


class skip_Model():
    def __init__(self, qnn, atten_layer = '1'):
        self.model = qnn
        self.down_name = None
        self.skip = 0
        self.atten_layer = atten_layer

    def set_skip_model(self, module: nn.Module):
        for name, module in module.named_children():
            if self.down_name == None and name == 'down':
                self.down_name = 'down'
            if self.down_name == 'down' and name == self.atten_layer and isinstance(module, BaseQuantBlock) == 0:
                logger.info('set skip for down 1 modulelist')
                if module.block[0].skip_start:
                    self.skip = 1
                module.block[0].skip_state = self.skip
                module.attn[0].skip_state = self.skip
                if module.block[1].skip_start:
                    self.skip = 1
                module.block[1].skip_state = self.skip
                module.attn[1].skip_state = self.skip
                module.downsample.conv.skip_state = self.skip
                self.down_name = 'over'
            elif isinstance(module, QuantModule):
                logger.info('set skip for layer {}'.format(name))
                if module.skip_start:
                    self.skip = 1
                elif module.skip_end:
                    self.skip = 2
                module.skip_state = self.skip
            elif isinstance(module, BaseQuantBlock):
                logger.info('set skip for block {}'.format(name))
                if module.skip_start:
                    self.skip = 1
                elif module.skip_end:
                    self.skip = 2
                module.skip_state = self.skip
            elif name == 'up':
                self.set_skip_up_model(module)
            else:
                self.set_skip_model(module)

    def set_skip_up_model(self, module: nn.Module):
        for up_name, up_module in reversed(list(module.named_children())):
            if up_name == self.atten_layer:
                logger.info('set skip for up 1 modulelist')
                if up_module.block[0].skip_end:
                    self.skip = 2
                up_module.block[0].skip_state = self.skip
                up_module.attn[0].skip_state = self.skip
                if up_module.block[1].skip_end:
                    self.skip = 2
                up_module.block[1].skip_state = self.skip
                up_module.attn[1].skip_state = self.skip
                if up_module.block[2].skip_end:
                    self.skip = 2
                up_module.block[2].skip_state = self.skip
                up_module.attn[2].skip_state = self.skip
                up_module.upsample.conv.skip_state = self.skip
            elif isinstance(up_module, QuantModule):
                logger.info('set skip for layer {}'.format(up_name))
                if up_module.skip_start:
                    self.skip = 1
                elif up_module.skip_end:
                    self.skip = 2
                up_module.skip_state = self.skip
            elif isinstance(up_module, BaseQuantBlock):
                logger.info('set skip for block {}'.format(up_name))
                if up_module.skip_start:
                    self.skip = 1
                elif up_module.skip_end:
                    self.skip = 2
                up_module.skip_state = self.skip
            else:
                self.set_skip_model(up_module)
                
    def set_skip(self):
        self.set_skip_model(self.model)
        self.model.set_skip_state()
        return self.model



