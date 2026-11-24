import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import surrogate, neuron
from typing import Callable, Any, Dict

import math
import numpy as np

def get_activation(name):
    return ATLIF

def get_thresolds(model):
    threshs = []
    for m in model.modules():
        if isinstance(m, ATLIF):
            if isinstance(m.thresh, float):
                threshs.append(m.thresh)
            else:
                threshs.append(m.thresh.item())
    return np.array(threshs)

def static_firing_rate(model):
    rates = []
    for m in model.modules():
        if isinstance(m, ATLIF):
            rates.append(m.r)
    return np.array(rates)


def static_spike_count(model):
    rates = []
    for m in model.modules():
        if isinstance(m, ATLIF):
            rates.append(m.s)
    return np.array(rates)

def zif_backward(x, thre):
    result = (1. - (x/thre).abs()).clamp_min(0)
    return result

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None
    
class Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thre, sp):
        out = (input >= thre).float()
        thre_updates = (sp * zif_backward(input-thre, thre) * out).sum(0).mean().item()
        ctx.save_for_backward(input, thre)
        return out*thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input, dummy):
        (input, thre) = ctx.saved_tensors
        input = (input - thre) / thre
        tmp = (1.0 - input.abs()).clamp(min=0)
        grad_input = grad_input * tmp
        grad_thre = - (grad_input * tmp).mean()
        return grad_input, grad_thre, None
    
class ATLIF(nn.Module):
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0, sp=0.):
        super(ATLIF, self).__init__()
        self.act = Surrogate.apply
        self.thresh = nn.Parameter(torch.tensor(thresh), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.T = T
        self.r = 0
        self.act_value = 0.
        self.sp = sp
        self.update_value = 0.
        self.s = 0.

    def forward(self, x):
        mem = 0.
        spike_pot = []
        for t in range(self.T):
            mem = mem * self.tau + x[t, ...]
            spike, thre_updates = self.act(mem, self.thresh, self.sp) 
            mem = (1. - spike/self.thresh.detach()) * mem
            spike_pot.append(spike)
            self.update_value += (thre_updates/self.T)
        x = torch.stack(spike_pot, dim=0)
        self.r = (x/self.thresh.data).mean().item()
        self.s = (x.mean(1)/self.thresh.data).sum().item()
        self.act_value = x.reshape(x.size(0), -1).mean(1).sum()
        return x

class BN(layer.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features, eps=0.00001, momentum=0.1, affine=True,
                         track_running_stats=True, step_mode='m')

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: Callable[..., Any] = BN,
        norm_layer_kwargs: Dict = {},
        activation: Callable[..., Any] = ATLIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = layer.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
            bias=bias,
            padding_mode='zeros',
            step_mode='m',
        )
        self.norm_layer = norm_layer(out_channels, **norm_layer_kwargs)
        self.activation = activation(**activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm_layer(out)
        out = self.activation(out)
        return out