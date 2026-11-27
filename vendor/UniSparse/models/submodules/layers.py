import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import surrogate, neuron
from typing import Callable, Any, Dict, List
from abc import ABC, abstractmethod


class IF(neuron.IFNode):
    def __init__(self):
        super().__init__(v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                         detach_reset=True, step_mode='m', backend='cupy', store_v_seq=False)


class LIF(neuron.LIFNode):
    def __init__(self):
        super().__init__(tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class BN(layer.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features, eps=0.00001, momentum=0.1, affine=True,
                         track_running_stats=True, step_mode='m')


class PseudoRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x


class SparseConv2d(layer.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, padding_mode='zeros', step_mode='m')
        assert groups == 1, 'groups > 1 is not supported'
        self.register_buffer('threshold', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('penalty', torch.tensor(0, dtype=torch.float32))

    def sparse_weight(self):
        return torch.sign(self.weight) * PseudoRelu.apply(torch.abs(self.weight) - self.threshold)

    @torch.no_grad()
    def sparsity(self):
        return (self.sparse_weight() == 0).float().mean().item()

    @torch.no_grad()
    def density(self):
        return 1 - self.sparsity()

    @torch.no_grad()
    def sparse_mask(self):
        return (self.sparse_weight() != 0).float()

    def clear_penalty(self):
        self.penalty.fill_(0)

    @torch.no_grad()
    def penalize(self, lmbda: float):
        self.threshold += lmbda * self.penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.sparse_weight()
        y_shape = [x.shape[0], x.shape[1]]
        y = x.flatten(0, 1)
        y = torch.nn.functional.conv2d(y, weight, self.bias, self.stride, self.padding,
                                       self.dilation, self.groups)
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class SparseBN(BN):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.register_buffer('penalty_gamma', torch.zeros(num_features, dtype=torch.float32))
        self.register_buffer('penalty_beta', torch.zeros(num_features, dtype=torch.float32))
        self.pre_norm_input = None

    def clear_penalty(self):
        self.penalty_gamma.fill_(0)
        self.penalty_beta.fill_(0)
        self.pre_norm_input = None

    @torch.no_grad()
    def penalize(self, lmbda: float):
        self.weight.data -= lmbda * self.penalty_gamma
        self.bias.data -= lmbda * self.penalty_beta

    def local_forward(self) -> torch.Tensor:
        shape = self.pre_norm_input.shape
        x = self.pre_norm_input.flatten(0, 1)
        x = torch.nn.functional.batch_norm(x, None, None, self.weight, self.bias, True,
                                           self.momentum, self.eps)
        x = x.view(shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.pre_norm_input = x.clone().detach()
        return super().forward(x)


class IdentityBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm_input = None

    def local_forward(self) -> torch.Tensor:
        return self.pre_norm_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.pre_norm_input = x.clone().detach()
        return x


class SparseNeuron(nn.Module, ABC):
    @abstractmethod
    def __init__(self, flatten: int = None):
        super().__init__()
        self.activation = None
        self.next_convs = []
        self.prev_bns = []
        self.flatten = flatten

    def register_next_convs(self, convs: List[SparseConv2d]):
        self.next_convs.extend(convs)

    def register_prev_bns(self, bns: List[SparseBN]):
        self.prev_bns.extend(bns)

    def extra_repr(self):
        return super().extra_repr(
        ) + f'next_convs={len(self.next_convs)}, prev_bns={len(self.prev_bns)}'

    def local_forward(self) -> torch.Tensor:
        # local forward
        pre_activations = []
        for bn in self.prev_bns:
            pre_activations.append(bn.local_forward())
        pre_activation = torch.sum(torch.stack(pre_activations), dim=0)
        spike = self.activation(pre_activation)
        self.activation.reset()
        return spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        return x


class SparseLIF(SparseNeuron):
    def __init__(self, flatten: int = None):
        super().__init__(flatten)
        self.activation = LIF()


class SparseLIFwithPooling(SparseNeuron):
    def __init__(self, pool: nn.Module, flatten: int = None):
        super().__init__(flatten)
        self.pool = pool
        self.activation = LIF()

    def local_forward(self) -> torch.Tensor:
        # local forward
        pre_activations = []
        for bn in self.prev_bns:
            pre_activations.append(bn.local_forward())
        pre_activation = torch.sum(torch.stack(pre_activations), dim=0)
        pooled = self.pool(pre_activation)
        spike = self.activation(pooled)
        self.activation.reset()
        return spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.activation(x)
        return x


class SparseLIF_attn(SparseNeuron):
    def __init__(self):
        super().__init__()
        self.activation = neuron.LIFNode(v_threshold=0.5, surrogate_function=surrogate.ATan(),
                                         detach_reset=True, step_mode='m', backend='cupy')


class TLIF(neuron.LIFNode):
    def __init__(self, channels: int, init_v_threshold: float = 1.):
        super().__init__(tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='torch', store_v_seq=False)
        del self.v_threshold
        self.register_parameter(
            'v_threshold',
            nn.Parameter(torch.ones(channels, dtype=torch.float32) * init_v_threshold))
        self.init_v_threshold = init_v_threshold

    def extra_repr(self):
        return f'init_v_threshold={self.init_v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}, tau={self.tau}'

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.surrogate_function(self.v - self.v_threshold[None, :, None, None])
        self.neuronal_reset(spike)
        return spike

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
        spike = torch.stack(y_seq)
        return spike


class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)
