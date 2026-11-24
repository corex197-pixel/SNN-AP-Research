import torch
from torch import nn
from .layers import ConvBlock, ATLIF, BN
from typing import Optional, Callable, Any, Dict


class BasicSEWBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
        norm_layer: Callable[..., Any] = BN,
        norm_layer_kwargs: Dict = {},
        activation: Callable[..., Any] = ATLIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(BasicSEWBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        self.conv1 = ConvBlock(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                               groups=groups, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv2 = ConvBlock(planes, planes, kernel_size=3, stride=1, padding=1,
                               groups=groups, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out


class BottleneckSEWBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
        norm_layer: Callable[..., Any] = BN,
        norm_layer_kwargs: Dict = {},
        activation: Callable[..., Any] = ATLIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(BottleneckSEWBlock, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.downsample = downsample
        self.conv1 = ConvBlock(inplanes, width, kernel_size=1, stride=1, padding=0, groups=groups,
                               norm_layer=norm_layer, norm_layer_kwargs=norm_layer_kwargs,
                               activation=activation, activation_kwargs=activation_kwargs)
        self.conv2 = ConvBlock(width, width, kernel_size=3, stride=stride, padding=1, groups=groups,
                               norm_layer=norm_layer, norm_layer_kwargs=norm_layer_kwargs,
                               activation=activation, activation_kwargs=activation_kwargs)
        self.conv3 = ConvBlock(width, planes * self.expansion, kernel_size=1, stride=1, padding=0,
                               groups=groups, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out
