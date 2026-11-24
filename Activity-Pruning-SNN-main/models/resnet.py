import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from typing import Optional, Callable, List, Dict, Any
from .submodules.layers import ATLIF, BN, ConvBlock
from .submodules.blocks import BasicSEWBlock, BottleneckSEWBlock

__all__ = [
    'sew_resnet19', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101',
    'sew_resnet152', 'sew_resnet18_tiny', 'sew_resnet34_tiny']

class ResNet(nn.Module):
    def __init__(
        self,
        block: Callable[..., Any],
        planes: List[int],
        layers: List[int],
        prologue: nn.Module,
        epilogue: nn.Module,
        T: int = 4,
        groups: int = 1,
        width_per_group: int = 64,
        zero_init_residual: bool = False,
        norm_layer: Callable[..., Any] = BN,
        norm_layer_kwargs: Dict = {},
        activation: Callable[..., Any] = ATLIF,
        activation_kwargs: Dict = {},
    ):
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.norm_layer_kwargs = norm_layer_kwargs
        self.activation = activation
        self.activation_kwargs = activation_kwargs

        self.skip = ['prologue']

        self.macs = 0.

        self.T = T
        self.inplanes = planes[0]

        self.groups = groups
        self.base_width = width_per_group

        self.prologue = prologue
        

        self.layers = nn.Sequential()
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(self._make_layer(block, planes[i], layers[i]))
            else:
                self.layers.append(self._make_layer(block, planes[i], layers[i], stride=2))
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.epilogue = epilogue

        self.init_weight()
        if zero_init_residual:
            self.zero_init_blocks()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, BottleneckSEWBlock):
                nn.init.constant_(m.conv3.norm_layer.weight, 0)
            elif isinstance(m, BasicSEWBlock):
                nn.init.constant_(m.conv2.norm_layer.weight, 0)

    def _make_layer(self, block: Callable[..., Any], planes: int, blocks: int, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(self.inplanes, planes * block.expansion, kernel_size=1,
                                   stride=stride, padding=0, norm_layer=self.norm_layer,
                                   norm_layer_kwargs=self.norm_layer_kwargs,
                                   activation=self.activation,
                                   activation_kwargs=self.activation_kwargs)

        layers = nn.Sequential()
        layers.append(
            block(self.inplanes, planes, stride=stride, groups=self.groups,
                  base_width=self.base_width, downsample=downsample, norm_layer=self.norm_layer,
                  norm_layer_kwargs=self.norm_layer_kwargs, activation=self.activation,
                  activation_kwargs=self.activation_kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, groups=self.groups,
                      base_width=self.base_width, downsample=None, norm_layer=self.norm_layer,
                      norm_layer_kwargs=self.norm_layer_kwargs, activation=self.activation,
                      activation_kwargs=self.activation_kwargs))

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [N, T, C, H, W] -> [T, N, C, H, W]
            x = x.transpose(0, 1)
        x = self.prologue(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.epilogue(x)
        return x


def sew_resnet19(
    num_classes: int = 10,
    norm_layer: Callable[..., Any] = BN,
    norm_layer_kwargs: Dict = {},
    activation: Callable[..., Any] = ATLIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:

    prologue = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer,
                         norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                         activation_kwargs=activation_kwargs)
    epilogue = nn.Sequential(
        layer.Linear(512, 256, step_mode='m'),
        activation(**activation_kwargs),
        layer.Linear(256, num_classes, step_mode='m'),
    )

    return ResNet(BasicSEWBlock, [128, 256, 512], [3, 3, 2], prologue, epilogue,
                  norm_layer=norm_layer, norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                  activation_kwargs=activation_kwargs, **kwargs)


def _resnet(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    norm_layer: Callable[..., Any] = BN,
    norm_layer_kwargs: Dict = {},
    activation: Callable[..., Any] = ATLIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:

    prologue = nn.Sequential(
        layer.Conv2d(3, 64, 7, 2, 3, bias=False, step_mode='m'),
        norm_layer(64, **norm_layer_kwargs),
        activation(**activation_kwargs),
        layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m'),
    )
    epilogue = layer.Linear(512 * block.expansion, num_classes, step_mode='m')

    return ResNet(block, [64, 128, 256, 512], layers, prologue, epilogue, norm_layer=norm_layer,
                  norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                  activation_kwargs=activation_kwargs, **kwargs)


def _resnet_tiny(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    norm_layer: Callable[..., Any] = BN,
    norm_layer_kwargs: Dict = {},
    activation: Callable[..., Any] = ATLIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:

    prologue = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer,
                         norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                         activation_kwargs=activation_kwargs)
    epilogue = layer.Linear(512 * block.expansion, num_classes, step_mode='m')

    return ResNet(block, [64, 128, 256, 512], layers, prologue, epilogue, norm_layer=norm_layer,
                  norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                  activation_kwargs=activation_kwargs, **kwargs)


def sew_resnet18(**kwargs):
    return _resnet(BasicSEWBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _resnet(BasicSEWBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return _resnet(BottleneckSEWBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet101(**kwargs):
    return _resnet(BottleneckSEWBlock, [3, 4, 23, 3], **kwargs)


def sew_resnet152(**kwargs):
    return _resnet(BottleneckSEWBlock, [3, 8, 36, 3], **kwargs)


def sew_resnet18_tiny(**kwargs):
    return _resnet_tiny(BasicSEWBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34_tiny(**kwargs):
    return _resnet_tiny(BasicSEWBlock, [3, 4, 6, 3], **kwargs)
