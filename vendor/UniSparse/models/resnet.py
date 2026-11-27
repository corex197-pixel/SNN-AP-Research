import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from typing import Optional, Callable, List, Dict, Any
from .submodules.layers import LIF
from .submodules.blocks import BasicBlock, BottleneckBlock, ConvBlock, DownsampleBlock
from .submodules.blocks import BasicSEWBlock, BottleneckSEWBlock, DownsampleSEWBlock
from .submodules.blocks import ResNetPrologue
from timm.models.registry import register_model


class ResNet(nn.Module):
    def __init__(
        self,
        block: Callable[..., Any],
        downsample: Callable[..., Any],
        planes: List[int],
        layers: List[int],
        prologue: nn.Module,
        epilogue: nn.Module,
        T: int = 4,
        groups: int = 1,
        width_per_group: int = 64,
        zero_init_residual: bool = False,
        **kwargs,
    ):
        super(ResNet, self).__init__()
        self.skip = ['prologue']

        self.T = T
        self.inplanes = planes[0]

        self.groups = groups
        self.base_width = width_per_group

        self.prologue = prologue
        self.epilogue = epilogue

        self.layers = nn.Sequential()
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(self._make_layer(block, downsample, planes[i], layers[i]))
            else:
                self.layers.append(
                    self._make_layer(block, downsample, planes[i], layers[i], stride=2))
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')

        self.init_weight()
        if zero_init_residual:
            self.zero_init_blocks()

        self.register_activations()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, BottleneckBlock):
                nn.init.constant_(m.norm3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.norm2.weight, 0)

    def _make_layer(self, block: Callable[..., Any], downsample_block: Callable[..., Any],
                    planes: int, blocks: int, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_block(self.inplanes, planes * block.expansion, stride=stride)

        layers = nn.Sequential()
        layers.append(
            block(self.inplanes, planes, stride=stride, groups=self.groups,
                  base_width=self.base_width, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, groups=self.groups,
                      base_width=self.base_width, downsample=None))

        return layers

    def register_activations(self):
        next_convs = []
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            for j in range(len(layer) - 1, -1, -1):
                block = layer[j]
                block.register_next_convs(next_convs)
                next_convs = block.first_convs()
        self.prologue.register_next_convs(next_convs)

        self.prologue.register_prev_bns()
        prev_bns = self.prologue.last_bns()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            for j in range(len(layer)):
                block = layer[j]
                block.register_prev_bns(prev_bns)
                prev_bns = block.last_bns()

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


@register_model
def spiking_resnet20(
    num_classes: int = 10,
    **kwargs,
) -> ResNet:

    prologue = ConvBlock(3, 16, kernel_size=3, stride=1, padding=1)
    epilogue = layer.Linear(64, num_classes, step_mode='m')

    return ResNet(BasicBlock, DownsampleBlock, [16, 32, 64], [3, 3, 3], prologue, epilogue,
                  **kwargs)


@register_model
def sew_resnet20(
    num_classes: int = 10,
    **kwargs,
) -> ResNet:

    prologue = ConvBlock(3, 16, kernel_size=3, stride=1, padding=1)
    epilogue = layer.Linear(64, num_classes, step_mode='m')

    return ResNet(BasicSEWBlock, DownsampleSEWBlock, [16, 32, 64], [3, 3, 3], prologue, epilogue,
                  **kwargs)


@register_model
def spiking_resnet19(
    num_classes: int = 10,
    **kwargs,
) -> ResNet:

    prologue = ConvBlock(3, 128, kernel_size=3, stride=1, padding=1)
    epilogue = nn.Sequential(
        layer.Linear(512, 256, step_mode='m'),
        LIF(),
        layer.Linear(256, num_classes, step_mode='m'),
    )

    return ResNet(BasicBlock, DownsampleBlock, [128, 256, 512], [3, 3, 2], prologue, epilogue,
                  **kwargs)


@register_model
def sew_resnet19(
    num_classes: int = 10,
    **kwargs,
) -> ResNet:

    prologue = ConvBlock(3, 128, kernel_size=3, stride=1, padding=1)
    epilogue = nn.Sequential(
        layer.Linear(512, 256, step_mode='m'),
        LIF(),
        layer.Linear(256, num_classes, step_mode='m'),
    )

    return ResNet(BasicSEWBlock, DownsampleSEWBlock, [128, 256, 512], [3, 3, 2], prologue, epilogue,
                  **kwargs)


def _resnet(
    block: Callable[..., Any],
    downsample: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    **kwargs,
) -> ResNet:

    prologue = ResNetPrologue(3, 64)
    epilogue = layer.Linear(512 * block.expansion, num_classes, step_mode='m')

    return ResNet(block, downsample, [64, 128, 256, 512], layers, prologue, epilogue, **kwargs)


def _resnet_tiny(
    block: Callable[..., Any],
    downsample: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    **kwargs,
) -> ResNet:

    prologue = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
    epilogue = layer.Linear(512 * block.expansion, num_classes, step_mode='m')

    return ResNet(block, downsample, [64, 128, 256, 512], layers, prologue, epilogue, **kwargs)


@register_model
def spiking_resnet18(**kwargs):
    return _resnet(BasicBlock, DownsampleBlock, [2, 2, 2, 2], **kwargs)


@register_model
def spiking_resnet34(**kwargs):
    return _resnet(BasicBlock, DownsampleBlock, [3, 4, 6, 3], **kwargs)


@register_model
def spiking_resnet50(**kwargs):
    return _resnet(BottleneckBlock, DownsampleBlock, [3, 4, 6, 3], **kwargs)


@register_model
def spiking_resnet101(**kwargs):
    return _resnet(BottleneckBlock, DownsampleBlock, [3, 4, 23, 3], **kwargs)


@register_model
def spiking_resnet152(**kwargs):
    return _resnet(BottleneckBlock, DownsampleBlock, [3, 8, 36, 3], **kwargs)


@register_model
def spiking_resnet18_tiny(**kwargs):
    return _resnet_tiny(BasicBlock, DownsampleBlock, [2, 2, 2, 2], **kwargs)


@register_model
def spiking_resnet34_tiny(**kwargs):
    return _resnet_tiny(BasicBlock, DownsampleBlock, [3, 4, 6, 3], **kwargs)


@register_model
def sew_resnet18(**kwargs):
    return _resnet(BasicSEWBlock, DownsampleSEWBlock, [2, 2, 2, 2], **kwargs)


@register_model
def sew_resnet34(**kwargs):
    return _resnet(BasicSEWBlock, DownsampleSEWBlock, [3, 4, 6, 3], **kwargs)


@register_model
def sew_resnet50(**kwargs):
    return _resnet(BottleneckSEWBlock, DownsampleSEWBlock, [3, 4, 6, 3], **kwargs)


@register_model
def sew_resnet101(**kwargs):
    return _resnet(BottleneckSEWBlock, DownsampleSEWBlock, [3, 4, 23, 3], **kwargs)


@register_model
def sew_resnet152(**kwargs):
    return _resnet(BottleneckSEWBlock, DownsampleSEWBlock, [3, 8, 36, 3], **kwargs)


@register_model
def sew_resnet18_tiny(**kwargs):
    return _resnet_tiny(BasicSEWBlock, DownsampleSEWBlock, [2, 2, 2, 2], **kwargs)


@register_model
def sew_resnet34_tiny(**kwargs):
    return _resnet_tiny(BasicSEWBlock, DownsampleSEWBlock, [3, 4, 6, 3], **kwargs)
