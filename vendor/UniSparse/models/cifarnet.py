import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from typing import Callable, Any, Dict
from .submodules.layers import SparseConv2d, SparseBN, SparseLIF, IdentityBN
from .submodules.blocks import ConvBlock
from timm.models.registry import register_model


class Cifar10Net(nn.Module):
    def __init__(
        self,
        T: int = 4,
        base_channels: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.skip = ['static_conv']
        self.T = T

        self.static_conv = ConvBlock(3, base_channels)
        self.conv1 = ConvBlock(base_channels, base_channels)
        self.conv2 = SparseConv2d(base_channels, base_channels, 3, 1, 1)
        self.bn2 = SparseBN(base_channels)
        self.activation2 = SparseLIF()
        self.maxpool2 = layer.MaxPool2d(2, 2, step_mode='m')
        self.conv3 = ConvBlock(base_channels, base_channels)
        self.conv4 = ConvBlock(base_channels, base_channels)
        self.conv5 = SparseConv2d(base_channels, base_channels, 3, 1, 1)
        self.bn5 = SparseBN(base_channels)
        self.activation5 = SparseLIF(flatten=16)
        self.maxpool5 = layer.MaxPool2d(2, 2, step_mode='m')
        self.fc1 = SparseConv2d(base_channels * 8 * 8, (base_channels // 2) * 4 * 4, 1, 1, 0,
                                bias=True)
        self.bn1 = IdentityBN()
        self.activation_fc1 = SparseLIF()
        self.fc2 = SparseConv2d((base_channels // 2) * 4 * 4, num_classes * 10, 1, 1, 0, bias=True)
        self.boost = layer.AvgPool1d(10, 10, step_mode='m')
        self.init_weight()

        self.register_activations()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def register_activations(self):
        self.activation_fc1.register_next_convs([self.fc2])
        self.activation_fc1.register_prev_bns([self.bn1])
        self.activation5.register_next_convs([self.fc1])
        self.activation5.register_prev_bns([self.bn5])
        self.conv4.register_next_convs([self.conv5])
        self.conv4.register_prev_bns([])
        self.conv3.register_next_convs(self.conv4.first_convs())
        self.conv3.register_prev_bns([])
        self.activation2.register_next_convs(self.conv3.first_convs())
        self.activation2.register_prev_bns([self.bn2])
        self.conv1.register_next_convs([self.conv2])
        self.conv1.register_prev_bns([])
        self.static_conv.register_next_convs(self.conv1.first_convs())
        self.static_conv.register_prev_bns([])

    def forward(self, x: torch.Tensor):
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [N, T, C, H, W] -> [T, N, C, H, W]
            x = x.transpose(0, 1)
        x = self.static_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation5(x)
        x = self.maxpool5(x)
        #### [T, N, C, H, W] -> [T, N, CxHxW, 1, 1]
        x = x.view(x.shape[0], x.shape[1], -1, 1, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation_fc1(x)
        x = self.fc2(x)
        x = x.flatten(2)

        x = x.unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = self.boost(x).squeeze(2)

        return out


@register_model
def cifarnet(num_classes: int = 10, T: int = 4, base_channels: int = 256,
             **kwargs: Any) -> Cifar10Net:
    return Cifar10Net(T, base_channels, num_classes)
