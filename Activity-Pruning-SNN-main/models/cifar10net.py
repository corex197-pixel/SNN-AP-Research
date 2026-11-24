import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from .submodules.layers import ATLIF, BN, ConvBlock
from typing import Callable, Any, Dict

__all__ = ['Cifar10Net']


class Cifar10Net(nn.Module):
    def __init__(
        self,
        T: int = 8,
        base_channels: int = 256,
        num_classes: int = 10,
        norm_layer: Callable[..., Any] = BN,
        norm_layer_kwargs: Dict = {},
        activation: Callable[..., Any] = ATLIF,
        activation_kwargs: Dict = {},
    ):
        super().__init__()
        self.skip = ['static_conv']
        self.T = T

        self.static_conv = ConvBlock(3, base_channels, norm_layer=norm_layer,
                                     norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                                     activation_kwargs=activation_kwargs)
        self.conv1 = ConvBlock(base_channels, base_channels, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv2 = ConvBlock(base_channels, base_channels, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.maxpool2 = layer.MaxPool2d(2, 2, step_mode='m')
        self.conv3 = ConvBlock(base_channels, base_channels, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv4 = ConvBlock(base_channels, base_channels, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv5 = ConvBlock(base_channels, base_channels, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.maxpool5 = layer.MaxPool2d(2, 2, step_mode='m')
        self.dp1 = layer.Dropout(0.5, step_mode='m')
        self.fc1 = nn.Sequential(
            layer.Linear(base_channels * 8 * 8, (base_channels // 2) * 4 * 4, step_mode='m'),
            activation(**activation_kwargs))
        self.dp2 = layer.Dropout(0.5, step_mode='m')
        self.fc2 = layer.Linear((base_channels // 2) * 4 * 4, num_classes * 10, step_mode='m')
        self.boost = layer.AvgPool1d(10, 10, step_mode='m')
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = x.view(x.shape[0], x.shape[1], -1)

        # print(x.shape)
        #### [T, N, C, H, W] -> [T, N, CxHxW, 1, 1]
        #x = self.dp1(x)
        x = self.fc1(x)
        #x = self.dp2(x)
        x = self.fc2(x)
        x = x.flatten(2)

        x = x.unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = self.boost(x).squeeze(2)

        return out
