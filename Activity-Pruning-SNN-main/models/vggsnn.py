import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from .submodules.layers import ATLIF, BN, ConvBlock
from typing import Callable, Any, Dict

__all__ = ['VGGSNN']


class VGGSNN(nn.Module):
    def __init__(
        self,
        T: int = 8,
        base_channels: int = 64,
        num_classes: int = 10,
        norm_layer: Callable[..., Any] = BN,
        norm_layer_kwargs: Dict = {},
        activation: Callable[..., Any] = ATLIF,
        activation_kwargs: Dict = {},
    ):
        super(VGGSNN, self).__init__()
        self.skip = ['conv1']
        self.T = T

        self.conv1 = ConvBlock(2, base_channels, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.pool1 = layer.AvgPool2d((2, 2),  step_mode='m')
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 4, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.pool2 = layer.AvgPool2d((2, 2),  step_mode='m')
        self.conv5 = ConvBlock(base_channels * 4, base_channels * 8, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv6 = ConvBlock(base_channels * 8, base_channels * 8, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.pool3 = layer.AvgPool2d((2, 2),  step_mode='m')
        self.conv7 = ConvBlock(base_channels * 8, base_channels * 8, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.conv8 = ConvBlock(base_channels * 8, base_channels * 8, norm_layer=norm_layer,
                               norm_layer_kwargs=norm_layer_kwargs, activation=activation,
                               activation_kwargs=activation_kwargs)
        self.pool4 = layer.AvgPool2d((2, 2),  step_mode='m')
        self.classifier = layer.Linear(base_channels * 8 * 3 * 3, num_classes * 10,  step_mode='m')
        self.boost = layer.AvgPool1d(10,  step_mode='m')
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

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.classifier(x)
        x = x.flatten(2).unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = self.boost(x).squeeze(2)
        return out
