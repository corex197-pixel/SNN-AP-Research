import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from .submodules.blocks import ConvBlock
from timm.models.registry import register_model
from typing import Any


class VGGSNN(nn.Module):
    def __init__(
        self,
        T: int = 10,
        base_channels: int = 64,
        num_classes: int = 10,
    ):
        super(VGGSNN, self).__init__()
        self.skip = ['conv1']
        self.T = T

        self.conv1 = ConvBlock(2, base_channels)
        self.conv2 = ConvBlock(base_channels, base_channels * 2)
        self.pool1 = layer.AvgPool2d(2, 2, step_mode='m')
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 4)
        self.pool2 = layer.AvgPool2d(2, 2, step_mode='m')
        self.conv5 = ConvBlock(base_channels * 4, base_channels * 8)
        self.conv6 = ConvBlock(base_channels * 8, base_channels * 8)
        self.pool3 = layer.AvgPool2d(2, 2, step_mode='m')
        self.conv7 = ConvBlock(base_channels * 8, base_channels * 8)
        self.conv8 = ConvBlock(base_channels * 8, base_channels * 8)
        self.pool4 = layer.AvgPool2d(2, 2, step_mode='m')
        self.classifier = layer.Linear(base_channels * 8 * 3 * 3, num_classes * 10)
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
        self.conv8.register_next_convs()
        self.conv8.register_prev_bns(self.conv7.last_bns())
        self.conv7.register_next_convs(self.conv8.first_convs())
        self.conv7.register_prev_bns(self.conv6.last_bns())
        self.conv6.register_next_convs(self.conv7.first_convs())
        self.conv6.register_prev_bns(self.conv5.last_bns())
        self.conv5.register_next_convs(self.conv6.first_convs())
        self.conv5.register_prev_bns(self.conv4.last_bns())
        self.conv4.register_next_convs(self.conv5.first_convs())
        self.conv4.register_prev_bns(self.conv3.last_bns())
        self.conv3.register_next_convs(self.conv4.first_convs())
        self.conv3.register_prev_bns(self.conv2.last_bns())
        self.conv2.register_next_convs(self.conv3.first_convs())
        self.conv2.register_prev_bns(self.conv1.last_bns())
        self.conv1.register_next_convs(self.conv2.first_convs())
        self.conv1.register_prev_bns()

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
        x = x.flatten(2)
        x = self.classifier(x)
        x = x.unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = self.boost(x).squeeze(2)
        return out


@register_model
def vggsnn(num_classes: int = 10, T: int = 10, base_channels: int = 64, **kwargs: Any) -> VGGSNN:
    return VGGSNN(T, base_channels, num_classes)
