from matplotlib.pyplot import xlim
import torch.nn as nn
from .submodules.layers import ATLIF, BN, ConvBlock
from .submodules.blocks import BasicSEWBlock, BottleneckSEWBlock
from spikingjelly.activation_based import layer
import torch

__all__ = ['ResNet20']

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, activation, activation_kwargs, stride=1):
        super().__init__()
        #residual function
        self.residual_function = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_channels, step_mode='m'),
            activation(**activation_kwargs),
            layer.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_channels * BasicBlock.expansion, step_mode='m')
        )
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False, step_mode='m'),
                layer.BatchNorm2d(out_channels * BasicBlock.expansion, step_mode='m')
            )
        self.act = activation(**activation_kwargs)

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        return self.act(x)
    

class ResNet20(nn.Module):
    def __init__(self, T, activation, activation_kwargs, num_classes=10):
        super().__init__()
        block = BasicBlock
        num_block = [3, 3, 3]
        self.in_channels = 16*2
        self.T = T
        self.conv1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(self.in_channels, step_mode='m'),
            activation(**activation_kwargs))
        
        self.skip = ['conv1']
        
        self.conv2_x = self._make_layer(block, 16*2, num_block[0], 1, activation, activation_kwargs)
        self.conv3_x = self._make_layer(block, 32*2, num_block[1], 2, activation, activation_kwargs)
        self.conv4_x = self._make_layer(block, 64*2, num_block[2], 2, activation, activation_kwargs)
        self.avg_pool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.fc = layer.Linear(64 * 2 * block.expansion, num_classes, step_mode='m')

    def _make_layer(self, block, out_channels, num_blocks, stride, activation, activation_kwags):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, activation, activation_kwags, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [N, T, C, H, W] -> [T, N, C, H, W]
            x = x.transpose(0, 1)
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x