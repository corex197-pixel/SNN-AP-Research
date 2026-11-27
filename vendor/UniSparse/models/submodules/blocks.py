import torch
from torch import nn
from .layers import SparseConv2d, SparseBN, SparseLIF
from typing import Optional, List
from spikingjelly.activation_based import layer

from models.submodules import layers


class ConvBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = SparseConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                 padding=padding, groups=groups, bias=bias)
        self.norm = SparseBN(planes)
        self.activation = SparseLIF()

    def first_convs(self) -> List[SparseConv2d]:
        return [self.conv]

    def last_bns(self) -> List[SparseBN]:
        return []

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.activation.register_next_convs(convs)

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.activation.register_prev_bns([self.norm])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.downsample = downsample
        self.conv1 = SparseConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                                  groups=groups, bias=False)
        self.norm1 = SparseBN(planes)
        self.activation1 = SparseLIF()
        self.conv2 = SparseConv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=groups,
                                  bias=False)
        self.norm2 = SparseBN(planes)
        self.activation2 = SparseLIF()

    def first_convs(self) -> List[SparseConv2d]:
        convs = [self.conv1]
        if self.downsample is not None:
            convs.extend(self.downsample.first_convs())
        return convs

    def last_bns(self) -> List[SparseBN]:
        return []

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.activation2.register_next_convs(convs)
        self.activation1.register_next_convs([self.conv2])
        if self.downsample is not None:
            self.downsample.register_next_convs()

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.activation1.register_prev_bns([self.norm1])
        if self.downsample is not None:
            self.activation2.register_prev_bns(self.downsample.last_bns() + [self.norm2])
            self.downsample.register_prev_bns(bns)
        else:
            self.activation2.register_prev_bns([self.norm2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        out = self.activation2(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.downsample = downsample
        self.conv1 = SparseConv2d(inplanes, width, kernel_size=1, stride=1, padding=0,
                                  groups=groups, bias=False)
        self.norm1 = SparseBN(width)
        self.activation1 = SparseLIF()
        self.conv2 = SparseConv2d(width, width, kernel_size=3, stride=stride, padding=1,
                                  groups=groups, bias=False)
        self.norm2 = SparseBN(width)
        self.activation2 = SparseLIF()
        self.conv3 = SparseConv2d(width, planes * self.expansion, kernel_size=1, stride=1,
                                  padding=0, groups=groups, bias=False)
        self.norm3 = SparseBN(planes * self.expansion)
        self.activation3 = SparseLIF()

    def first_convs(self) -> List[SparseConv2d]:
        convs = [self.conv1]
        if self.downsample is not None:
            convs.extend(self.downsample.first_convs())
        return convs

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.activation3.register_next_convs(convs)
        self.activation2.register_next_convs([self.conv3])
        self.activation1.register_next_convs([self.conv2])
        if self.downsample is not None:
            self.downsample.register_next_convs()

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.activation1.register_prev_bns([self.norm1])
        self.activation2.register_prev_bns([self.norm2])
        if self.downsample is not None:
            self.activation3.register_prev_bns(self.downsample.last_bns() + [self.norm3])
            self.downsample.register_prev_bns(bns)
        else:
            self.activation3.register_prev_bns([self.norm3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        out = self.activation3(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super(DownsampleBlock, self).__init__()
        self.conv = SparseConv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0,
                                 bias=False)
        self.norm = SparseBN(planes)

    def first_convs(self) -> List[SparseConv2d]:
        return [self.conv]

    def last_bns(self) -> List[SparseBN]:
        return [self.norm]

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        pass

    def register_prev_bns(self, bns: List[SparseBN] = []):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


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
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.downsample = downsample
        self.conv1 = ConvBlock(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                               groups=groups, bias=False)
        self.conv2 = ConvBlock(planes, planes, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)

    def first_convs(self) -> List[SparseConv2d]:
        convs = self.conv1.first_convs()
        if self.downsample is not None:
            convs.extend(self.downsample.first_convs())
        return convs

    def last_bns(self) -> List[SparseBN]:
        return []

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.conv2.register_next_convs(convs)
        self.conv1.register_next_convs(self.conv2.first_convs())
        if self.downsample is not None:
            self.downsample.register_next_convs(convs)

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.conv1.register_prev_bns(bns)
        self.conv2.register_prev_bns(self.conv1.last_bns())
        if self.downsample is not None:
            self.downsample.register_prev_bns(bns)

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
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.downsample = downsample
        self.conv1 = ConvBlock(inplanes, width, kernel_size=1, stride=1, padding=0, groups=groups,
                               bias=False)
        self.conv2 = ConvBlock(width, width, kernel_size=3, stride=stride, padding=1, groups=groups,
                               bias=False)
        self.conv3 = ConvBlock(width, planes * self.expansion, kernel_size=1, stride=1, padding=0,
                               groups=groups, bias=False)

    def first_convs(self) -> List[SparseConv2d]:
        convs = self.conv1.first_convs()
        if self.downsample is not None:
            convs.extend(self.downsample.first_convs())
        return convs

    def last_bns(self) -> List[SparseBN]:
        return []

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.conv3.register_next_convs(convs)
        self.conv2.register_next_convs(self.conv3.first_convs())
        self.conv1.register_next_convs(self.conv2.first_convs())
        if self.downsample is not None:
            self.downsample.register_next_convs(convs)

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.conv1.register_prev_bns(bns)
        self.conv2.register_prev_bns(self.conv1.last_bns())
        self.conv3.register_prev_bns(self.conv2.last_bns())
        if self.downsample is not None:
            self.downsample.register_prev_bns(bns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out


class DownsampleSEWBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = ConvBlock(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def first_convs(self) -> List[SparseConv2d]:
        return self.conv.first_convs()

    def last_bns(self) -> List[SparseBN]:
        return self.conv.last_bns()

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.conv.register_next_convs(convs)

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.conv.register_prev_bns(bns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResNetPrologue(nn.Module):
    def __init__(self, in_channels: int, planes: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, planes, kernel_size=7, stride=2, padding=3)
        self.pool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m')

    def first_convs(self) -> List[SparseConv2d]:
        return self.conv.first_convs()

    def last_bns(self) -> List[SparseBN]:
        return self.conv.last_bns()

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.conv.register_next_convs(convs)

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.conv.register_prev_bns(bns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        return x
