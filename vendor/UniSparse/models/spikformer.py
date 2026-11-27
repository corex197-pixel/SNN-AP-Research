import torch
import torch.nn as nn
from .submodules.blocks import ConvBlock
from .submodules.layers import SparseConv2d, SparseBN, SparseLIF, SpikingMatmul, SparseLIF_attn, IdentityBN, SparseLIFwithPooling
from spikingjelly.activation_based import layer, neuron, surrogate
from typing import List
from timm.models.registry import register_model


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
    ) -> None:
        super().__init__()
        self.up = ConvBlock(in_features, hidden_features, 1, 1, 0)
        self.down = ConvBlock(hidden_features, out_features, 1, 1, 0)

    def first_convs(self) -> List[SparseConv2d]:
        return self.up.first_convs()

    def last_bns(self) -> List[SparseBN]:
        return self.down.last_bns()

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.down.register_next_convs(convs)
        self.up.register_next_convs(self.down.first_convs())

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.up.register_prev_bns(bns)
        self.down.register_prev_bns(self.up.last_bns())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat = x.clone()
        x = self.up(x)
        x = self.down(x)
        return x + x_feat


class SSA(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.qkv = ConvBlock(dim, 3 * dim, 1, 1, 0)
        self.proj = ConvBlock(dim, dim, 1, 1, 0)
        self.idbn = IdentityBN()
        self.attn = SparseLIF_attn()
        self.mul1 = SpikingMatmul('both')
        self.mul2 = SpikingMatmul('r')

    def first_convs(self) -> List[SparseConv2d]:
        return self.qkv.first_convs()

    def last_bns(self) -> List[SparseBN]:
        return self.proj.last_bns()

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.proj.register_next_convs(convs)
        self.attn.register_next_convs(self.proj.first_convs())
        self.qkv.register_next_convs([])

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.qkv.register_prev_bns(bns)
        self.attn.register_prev_bns([self.idbn])
        self.proj.register_prev_bns([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, N, C, H, W]
        x_feat = x.clone()
        T, N, C, H, W = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(T, N, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = torch.unbind(qkv, dim=2)

        out = self.mul1(v, k.transpose(-2, -1))
        out = self.mul2(out, q)
        out = out.view(T, N, C, H, W)
        out = out * self.scale
        out = self.idbn(out)
        out = self.attn(out)

        out = self.proj(out)
        return out + x_feat


class SPS(nn.Module):
    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            SparseConv2d(in_channels, dim // 8, kernel_size=3, stride=1, padding=1),
            SparseBN(dim // 8),
            SparseLIFwithPooling(layer.MaxPool2d(3, 2, 1, step_mode='m')),
        )
        self.conv2 = nn.Sequential(
            SparseConv2d(dim // 8, dim // 4, kernel_size=3, stride=1, padding=1),
            SparseBN(dim // 4),
            SparseLIFwithPooling(layer.MaxPool2d(3, 2, 1, step_mode='m')),
        )
        self.conv3 = nn.Sequential(
            SparseConv2d(dim // 4, dim // 2, kernel_size=3, stride=1, padding=1),
            SparseBN(dim // 2),
            SparseLIFwithPooling(layer.MaxPool2d(3, 2, 1, step_mode='m')),
        )
        self.conv4 = nn.Sequential(
            SparseConv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
            SparseBN(dim),
            SparseLIFwithPooling(layer.MaxPool2d(3, 2, 1, step_mode='m')),
        )
        self.proj = ConvBlock(dim, dim)

    def first_convs(self) -> List[SparseConv2d]:
        return [self.conv1[0]]

    def last_bns(self) -> List[SparseBN]:
        return self.proj.last_bns()

    def register_next_convs(self, convs: List[SparseConv2d] = []):
        self.proj.register_next_convs(convs)
        self.conv4[-1].register_next_convs(convs + self.proj.first_convs())
        self.conv3[-1].register_next_convs([self.conv4[0]])
        self.conv2[-1].register_next_convs([self.conv3[0]])
        self.conv1[-1].register_next_convs([self.conv2[0]])

    def register_prev_bns(self, bns: List[SparseBN] = []):
        self.conv1[-1].register_prev_bns([self.conv1[1]])
        self.conv2[-1].register_prev_bns([self.conv2[1]])
        self.conv3[-1].register_prev_bns([self.conv3[1]])
        self.conv4[-1].register_prev_bns([self.conv4[1]])
        self.proj.register_prev_bns([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.proj(x) + x
        return x


class Spikformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        T: int = 4,
        mlp_ratio: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.skip = ['head.conv1']

        self.T = T
        self.head = SPS(in_channels, dim)
        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(SSA(dim, num_heads))
            self.layers.append(MLP(dim, dim, dim * mlp_ratio))
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.classifier = layer.Linear(dim, num_classes, step_mode='m')

        self.register_activations()

    def register_activations(self):
        next_convs = []
        for layer in self.layers[::-1]:
            layer.register_next_convs(next_convs)
            next_convs.extend(layer.first_convs())
        self.head.register_next_convs(next_convs)

        self.head.register_prev_bns()
        prev_bns = self.head.last_bns()
        for layer in self.layers:
            layer.register_prev_bns(prev_bns)
            prev_bns = layer.last_bns()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [N, T, C, H, W] -> [T, N, C, H, W]
            x = x.transpose(0, 1)
        x = self.head(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


@register_model
def spikformer_8_384(**kwargs):
    return Spikformer(in_channels=3, dim=384, num_heads=8, num_layers=8, **kwargs)
