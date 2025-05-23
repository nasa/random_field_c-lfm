"""
Code sources:
https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=1841s
https://huggingface.co/blog/annotated-diffusion
"""

import torch
from torch import nn, Tensor, einsum
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np


def exists(x):
    return x is not None


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super().__init__()
        self.rearrange = Rearrange("b c (l p1) -> b (c p1) l", p1=2)
        self.conv = nn.Conv1d(in_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.rearrange(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.upsample(x))


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale: Tensor = None, shift: Tensor = None) -> Tensor:
        x = self.norm(self.conv(x))
        if exists(scale) and exists(shift):
            x = x * (scale + 1.0) + shift
        return self.act(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor):

        h = self.block1(x)
        return self.block2(h) + self.skip(x)


class RMSNorm(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, in_channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


# https://huggingface.co/blog/annotated-diffusion


class LinearAttention(nn.Module):
    """
    Computes global attention via a weird hack.
    """

    def __init__(self, in_channels: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = heads * dim_head

        self.norm = RMSNorm(in_channels)
        self.qkv = nn.Conv1d(in_channels, 3 * hidden_dim, kernel_size=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, in_channels, kernel_size=1), RMSNorm(in_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, l = x.shape

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # takes the channel dimention and reshapes into heads, channels
        # also flattens the feature maps into vectors
        # the shape is now b h c d where d is dim_head
        q = rearrange(q, "b (h c) l -> b h c l", h=self.heads)
        k = rearrange(k, "b (h c) l -> b h c l", h=self.heads)
        v = rearrange(v, "b (h c) l -> b h c l", h=self.heads)
        # softmax along dim_head dim
        q = q.softmax(dim=2) * self.scale
        # softmax along flattened image dim
        k = k.softmax(dim=3)
        # compute comparison betweeen keys and values to produce context.
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c l -> b (h c) l")
        return self.output(out)


# https://huggingface.co/blog/annotated-diffusion


class Attention(nn.Module):
    """
    Computes full pixelwise attention.
    """

    def __init__(self, in_channels: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = heads * dim_head

        self.norm = RMSNorm(in_channels)
        self.qkv = nn.Conv1d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, in_channels, kernel_size=1), RMSNorm(in_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, l = x.shape

        x = self.norm(x)

        # compute the queries, keys, and values of the incoming feature maps
        q, k, v = torch.chunk(self.qkv(x), 3, dim=1)
        # takes the channel dimention and reshapes into heads, channels
        q = rearrange(q, "b (h c) l -> b h c l", h=self.heads)
        k = rearrange(k, "b (h c) l -> b h c l", h=self.heads)
        v = rearrange(v, "b (h c) l -> b h c l", h=self.heads)
        q = q * self.scale
        # multiplication of the query and key matrixes for each head
        sim = einsum("b h c i, b h c j -> b h i j", q, k)
        # subtract the maximum value of each row
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        # make each row into a probability distribution
        attn = sim.softmax(dim=-1)
        # weight the values according to the rows of the attention matrix
        y = einsum("b h i j, b h d j -> b h i d", attn, v)
        # reshape the weighted values back into feature maps
        y = rearrange(y, "b h l d -> b (h d) l")
        return self.output(y)


class Encoder1d(nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int = 64, heads: int = 4, dim_head: int = 32
    ):
        super().__init__()

        self.input_layer = nn.Conv1d(in_channels, 32, kernel_size=1)

        self.downs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResBlock(32, 32),
                        ResBlock(32, 32),
                        LinearAttention(32, heads, dim_head),
                        DownBlock(32, 32),
                    ]
                ),
                nn.ModuleList(
                    [
                        ResBlock(32, 32),
                        ResBlock(32, 32),
                        LinearAttention(32, heads, dim_head),
                        DownBlock(32, 64),
                    ]
                ),
                nn.ModuleList(
                    [
                        ResBlock(64, 64),
                        ResBlock(64, 64),
                        LinearAttention(64, heads, dim_head),
                        DownBlock(64, 64),
                    ]
                ),
                nn.ModuleList(
                    [
                        ResBlock(64, 64),
                        ResBlock(64, 64),
                        LinearAttention(64, heads, dim_head),
                        nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    ]
                ),
            ]
        )

        self.mid_block1 = ResBlock(128, 128)
        self.mid_attention = Attention(128, heads, dim_head)
        self.mid_block2 = ResBlock(128, 128)
        self.latent_projection = nn.Linear(128, latent_dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.input_layer(x)  # (b x 32 x l)

        # residuals = []
        for res1, res2, attention, downsample in self.downs:
            y = res1(y)
            y = res2(y)
            y = attention(y) + y
            y = downsample(y)

        # (b x 128 x l)
        y = self.mid_block1(y)
        y = self.mid_attention(y) + y
        # y = self.mid_block2(y).max(dim = 2).values
        y = self.mid_block2(y).mean(dim=2)

        # Project to desired latent dimension
        y = self.latent_projection(y)
        return y


class Decoder1d(nn.Module):
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.ups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResBlock(64, 64),
                        ResBlock(64, 64),
                        LinearAttention(64, heads, dim_head),
                        UpBlock(64, 64),
                    ]
                ),
                nn.ModuleList(
                    [
                        ResBlock(64, 64),
                        ResBlock(64, 64),
                        LinearAttention(64, heads, dim_head),
                        UpBlock(64, 32),
                    ]
                ),
                nn.ModuleList(
                    [
                        ResBlock(32, 32),
                        ResBlock(32, 32),
                        LinearAttention(32, heads, dim_head),
                        nn.Conv1d(32, 32, kernel_size=3, padding=1),
                    ]
                ),
            ]
        )

        self.output_res = ResBlock(32, 32)
        self.output_layer = nn.Conv1d(32, in_channels, kernel_size=1)

    def forward(self, z: Tensor) -> Tensor:

        for res1, res2, attention, upsample in self.ups:
            z = res1(z)
            z = res2(z)
            z = attention(z) + z
            z = upsample(z)

        # final skip connection to residual layer
        z = self.output_res(z)
        z = self.output_layer(z)
        return z
