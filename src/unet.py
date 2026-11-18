"""Class-conditional U-Net used by every run (A-E), so gains come from the formulation only.

Channels 64/128/256 at resolutions 32/16/8, two residual blocks per level, self-attention at
8x8 and in the middle block, sinusoidal time embedding, and a learned class embedding with
an extra "null" label (index = num_classes) for classifier-free guidance.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def group_norm(ch):
    return nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)


def zero_init(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout):
        super().__init__()
        self.norm1 = group_norm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)  # FiLM scale & shift
        self.norm2 = group_norm(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = group_norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = zero_init(nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))  # b, heads, hw, d
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class EmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, ResBlock) else layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes=7, in_ch=3, base=64, mults=(1, 2, 4), num_res=2,
                 attn_res=(8,), img_size=32, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.null_label = num_classes
        self.base = base
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        chs, ch, res = [base], base, img_size
        self.down = nn.ModuleList()
        for i, m in enumerate(mults):
            out = base * m
            for _ in range(num_res):
                layers = [ResBlock(ch, out, emb_dim, dropout)]
                ch = out
                if res in attn_res:
                    layers.append(AttnBlock(ch))
                self.down.append(EmbedSequential(*layers))
                chs.append(ch)
            if i != len(mults) - 1:
                self.down.append(EmbedSequential(Downsample(ch)))
                chs.append(ch)
                res //= 2

        self.mid = EmbedSequential(ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch),
                                   ResBlock(ch, ch, emb_dim, dropout))

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            out = base * m
            for j in range(num_res + 1):
                layers = [ResBlock(ch + chs.pop(), out, emb_dim, dropout)]
                ch = out
                if res in attn_res:
                    layers.append(AttnBlock(ch))
                if i != 0 and j == num_res:
                    layers.append(Upsample(ch))
                    res *= 2
                self.up.append(EmbedSequential(*layers))

        self.out = nn.Sequential(group_norm(ch), nn.SiLU(), zero_init(nn.Conv2d(ch, in_ch, 3, padding=1)))

    def forward(self, x, t, y):
        emb = self.time_mlp(timestep_embedding(t, self.base)) + self.label_emb(y)
        h = self.conv_in(x)
        hs = [h]
        for block in self.down:
            h = block(h, emb)
            hs.append(h)
        h = self.mid(h, emb)
        for block in self.up:
            h = block(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)
