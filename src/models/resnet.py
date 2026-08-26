

















from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:




    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class ResBlock2D(nn.Module):


    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.gn1 = _make_gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.gn2 = _make_gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        return F.gelu(x + h)


class Downsample2D(nn.Module):


    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1)
        self.gn = _make_gn(ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        return F.gelu(x)


class Upsample2D(nn.Module):


    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.gn = _make_gn(ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.gn(x)
        return F.gelu(x)


class ResNetED2D(nn.Module):










    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 32,
        levels: int = 2,
        blocks_per_level: int = 2,
        bottleneck_blocks: int = 4,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.levels = levels

        
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            _make_gn(base_ch),
            nn.GELU(),
        )

        
        enc_blocks = []
        downs = []
        ch = base_ch
        for lv in range(levels):
            for _ in range(blocks_per_level):
                enc_blocks.append(ResBlock2D(ch))
            ch_next = ch * 2
            downs.append(Downsample2D(ch, ch_next))
            ch = ch_next
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)

        
        self.bottleneck = nn.Sequential(*[ResBlock2D(ch) for _ in range(bottleneck_blocks)])

        
        ups = []
        dec_blocks = []
        for lv in range(levels):
            ch_prev = ch // 2
            ups.append(Upsample2D(ch, ch_prev))
            ch = ch_prev
            for _ in range(blocks_per_level):
                dec_blocks.append(ResBlock2D(ch))
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        
        self.head = nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        
        idx = 0
        for lv in range(self.levels):
            for _ in range(len(self.enc_blocks) // self.levels):
                x = self.enc_blocks[idx](x)
                idx += 1
            x = self.downs[lv](x)

        
        x = self.bottleneck(x)

        
        didx = 0
        for lv in range(self.levels):
            x = self.ups[lv](x)
            for _ in range(len(self.dec_blocks) // self.levels):
                x = self.dec_blocks[didx](x)
                didx += 1

        return self.head(x)
