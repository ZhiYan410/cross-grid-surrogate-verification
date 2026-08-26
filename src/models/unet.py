














from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:





    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):








    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32, depth: int = 4) -> None:
        super().__init__()

        
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_ch
        for d in range(depth):
            outc = base_ch * (2**d)
            self.down_blocks.append(conv_block(ch, outc))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            ch = outc

        
        self.bottleneck = conv_block(ch, ch * 2)
        ch = ch * 2

        
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for d in reversed(range(depth)):
            outc = base_ch * (2**d)
            
            self.up_convs.append(nn.ConvTranspose2d(ch, outc, kernel_size=2, stride=2))
            
            self.up_blocks.append(conv_block(outc * 2, outc))
            ch = outc

        
        self.head = nn.Conv2d(ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:




        skips = []

        
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)   
            x = pool(x)       

        
        x = self.bottleneck(x)

        
        for up, block in zip(self.up_convs, self.up_blocks):
            x = up(x)                     
            skip = skips.pop()            
            x = torch.cat([x, skip], dim=1)  
            x = block(x)

        
        x = self.head(x)
        return x
