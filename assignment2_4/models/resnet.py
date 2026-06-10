import torch
import torch.nn as nn

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=1280, groups=32, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
            
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU()
        
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        
        if temb is not None and self.time_emb_proj is not None:
            # Apply non-linearity to temb before projecting (diffusers design)
            temb_proj = self.time_emb_proj(self.nonlinearity(temb))
            h = h + temb_proj[:, :, None, None]
            
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        
        return self.conv_shortcut(x) + h
