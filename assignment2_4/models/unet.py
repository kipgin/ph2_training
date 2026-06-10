import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import get_timestep_embedding, TimeEmbedding
from .resnet import ResnetBlock2D
from .attention import Transformer2DModel

class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, padding=1):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, n_heads, d_head, context_dim, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels, temb_channels),
            ResnetBlock2D(out_channels, out_channels, temb_channels)
        ])
        self.attentions = nn.ModuleList([
            Transformer2DModel(out_channels, n_heads, d_head, context_dim),
            Transformer2DModel(out_channels, n_heads, d_head, context_dim)
        ])
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels, padding=1)])
        else:
            self.downsamplers = None

    def forward(self, x, temb=None, context=None):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, temb)
            x = attn(x, context)
            output_states.append(x)
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
            output_states.append(x)
            
        return x, output_states


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels, temb_channels),
            ResnetBlock2D(out_channels, out_channels, temb_channels)
        ])
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels, padding=1)])
        else:
            self.downsamplers = None

    def forward(self, x, temb=None, context=None):
        output_states = []
        for resnet in self.resnets:
            x = resnet(x, temb)
            output_states.append(x)
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
            output_states.append(x)
            
        return x, output_states


class UNetMidBlock2DModelCrossAttn(nn.Module):
    def __init__(self, in_channels, temb_channels, n_heads, d_head, context_dim):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, in_channels, temb_channels),
            ResnetBlock2D(in_channels, in_channels, temb_channels)
        ])
        self.attentions = nn.ModuleList([
            Transformer2DModel(in_channels, n_heads, d_head, context_dim)
        ])

    def forward(self, x, temb=None, context=None):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        x = self.resnets[1](x, temb)
        return x


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, prev_output_channel, out_channels, temb_channels, n_heads, d_head, context_dim, skip_channels_list, add_upsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels + skip_channels_list[0], out_channels, temb_channels),
            ResnetBlock2D(out_channels + skip_channels_list[1], out_channels, temb_channels),
            ResnetBlock2D(out_channels + skip_channels_list[2], out_channels, temb_channels)
        ])
        self.attentions = nn.ModuleList([
            Transformer2DModel(out_channels, n_heads, d_head, context_dim),
            Transformer2DModel(out_channels, n_heads, d_head, context_dim),
            Transformer2DModel(out_channels, n_heads, d_head, context_dim)
        ])
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])
        else:
            self.upsamplers = None

    def forward(self, x, res_xs, temb=None, context=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_x = res_xs.pop()
            x = torch.cat([x, res_x], dim=1)
            x = resnet(x, temb)
            x = attn(x, context)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, prev_output_channel, out_channels, temb_channels, skip_channels_list, add_upsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels + skip_channels_list[0], out_channels, temb_channels),
            ResnetBlock2D(out_channels + skip_channels_list[1], out_channels, temb_channels),
            ResnetBlock2D(out_channels + skip_channels_list[2], out_channels, temb_channels)
        ])
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])
        else:
            self.upsamplers = None

    def forward(self, x, res_xs, temb=None, context=None):
        for resnet in self.resnets:
            res_x = res_xs.pop()
            x = torch.cat([x, res_x], dim=1)
            x = resnet(x, temb)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class UNet2DConditionModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, sample_size=64, cross_attention_dim=768, attention_head_dim=8, block_out_channels=[320, 640, 1280, 1280], layers_per_block=2, norm_num_groups=32, norm_eps=1e-5):
        super().__init__()
        norm_eps = float(norm_eps)
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        class UNetConfig:
            def __init__(self, in_channels, sample_size):
                self.in_channels = in_channels
                self.sample_size = sample_size
        self.config = UNetConfig(in_channels, sample_size)
        
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.time_proj = nn.Identity()
        self.time_embedding = TimeEmbedding(block_out_channels[0], block_out_channels[0] * 4)
        temb_channels = block_out_channels[0] * 4
        
        self.down_blocks = nn.ModuleList([
            CrossAttnDownBlock2D(block_out_channels[0], block_out_channels[0], temb_channels, attention_head_dim, block_out_channels[0] // attention_head_dim, cross_attention_dim, add_downsample=True),
            CrossAttnDownBlock2D(block_out_channels[0], block_out_channels[1], temb_channels, attention_head_dim, block_out_channels[1] // attention_head_dim, cross_attention_dim, add_downsample=True),
            CrossAttnDownBlock2D(block_out_channels[1], block_out_channels[2], temb_channels, attention_head_dim, block_out_channels[2] // attention_head_dim, cross_attention_dim, add_downsample=True),
            DownBlock2D(block_out_channels[2], block_out_channels[3], temb_channels, add_downsample=False)
        ])
        
        self.mid_block = UNetMidBlock2DModelCrossAttn(block_out_channels[3], temb_channels, attention_head_dim, block_out_channels[3] // attention_head_dim, cross_attention_dim)
        
        # xs stack layout after conv_in + all down_blocks (bottom=index 0, top=last):
        # idx  0        1        2        3        4        5
        #      conv_in  D0r0     D0r1     D0d      D1r0     D1r1
        #      =320     =320     =320     =320     =640     =640
        # idx  6        7        8        9        10       11
        #      D1d      D2r0     D2r1     D2d      D3r0     D3r1
        #      =640     =1280    =1280    =1280    =1280    =1280
        # Total: 12 elements.  Each up block pops 3 (LIFO), 4 blocks × 3 = 12. ✓
        self.up_blocks = nn.ModuleList([
            # pops idx11=1280, idx10=1280, idx9=1280  (D3r1, D3r0, D2d)
            UpBlock2D(block_out_channels[3], block_out_channels[3], block_out_channels[3], temb_channels,
                      [block_out_channels[3], block_out_channels[3], block_out_channels[3]], add_upsample=True),
            # pops idx8=1280, idx7=1280, idx6=640  (D2r1, D2r0, D1d)
            CrossAttnUpBlock2D(block_out_channels[3], block_out_channels[3], block_out_channels[2], temb_channels,
                               attention_head_dim, block_out_channels[2] // attention_head_dim, cross_attention_dim,
                               [block_out_channels[3], block_out_channels[3], block_out_channels[1]], add_upsample=True),
            # pops idx5=640, idx4=640, idx3=320  (D1r1, D1r0, D0d)
            CrossAttnUpBlock2D(block_out_channels[2], block_out_channels[2], block_out_channels[1], temb_channels,
                               attention_head_dim, block_out_channels[1] // attention_head_dim, cross_attention_dim,
                               [block_out_channels[1], block_out_channels[1], block_out_channels[0]], add_upsample=True),
            # pops idx2=320, idx1=320, idx0=320  (D0r1, D0r0, conv_in)
            CrossAttnUpBlock2D(block_out_channels[1], block_out_channels[1], block_out_channels[0], temb_channels,
                               attention_head_dim, block_out_channels[0] // attention_head_dim, cross_attention_dim,
                               [block_out_channels[0], block_out_channels[0], block_out_channels[0]], add_upsample=False)
        ])
        
        self.conv_norm_out = nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[0], eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
            
        if timestep.shape[0] == 1:
            timestep = timestep.expand(sample.shape[0])
            
        temb_sin = get_timestep_embedding(timestep, self.conv_in.out_channels)
        temb = self.time_embedding(temb_sin)
        
        x = self.conv_in(sample)
        # conv_in output is the first skip (consumed last by the final up block)
        xs = [x]
        
        for block in self.down_blocks:
            x, intermediate_states = block(x, temb, encoder_hidden_states)
            xs.extend(intermediate_states)
            
        x = self.mid_block(x, temb, encoder_hidden_states)
        
        for block in self.up_blocks:
            x = block(x, xs, temb, encoder_hidden_states)
            
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        
        class UNetOutput:
            def __init__(self, sample):
                self.sample = sample
        return UNetOutput(x)

    @classmethod
    def from_pretrained(cls, model_id, subfolder="unet", torch_dtype=torch.float32, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        from diffusers import UNet2DConditionModel as HFUNet2DConditionModel
        model = cls(**kwargs)
        hf_model = HFUNet2DConditionModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch_dtype)
        model = model.to(dtype=torch_dtype)
        result = model.load_state_dict(hf_model.state_dict(), strict=False)
        if result.missing_keys:
            print(f"[UNet] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]} ...")
        if result.unexpected_keys:
            print(f"[UNet] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]} ...")
        if not result.missing_keys and not result.unexpected_keys:
            print("[UNet] All keys loaded successfully ✅")
        
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
