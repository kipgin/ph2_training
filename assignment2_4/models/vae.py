import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResnetBlock2D
from .attention import VAEAttention
from diffusers import AutoencoderKL as HFAutoencoderKL

class VAEDownsample2D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # Manual asymmetric padding to match diffusers behavior
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels, temb_channels=None),
            ResnetBlock2D(out_channels, out_channels, temb_channels=None)
        ])
        if add_downsample:
            self.downsamplers = nn.ModuleList([VAEDownsample2D(out_channels, out_channels)])
        else:
            self.downsamplers = None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, add_upsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels, temb_channels=None),
            ResnetBlock2D(out_channels, out_channels, temb_channels=None),
            ResnetBlock2D(out_channels, out_channels, temb_channels=None)
        ])
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])
        else:
            self.upsamplers = None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class VAEMidBlock2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, in_channels, temb_channels=None),
            ResnetBlock2D(in_channels, in_channels, temb_channels=None)
        ])
        self.attentions = nn.ModuleList([
            VAEAttention(in_channels)
        ])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, block_out_channels=[128, 256, 512, 512]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList([
            DownEncoderBlock2D(block_out_channels[0], block_out_channels[0], add_downsample=True),
            DownEncoderBlock2D(block_out_channels[0], block_out_channels[1], add_downsample=True),
            DownEncoderBlock2D(block_out_channels[1], block_out_channels[2], add_downsample=True),
            DownEncoderBlock2D(block_out_channels[2], block_out_channels[3], add_downsample=False)
        ])
        
        self.mid_block = VAEMidBlock2D(block_out_channels[-1])
        
        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out_channels[-1], eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_channels=4, block_out_channels=[128, 256, 512, 512]):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=3, padding=1)
        
        self.mid_block = VAEMidBlock2D(block_out_channels[-1])
        
        self.up_blocks = nn.ModuleList([
            UpDecoderBlock2D(block_out_channels[3], block_out_channels[3], add_upsample=True),
            UpDecoderBlock2D(block_out_channels[3], block_out_channels[2], add_upsample=True),
            UpDecoderBlock2D(block_out_channels[2], block_out_channels[1], add_upsample=True),
            UpDecoderBlock2D(block_out_channels[1], block_out_channels[0], add_upsample=False)
        ])
        
        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out_channels[0], eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, generator=None):
        device = self.parameters.device
        noise = torch.randn(self.mean.shape, device=device, dtype=self.mean.dtype)
        x = self.mean + self.std * noise
        return x


class AutoencoderKLOutput:
    def __init__(self, latent_dist):
        self.latent_dist = latent_dist


class DecoderOutput:
    def __init__(self, sample):
        self.sample = sample


class AutoencoderKL(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_channels=4, block_out_channels=[128, 256, 512, 512], scaling_factor=0.18215):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, block_out_channels)
        self.decoder = Decoder(out_channels, latent_channels, block_out_channels)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        
        # Add config attribute to mimic diffusers
        class VAEConfig:
            def __init__(self, scaling_factor=0.18215):
                self.scaling_factor = scaling_factor
        self.config = VAEConfig(scaling_factor=scaling_factor)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return DecoderOutput(sample=dec)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mean
        dec = self.decode(z).sample
        return dec

    @classmethod
    def from_pretrained(cls, model_id, subfolder="vae", torch_dtype=torch.float32, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # from diffusers import AutoencoderKL as HFAutoencoderKL
        model = cls(**kwargs)
        hf_model = HFAutoencoderKL.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch_dtype)
        model = model.to(dtype=torch_dtype)
        result = model.load_state_dict(hf_model.state_dict(), strict=False)
        if result.missing_keys:
            print(f"[VAE] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]} ...")
        if result.unexpected_keys:
            print(f"[VAE] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]} ...")
        if not result.missing_keys and not result.unexpected_keys:
            print("[VAE] All keys loaded successfully ✅")
        
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
