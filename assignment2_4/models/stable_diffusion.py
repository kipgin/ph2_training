import torch
import torch.nn as nn

# Re-expose all symbols for backward compatibility with existing tests and notebooks
from .embeddings import get_timestep_embedding, TimeEmbedding, CLIPTextEmbeddings
from .resnet import ResnetBlock2D
from .attention import (
    Attention, GEGLU, FeedForward, BasicTransformerBlock, Transformer2DModel,
    CLIPAttention, CLIPMLP, CLIPEncoderLayer, CLIPEncoder, VAEAttention
)
from .unet import (
    UNet2DConditionModel, CrossAttnDownBlock2D, DownBlock2D,
    UNetMidBlock2DModelCrossAttn, CrossAttnUpBlock2D, UpBlock2D,
    Downsample2D, Upsample2D
)
from .vae import (
    AutoencoderKL, Encoder, Decoder, VAEDownsample2D,
    DownEncoderBlock2D, UpDecoderBlock2D, DiagonalGaussianDistribution
)
from .clip import CLIPTextModel


class StableDiffusion(nn.Module):
    def __init__(self, unet=None, vae=None, text_encoder=None, tokenizer=None, scheduler=None, torch_dtype=torch.float32):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.torch_dtype = torch_dtype

    def load_weight(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device=None, quantization="fp16", config=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Set up data type for quantization
        self.torch_dtype = torch.float16 if (quantization == "fp16" and "cuda" in str(device)) else torch.float32
        
        import gc
        gc.collect()
        if "cuda" in str(device):
            torch.cuda.empty_cache()
            
        unet_kwargs = {}
        vae_kwargs = {}
        clip_kwargs = {}
        scheduler_kwargs = {}
        if config is not None:
            unet_kwargs = config.get("unet_config", {})
            vae_kwargs = config.get("vae_config", {})
            clip_kwargs = config.get("clip_config", {})
            scheduler_kwargs = config.get("scheduler_config", {})
            
        from transformers import CLIPTokenizer
        from diffusers import PNDMScheduler
        
        # Load tokenizer and scheduler
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder='scheduler', **scheduler_kwargs)
        
        # Load weights into submodules (or instantiate them if None)
        if self.text_encoder is None:
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder', torch_dtype=self.torch_dtype, **clip_kwargs).to(device)
        else:
            from transformers import CLIPTextModel as HFCLIPTextModel
            hf_clip = HFCLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self.torch_dtype)
            self.text_encoder.to(device).to(dtype=self.torch_dtype)
            result = self.text_encoder.load_state_dict(hf_clip.state_dict(), strict=False)
            missing_keys = [k for k in result.missing_keys if k != "embeddings.position_ids"]
            if missing_keys:
                print(f"[CLIP] Missing keys ({len(missing_keys)}): {missing_keys[:5]} ...")
            if result.unexpected_keys:
                print(f"[CLIP] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]} ...")
            if not missing_keys and not result.unexpected_keys:
                print("[CLIP] All keys loaded successfully ✅")
            del hf_clip
            gc.collect()
            if "cuda" in str(device):
                torch.cuda.empty_cache()
            
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=self.torch_dtype, **vae_kwargs).to(device)
        else:
            from diffusers import AutoencoderKL as HFAutoencoderKL
            hf_vae = HFAutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.torch_dtype)
            self.vae.to(device).to(dtype=self.torch_dtype)
            result = self.vae.load_state_dict(hf_vae.state_dict(), strict=False)
            if result.missing_keys:
                print(f"[VAE] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]} ...")
            if result.unexpected_keys:
                print(f"[VAE] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]} ...")
            if not result.missing_keys and not result.unexpected_keys:
                print("[VAE] All keys loaded successfully ✅")
            del hf_vae
            gc.collect()
            if "cuda" in str(device):
                torch.cuda.empty_cache()
            
        if self.unet is None:
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet', torch_dtype=self.torch_dtype, **unet_kwargs).to(device)
        else:
            from diffusers import UNet2DConditionModel as HFUNet2DConditionModel
            hf_unet = HFUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.torch_dtype)
            self.unet.to(device).to(dtype=self.torch_dtype)
            result = self.unet.load_state_dict(hf_unet.state_dict(), strict=False)
            if result.missing_keys:
                print(f"[UNet] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]} ...")
            if result.unexpected_keys:
                print(f"[UNet] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]} ...")
            if not result.missing_keys and not result.unexpected_keys:
                print("[UNet] All keys loaded successfully ✅")
            del hf_unet
            gc.collect()
            if "cuda" in str(device):
                torch.cuda.empty_cache()
            
        gc.collect()
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    def forward(self, prompts, noise, num_steps=50, device=None, guidance_scale=7.5):
        if device is None:
            if self.unet is not None:
                device = next(self.unet.parameters()).device
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
        if isinstance(prompts, str):
            prompts = [prompts]
            
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        autocast_ctx = torch.autocast(device_type=device_type, dtype=self.torch_dtype) if device_type == "cuda" else torch.no_grad()
        
        with autocast_ctx:
            # 1. Tokenize and encode prompts
            text_inputs = self.tokenizer(
                prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                
            # Generate unconditional (negative) embeddings for Classifier-Free Guidance
            uncond_inputs = self.tokenizer(
                [""] * len(prompts),
                padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                uncond_embeddings = self.text_encoder(uncond_inputs.input_ids)[0]
                
            # Concatenate uncond and cond embeddings into a single batch
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # 2. Denoising loop
            self.scheduler.set_timesteps(num_steps, device=device)
            # Scale initial noise by scheduler's sigma (required for PNDM/DDIM schedulers)
            latents = noise.to(device, dtype=self.torch_dtype) * self.scheduler.init_noise_sigma
            
            for t in self.scheduler.timesteps:
                # Duplicate latents for CFG
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                # Scale model input according to current timestep (required by scheduler)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
            # 3. Decode latents
            with torch.no_grad():
                images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
                
        return images
