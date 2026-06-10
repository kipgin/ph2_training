import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPTextModel as HFCLIPTextModel
from diffusers import PNDMScheduler

# Import our custom modules
from models.stable_diffusion import CLIPTextModel, AutoencoderKL, UNet2DConditionModel

class StableDiffusionLoader(nn.Module):
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32, config=None):
        super().__init__()

        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.config = config

        unet_kwargs = {}
        vae_kwargs = {}
        clip_kwargs = {}
        scheduler_kwargs = {}
        if config is not None:
            unet_kwargs = config.get("unet_config", {})
            vae_kwargs = config.get("vae_config", {})
            clip_kwargs = config.get("clip_config", {})
            scheduler_kwargs = config.get("scheduler_config", {})

        # Load tokenizer and image processor from HF
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder='tokenizer')
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_id, subfolder='feature_extractor')

        # Load our custom model sub-modules from pretrained HF checkpoints
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder='text_encoder', torch_dtype=self.torch_dtype, **clip_kwargs)
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder='vae', torch_dtype=self.torch_dtype, **vae_kwargs)
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder='unet', torch_dtype=self.torch_dtype, **unet_kwargs)

        # Support clip_model loaded from CLIPTextModel to maintain backward compatibility
        self.clip_model = HFCLIPTextModel.from_pretrained(self.model_id, subfolder='text_encoder', torch_dtype=self.torch_dtype)

        # Load scheduler from HF config
        self.scheduler = PNDMScheduler.from_pretrained(self.model_id, subfolder='scheduler', **scheduler_kwargs)

    def load_models(self, device):
        # Garbage collect and clear CUDA cache to free up VRAM from any previous runs
        import gc
        gc.collect()
        if "cuda" in str(device):
            torch.cuda.empty_cache()

        self.text_encoder.to(device)
        self.vae.to(device)
        self.unet.to(device)
        self.clip_model.to(device)
        return self.tokenizer, self.text_encoder, self.vae, self.unet, self.image_processor, self.clip_model

    def encode_prompt(self, prompt, device):
        # Ensure prompt is a list of strings
        if isinstance(prompt, str):
            prompt = [prompt]

        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_inputs = text_inputs.to(device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

        # Generate unconditional (negative) embeddings for Classifier-Free Guidance
        uncond_inputs = self.tokenizer(
            [""] * len(prompt),
            padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_inputs.input_ids)[0]

        # Concatenate uncond and cond embeddings into a single batch
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    def sample(self, noise, text_embeddings, num_steps, device, guidance_scale=7.5):
        self.scheduler.set_timesteps(num_steps, device=device)
        latents = noise
        for t in self.scheduler.timesteps:
            # Duplicate latents for CFG
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    # forward method for inference
    def forward(self, noise, prompt, num_steps, device, guidance_scale=7.5):
        self.load_models(device)
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        
        # Use autocast context manager to enable mixed precision during inference
        autocast_ctx = torch.autocast(device_type=device_type, dtype=self.torch_dtype) if device_type == "cuda" else torch.no_grad()
        
        if isinstance(prompt, str):
            prompt = [prompt]

        with autocast_ctx:
            text_embeddings = self.encode_prompt(prompt, device)
            noise = noise.to(dtype=self.torch_dtype)
            latents = self.sample(noise, text_embeddings, num_steps, device, guidance_scale=guidance_scale)
            with torch.no_grad():
                images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return images