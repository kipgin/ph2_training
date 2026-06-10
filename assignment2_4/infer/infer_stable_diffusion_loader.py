import torch
import yaml
from diffusers import PNDMScheduler

from models.stable_diffusion_loader import StableDiffusionLoader

def infer_stable_diffusion_loader(config_path, prompts):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default to fp16 for CUDA to prevent OOMs, and fp32 for CPU
    loader_config = config.get('stable_diffusion_loader', {})
    quantization = loader_config.get('quantization', 'fp16' if 'cuda' in str(device) else 'fp32')
    dtype = torch.float16 if (quantization == 'fp16' and 'cuda' in str(device)) else torch.float32

    model = StableDiffusionLoader(
        model_id=loader_config.get('model_id', "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        torch_dtype=dtype,
        config=loader_config
    )

    batch_size = len(prompts)

    # create random noise for each prompt in the correct dtype and device
    noise = torch.randn(
        batch_size, 
        model.unet.config.in_channels, 
        model.unet.config.sample_size, 
        model.unet.config.sample_size, 
        device=device,
        dtype=dtype
    )

    num_steps = loader_config.get('num_steps', 50)
    guidance_scale = loader_config.get('guidance_scale', 7.5)

    generated_images = model.forward(noise, prompts, num_steps, device, guidance_scale=guidance_scale)
    
    return generated_images