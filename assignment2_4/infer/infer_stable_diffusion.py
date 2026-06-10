import os
import yaml
import torch
from models.stable_diffusion import StableDiffusion

def infer_stable_diffusion(
    prompts,
    noise,
    text_encoder_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    unet_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    num_steps=50,
    device=None,
    quantization="fp16",  # Options: "fp16", "fp32"
    guidance_scale=7.5    # Classifier-Free Guidance scale
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration parameters from config.yaml if it exists
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    config = None
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f)
            config = raw_config.get("stable_diffusion_loader", {})
        except Exception:
            pass

    # Instantiate StableDiffusion pipeline
    sd = StableDiffusion()
    sd.load_weight(model_id=unet_id, device=device, quantization=quantization, config=config)
    
    # Run forward pass
    return sd(prompts, noise, num_steps=num_steps, device=device, guidance_scale=guidance_scale)
