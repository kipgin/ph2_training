import os
import argparse
import yaml
import torch
import sys
import importlib.util

# Mock torchao check to bypass version incompatibility in peft under older Colab environments
_original_find_spec = importlib.util.find_spec
def _mocked_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _mocked_find_spec

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from PIL import Image

# Ensure the workspace root is in path to import models and src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.stable_diffusion import StableDiffusion
from src.dataset import PromptDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Generate prior regularization images for DreamBooth training.")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path to the training configuration YAML file."
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of prior images to generate (overrides config if specified)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        # Fallback to duplicate file path if needed
        fallback_config = "trainging_config.yaml"
        if os.path.exists(fallback_config):
            args.config = fallback_config
        else:
            raise FileNotFoundError(f"Configuration file {args.config} not found.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_id = config["model"]["pretrained_model_name_or_path"]
    
    class_data_dir = config["data"]["class_data_dir"]
    class_prompt = config["data"]["class_prompt"]
    
    # Check current number of class images to avoid duplicating generation
    os.makedirs(class_data_dir, exist_ok=True)
    existing_images = [
        f for f in os.listdir(class_data_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    num_existing = len(existing_images)
    
    num_to_generate = args.num_images - num_existing
    if num_to_generate <= 0:
        print(f"Found {num_existing} existing prior images in {class_data_dir}. No new images need to be generated.")
        return

    print(f"Generating {num_to_generate} prior regularization images for prompt: '{class_prompt}'...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load custom Stable Diffusion model and weights
    print("Loading custom Stable Diffusion model...")
    sd = StableDiffusion()
    sd.load_weight(
        model_id=model_id,
        device=device,
        quantization="fp16" if device == "cuda" else "fp32"
    )
    print("Base custom model loaded successfully!")
    
    # Prompt Dataset & DataLoader for batching
    dataset = PromptDataset(class_prompt, num_to_generate)
    dataloader = DataLoader(dataset, batch_size=4 if device == "cuda" else 1)

    # Generate images
    for batch in tqdm(dataloader, desc="Generating priors"):
        prompts = batch["prompt"]
        indices = batch["index"]
        batch_size = len(prompts)
        
        # Generate random initial noise (shape: [batch_size, 4, 64, 64])
        noise = torch.randn(batch_size, 4, 64, 64, device=device, dtype=torch_dtype)
        
        with torch.no_grad():
            output_tensors = sd(
                prompts=prompts,
                noise=noise,
                num_steps=30,
                device=device,
                guidance_scale=7.5
            )
            
        for img_tensor, idx in zip(output_tensors, indices):
            # Convert normalized tensor to PIL image
            img_tensor = (img_tensor / 2.0 + 0.5).clamp(0, 1)
            img_numpy = img_tensor.cpu().permute(1, 2, 0).numpy()
            img_numpy = (img_numpy * 255.0).astype("uint8")
            img = Image.fromarray(img_numpy)
            
            image_name = f"prior_{idx + num_existing:04d}.png"
            img.save(os.path.join(class_data_dir, image_name))

    print(f"Prior images generation completed. Saved to {class_data_dir}")

if __name__ == "__main__":
    main()
