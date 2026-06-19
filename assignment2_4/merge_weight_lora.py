import sys
import os
import yaml
import torch
import types

# Add the trainer directory and current directory to sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("dreamboth-lora-trainer"))

from models.stable_diffusion import StableDiffusion
from src.model_utils import inject_lora
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline
# from safetensors.torch import save_file
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the custom model layers and base weights
sd = StableDiffusion()
sd.load_weight(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device="cpu", quantization="fp32")

# 2. Inject LoRA layers
with open("dreamboth-lora-trainer/training_config.yaml", "r") as f:
    config = yaml.safe_load(f)
sd.unet, sd.text_encoder = inject_lora(sd.unet, sd.text_encoder, config)

# 3. Load your fine-tuned LoRA weights
lora_path = "checkpoints/final_lora_weights.safetensors"
if os.path.exists(lora_path):
    from safetensors.torch import load_file
    state_dict = load_file(lora_path)

    unet_state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k.startswith("unet.")}
    sd.unet.load_state_dict(unet_state_dict, strict=False)

    text_state_dict = {k.replace("text_encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.")}
    if len(text_state_dict) > 0 and sd.text_encoder is not None:
        sd.text_encoder.load_state_dict(text_state_dict, strict=False)

# 4. Monkey-patch config objects so PEFT's merge_and_unload can query them like dictionaries
def config_get(self, key, default=None):
    return getattr(self, key, default)

for model in [sd.unet, sd.text_encoder]:
    if model is not None:
        # Resolve target model from PeftModel if wrapped
        base_model = model.base_model.model if hasattr(model, "base_model") else model
        if hasattr(base_model, "config") and not hasattr(base_model.config, "get"):
            base_model.config.get = types.MethodType(config_get, base_model.config)

# 5. Merge LoRA weights into the base model parameters (collapses LoRA back into base layers if wrapped)
if hasattr(sd.unet, "merge_and_unload"):
    sd.unet = sd.unet.merge_and_unload()
if sd.text_encoder is not None and hasattr(sd.text_encoder, "merge_and_unload"):
    sd.text_encoder = sd.text_encoder.merge_and_unload()


# # 6. Extract the merged state dict for standard SD1.5 format saving
# merged_state_dict = {}
# # Extract UNet weights
# for k, v in sd.unet.state_dict().items():
#     merged_state_dict[f"model.diffusion_model.{k}"] = v
# # Extract VAE weights
# for k, v in sd.vae.state_dict().items():
#     merged_state_dict[f"first_stage_model.{k}"] = v
# # Extract Text Encoder weights
# for k, v in sd.text_encoder.state_dict().items():
#     merged_state_dict[f"cond_stage_model.transformer.text_model.{k}"] = v

# # import os
# from safetensors.torch import save_file

# drive_checkpoints_dir = "/content/drive/MyDrive/ComfyUI/models/checkpoints"

# os.makedirs(drive_checkpoints_dir, exist_ok=True)

# output_checkpoint = os.path.join(drive_checkpoints_dir, "my_lora_merged_model.safetensors")

# # 4. Lưu file
# save_file(merged_state_dict, output_checkpoint)
# print(f"Successfully exported standard checkpoint to: {output_checkpoint}")
pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# 2. Swap in your newly merged UNet and Text Encoder 
#    (Note: You only need to load your LoRA's state dicts as you've already done)
pipeline.unet.load_state_dict(sd.unet.state_dict())
pipeline.text_encoder.load_state_dict(sd.text_encoder.state_dict())

# 3. Export to a ComfyUI-compatible single checkpoint
drive_checkpoints_dir = "/content/drive/MyDrive/ComfyUI/models/checkpoints"

os.makedirs(drive_checkpoints_dir, exist_ok=True)

output_checkpoint = os.path.join(drive_checkpoints_dir, "my_lora_merged_model.safetensors")

# 4. Lưu file
# save_file(merged_state_dict, output_checkpoint)
print(f"Successfully exported standard checkpoint to: {output_checkpoint}")
pipeline.save_single_file(output_checkpoint)
print(f"Successfully exported standard checkpoint to: {output_checkpoint}")