import os
import sys
import importlib.util
import torch

# ---------------------------------------------------------------------------
# Torchao compatibility shim
# PEFT's LoRA dispatcher calls is_torchao_available() → importlib.util.find_spec("torchao").
# If torchao is installed but its version is below PEFT's minimum (0.16.0), PEFT raises
# an ImportError that aborts inject_lora. Masking torchao here tells PEFT the package is
# absent so it skips the torchao dispatcher and falls through to the standard nn.Linear one.
# This must live in model_utils (not only in train.py) so it is active whenever
# inject_lora is imported directly — e.g. from a notebook cell.
# ---------------------------------------------------------------------------
_original_find_spec = importlib.util.find_spec
def _mocked_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _mocked_find_spec

# `models/` lives at the workspace root (two levels above this src/ directory).
# Insert it into sys.path so the import works regardless of the launch CWD.
_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from models import StableDiffusion

def load_base_models(model_id, weight_dtype):
    """
    Loads frozen components (tokenizer, text_encoder, vae, unet) from our custom implementation.
    Crucially, sets .requires_grad_(False) on all base components and switches them to the target precision.
    """
    sd = StableDiffusion()
    quantization = "fp16" if weight_dtype == torch.float16 else "fp32"
    sd.load_weight(model_id=model_id, device="cpu", quantization=quantization)
    
    tokenizer = sd.tokenizer
    text_encoder = sd.text_encoder
    vae = sd.vae
    unet = sd.unet

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    return tokenizer, text_encoder, vae, unet

def inject_lora(unet, text_encoder, config):
    """
    Attaches trainable LoRA adapters onto the attention layers (typically Q, K, V, O projections)
    of the U-Net and optionally the text encoder. Returns the modified models.
    """
    from peft import LoraConfig, get_peft_model

    lora_rank = config["training"].get("lora_rank", 8)
    lora_alpha = config["training"].get("lora_alpha", 8)
    
    # Configure LoRA for U-Net (targets custom attention linear layers)
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
        init_lora_weights="gaussian"  # Paper-correct: lora_A ~ N(0,sigma), lora_B = 0
    )
    unet = get_peft_model(unet, unet_lora_config)
    
    # Check if we should train the text encoder (optionally)
    train_text_encoder = config["training"].get("train_text_encoder", False)
    if train_text_encoder and text_encoder is not None:
        text_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian"  # Paper-correct: lora_A ~ N(0,sigma), lora_B = 0
        )
        text_encoder = get_peft_model(text_encoder, text_lora_config)
        
        # Ensure only LoRA parameters are trainable in text encoder
        for name, param in text_encoder.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            
    # Ensure only LoRA parameters in U-Net are trainable
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return unet, text_encoder

def save_lora_weights(output_dir, unet, text_encoder, step, is_final=False):
    """
    Helper to extract and format raw LoRA weights for saving.
    Saves in safetensors format (with a torch fallback).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract state dict for LoRA parameters
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if "lora" in name:
            lora_state_dict[f"unet.{name}"] = param.cpu()
            
    if text_encoder is not None:
        for name, param in text_encoder.named_parameters():
            if "lora" in name:
                lora_state_dict[f"text_encoder.{name}"] = param.cpu()
                
    if is_final:
        save_path = os.path.join(output_dir, "final_lora_weights.safetensors")
    else:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
        
    try:
        from safetensors.torch import save_file
        save_file(lora_state_dict, save_path)
        print(f"Saved LoRA weights to {save_path} using safetensors")
    except ImportError:
        # Fallback to standard torch.save if safetensors is not installed
        if save_path.endswith(".safetensors"):
            save_path = save_path.replace(".safetensors", ".bin")
        torch.save(lora_state_dict, save_path)
        print(f"Saved LoRA weights to {save_path} using torch.save")
