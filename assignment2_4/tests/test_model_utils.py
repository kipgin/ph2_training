import os
import tempfile
import torch
import torch.nn as nn
import pytest
import sys
import importlib.util

# Mock torchao check to bypass version incompatibility in peft under older Colab environments
_original_find_spec = importlib.util.find_spec
def _mocked_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _mocked_find_spec

# Add the dreamboth-lora-trainer directory to the python path for importing src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dreamboth-lora-trainer")))

from src.model_utils import inject_lora, save_lora_weights

class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(10, 10)
        self.to_k = nn.Linear(10, 10)
        self.to_v = nn.Linear(10, 10)
        self.to_out = nn.ModuleList([nn.Linear(10, 10)])

class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()
        self.conv = nn.Conv2d(3, 3, 3)

class DummyCLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(10, 10)
        self.k_proj = nn.Linear(10, 10)
        self.v_proj = nn.Linear(10, 10)
        self.out_proj = nn.Linear(10, 10)

class DummyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyCLIPAttention()

def test_inject_lora():
    unet = DummyUNet()
    text_encoder = DummyTextEncoder()
    
    config = {
        "training": {
            "lora_rank": 4,
            "lora_alpha": 4,
            "train_text_encoder": True
        }
    }
    
    # Check that initially everything is trainable
    assert all(p.requires_grad for p in unet.parameters())
    
    # Inject LoRA
    unet, text_encoder = inject_lora(unet, text_encoder, config)
    
    # Check that base parameters are frozen and only LoRA parameters have requires_grad=True
    for name, param in unet.named_parameters():
        if "lora" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad
            
    for name, param in text_encoder.named_parameters():
        if "lora" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad

def test_save_lora_weights():
    unet = DummyUNet()
    text_encoder = DummyTextEncoder()
    config = {
        "training": {
            "lora_rank": 4,
            "lora_alpha": 4,
            "train_text_encoder": True
        }
    }
    
    unet, text_encoder = inject_lora(unet, text_encoder, config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_lora_weights(tmpdir, unet, text_encoder, step=100, is_final=False)
        checkpoint_file = os.path.join(tmpdir, "checkpoint-100", "pytorch_lora_weights.bin")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(tmpdir, "checkpoint-100", "pytorch_lora_weights.safetensors")
        assert os.path.exists(checkpoint_file)
        
        save_lora_weights(tmpdir, unet, text_encoder, step=100, is_final=True)
        final_file = os.path.join(tmpdir, "final_lora_weights.bin")
        if not os.path.exists(final_file):
            final_file = os.path.join(tmpdir, "final_lora_weights.safetensors")
        assert os.path.exists(final_file)
