import os
import tempfile
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
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

# Add paths to make imports work from dreamboth-lora-trainer and workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataset import DreamBoothDataset, DreamBoothCollator
from src.model_utils import inject_lora
from src.pipeline import TrainingPipeline

# 1. Mock tokenizer for dataset testing
class MockTokenizer:
    def __init__(self):
        self.model_max_length = 15

    def __call__(self, text, padding=None, truncation=None, max_length=None):
        class Output(dict):
            def __init__(self):
                super().__init__()
                self.input_ids = [101, 102, 103, 104]
                self["input_ids"] = self.input_ids
        return Output()

    def pad(self, dict_of_ids, padding=None, max_length=None, return_tensors=None):
        ids = dict_of_ids["input_ids"]
        padded = []
        for seq in ids:
            padded_seq = seq + [0] * (max_length - len(seq))
            padded.append(padded_seq)
        class Output(dict):
            def __init__(self, input_ids):
                super().__init__()
                self.input_ids = input_ids
                self["input_ids"] = input_ids
        return Output(torch.tensor(padded, dtype=torch.long))

# 2. Mock modules for model behavior testing
class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.ModuleList([nn.Linear(8, 8)])

    def forward(self, x, context=None):
        q = self.to_q(x)
        k = self.to_k(context if context is not None else x)
        v = self.to_v(context if context is not None else x)
        attn = torch.bmm(q, k.transpose(-1, -2))
        out = torch.bmm(attn, v)
        for layer in self.to_out:
            out = layer(out)
        return out

class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states=None):
        B, C, H, W = sample.shape
        x_attn = torch.cat([sample.mean(dim=(2, 3)), sample.mean(dim=(2, 3))], dim=-1).unsqueeze(1)
        
        context = None
        if encoder_hidden_states is not None:
            context = encoder_hidden_states
            
        attn_out = self.attn(x_attn, context=context)
        attn_val = attn_out.mean()
        
        out = self.conv(sample) + attn_val * 0.001
        
        class UNetOutput:
            def __init__(self, sample):
                self.sample = sample
        return UNetOutput(out)

class DummyCLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)

    def forward(self, hidden_states, attention_mask=None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn = torch.bmm(q, k.transpose(-1, -2))
        out = torch.bmm(attn, v)
        out = self.out_proj(out)
        return out

class DummyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyCLIPAttention()
        self.param = nn.Parameter(torch.randn(1))

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        # Create dummy initial tensor on target device/dtype
        x = torch.randn(bsz, seq_len, 8, dtype=self.param.dtype, device=self.param.device)
        # Pass through the attention block to register text lora parameters in grad graph
        x_out = self.attn(x)
        return (x_out,)

class MockVAEConfig:
    def __init__(self):
        self.scaling_factor = 0.18215

class MockLatentDist:
    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device
    def sample(self):
        return torch.randn(self.shape, dtype=self.dtype, device=self.device)

class MockVAEOutput:
    def __init__(self, shape, dtype, device):
        self.latent_dist = MockLatentDist(shape, dtype, device)

class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockVAEConfig()
        self.param = nn.Parameter(torch.randn(1))
    def encode(self, x):
        bsz = x.shape[0]
        latent_shape = (bsz, 4, x.shape[2] // 8, x.shape[3] // 8)
        return MockVAEOutput(latent_shape, x.dtype, x.device)

class MockSchedulerConfig:
    def __init__(self):
        self.num_train_timesteps = 1000

class MockScheduler:
    def __init__(self):
        self.config = MockSchedulerConfig()
    def add_noise(self, latents, noise, timesteps):
        return latents + noise


def test_data_generation_shape_and_type():
    """
    Tests that DreamBoothDataset and DreamBoothCollator generate
    data with correct shapes, types, and keys.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        instance_dir = os.path.join(tmpdir, "instance")
        class_dir = os.path.join(tmpdir, "class")
        os.makedirs(instance_dir)
        os.makedirs(class_dir)

        # Save dummy images
        for i in range(2):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(instance_dir, f"instance_{i}.png"))
        
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(class_dir, f"class_{i}.png"))

        tokenizer = MockTokenizer()
        
        # Instantiate dataset with prior preservation
        dataset = DreamBoothDataset(
            instance_data_root=instance_dir,
            instance_prompt="a photo of sks person",
            tokenizer=tokenizer,
            class_data_root=class_dir,
            class_prompt="a photo of a person",
            size=32,
            center_crop=True
        )

        assert len(dataset) == 3
        item = dataset[0]
        
        # Verify shape and type of individual items
        assert "instance_images" in item
        assert "class_images" in item
        assert isinstance(item["instance_images"], torch.Tensor)
        assert item["instance_images"].shape == (3, 32, 32)
        assert item["instance_images"].dtype == torch.float32
        
        assert isinstance(item["instance_prompt_ids"], list)
        
        # Verify collator output batch
        collator = DreamBoothCollator(tokenizer=tokenizer, with_prior_preservation=True)
        batch = collator([dataset[0], dataset[1]])
        
        assert "pixel_values" in batch
        assert "input_ids" in batch
        
        # Since with_prior_preservation=True, batch size is doubled: 2 * 2 = 4
        assert batch["pixel_values"].shape == (4, 3, 32, 32)
        assert batch["pixel_values"].dtype == torch.float32
        
        assert batch["input_ids"].shape == (4, 15)
        assert batch["input_ids"].dtype == torch.long


def test_gradient_calculation_and_updates():
    """
    Tests that only LoRA parameters calculate gradients, forward pass outputs
    the correct scalar loss, and optimizer steps update only the LoRA parameters.
    """
    unet = DummyUNet()
    text_encoder = DummyTextEncoder()
    vae = MockVAE()
    noise_scheduler = MockScheduler()
    
    config = {
        "training": {
            "lora_rank": 4,
            "lora_alpha": 4,
            "train_text_encoder": True,
            "learning_rate": 1e-1
        }
    }
    
    # Inject LoRA adapters
    unet, text_encoder = inject_lora(unet, text_encoder, config)
    
    # Store initial state dicts of base and LoRA weights to verify updates
    initial_base_unet_weight = unet.attn.to_q.weight.clone()
    initial_base_text_weight = text_encoder.attn.q_proj.weight.clone()
    
    # Locate LoRA B weights specifically.
    # With paper-correct init (lora_A ~ N(0,sigma), lora_B = 0),
    # only lora_B receives a non-zero gradient:
    #   ∂loss/∂lora_B = grad_output @ (lora_A @ x)^T  ≠ 0
    #   ∂loss/∂lora_A = lora_B^T @ grad_output        = 0  (since lora_B=0)
    # Therefore we assert on lora_B, the matrix that is actually updated.
    lora_unet_param = next(p for n, p in unet.named_parameters() if "lora_B" in n)
    lora_text_param = next(p for n, p in text_encoder.named_parameters() if "lora_B" in n)
    
    initial_lora_unet_weight = lora_unet_param.clone()
    initial_lora_text_weight = lora_text_param.clone()
    
    # Define optimizer over only the trainable parameters
    params_to_optimize = [p for p in unet.parameters() if p.requires_grad] + \
                         [p for p in text_encoder.parameters() if p.requires_grad]
                         
    assert len(params_to_optimize) > 0
    # Ensure base weights are frozen
    assert not unet.attn.to_q.weight.requires_grad
    assert not text_encoder.attn.q_proj.weight.requires_grad
    
    optimizer = optim.SGD(params_to_optimize, lr=0.1)
    
    pipeline = TrainingPipeline(
        accelerator=None,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        lr_scheduler=None,
        with_prior_preservation=True,
        prior_loss_weight=1.0
    )
    
    # Prepare mock batch (2 instance + 2 class stacked = 4)
    batch = {
        "pixel_values": torch.randn(4, 3, 32, 32),
        "input_ids": torch.randint(0, 10, (4, 15), dtype=torch.long)
    }
    
    # Run forward pass
    loss, loss_instance, loss_prior = pipeline.training_step(batch)
    
    # Verify loss output is correct
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar
    
    # Run backward pass
    loss.backward()
    
    # Verify gradients are calculated for LoRA weights, but NOT for base weights
    assert unet.attn.to_q.weight.grad is None
    assert text_encoder.attn.q_proj.weight.grad is None
    assert lora_unet_param.grad is not None
    assert lora_text_param.grad is not None
    
    # Step the optimizer
    optimizer.step()
    
    # Verify updates:
    # 1. Base weights must remain completely unchanged
    assert torch.equal(unet.attn.to_q.weight, initial_base_unet_weight)
    assert torch.equal(text_encoder.attn.q_proj.weight, initial_base_text_weight)
    
    # 2. LoRA weights must have changed (updated)
    assert not torch.equal(lora_unet_param, initial_lora_unet_weight)
    assert not torch.equal(lora_text_param, initial_lora_text_weight)
