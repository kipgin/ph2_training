import os
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

from src.pipeline import TrainingPipeline

class MockVAEConfig:
    def __init__(self):
        self.scaling_factor = 0.18215

class MockLatentDist:
    def __init__(self, shape):
        self.shape = shape
    def sample(self):
        return torch.randn(self.shape)

class MockVAEOutput:
    def __init__(self, shape):
        self.latent_dist = MockLatentDist(shape)

class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockVAEConfig()
        self.param = nn.Parameter(torch.randn(1))
    def encode(self, x):
        bsz = x.shape[0]
        # output shape is (bsz, 4, x_height // 8, x_width // 8)
        latent_shape = (bsz, 4, x.shape[2] // 8, x.shape[3] // 8)
        return MockVAEOutput(latent_shape)

class MockTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
    def forward(self, input_ids):
        # output is a tuple with (hidden_states,)
        bsz, seq_len = input_ids.shape
        hidden_states = torch.randn(bsz, seq_len, 8)
        return (hidden_states,)

class MockUNetOutput:
    def __init__(self, sample):
        self.sample = sample

class MockUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
    def forward(self, sample, timestep, encoder_hidden_states=None):
        return MockUNetOutput(torch.randn_like(sample))

class MockSchedulerConfig:
    def __init__(self):
        self.num_train_timesteps = 1000

class MockScheduler:
    def __init__(self):
        self.config = MockSchedulerConfig()
    def add_noise(self, latents, noise, timesteps):
        return latents + noise

def test_training_pipeline_step():
    vae = MockVAE()
    text_encoder = MockTextEncoder()
    unet = MockUNet()
    noise_scheduler = MockScheduler()
    
    # Test without prior preservation
    pipeline = TrainingPipeline(
        accelerator=None,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        optimizer=None,
        lr_scheduler=None,
        with_prior_preservation=False
    )
    
    batch = {
        "pixel_values": torch.randn(2, 3, 32, 32),
        "input_ids": torch.randint(0, 100, (2, 10))
    }
    
    loss = pipeline.training_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar tensor
    
    # Test with prior preservation
    pipeline_prior = TrainingPipeline(
        accelerator=None,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        optimizer=None,
        lr_scheduler=None,
        with_prior_preservation=True,
        prior_loss_weight=1.0
    )
    
    batch_prior = {
        "pixel_values": torch.randn(4, 3, 32, 32), # 2 instance + 2 class stacked
        "input_ids": torch.randint(0, 100, (4, 10))
    }
    
    loss_prior = pipeline_prior.training_step(batch_prior)
    assert isinstance(loss_prior, torch.Tensor)
    assert loss_prior.ndim == 0 # Scalar tensor
