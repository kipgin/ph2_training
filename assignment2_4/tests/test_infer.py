import pytest
import torch
import os
import yaml
from infer.infer_stable_diffusion import infer_stable_diffusion
from infer.infer_stable_diffusion_loader import infer_stable_diffusion_loader

def test_infer_signatures():
    # Verify both inference entrypoints exist and are callable
    assert callable(infer_stable_diffusion)
    assert callable(infer_stable_diffusion_loader)

@pytest.mark.skip(reason="Downloads full Stable Diffusion weights (2GB) which is slow for unit testing.")
def test_full_inference_pipeline():
    prompts = ["a beautiful sunset over a lake"]
    # Initial noise of shape (batch_size, channels, height, width)
    # Latent size for SD v1.4 is 64x64
    noise = torch.randn(1, 4, 64, 64)
    
    # We run 2 steps on CPU for validation
    images = infer_stable_diffusion(
        prompts=prompts,
        noise=noise,
        num_steps=2,
        device="cpu"
    )
    # Output image shape should be (1, 3, 512, 512)
    assert images.shape == (1, 3, 512, 512)
