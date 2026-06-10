import pytest
import torch
from models import (
    TimeEmbedding,
    ResnetBlock2D,
    Attention,
    Transformer2DModel,
    CLIPTextModel,
    AutoencoderKL,
    UNet2DConditionModel,
    get_timestep_embedding,
    StableDiffusion
)

def test_timestep_embedding():
    timesteps = torch.tensor([1, 10, 50], dtype=torch.long)
    emb = get_timestep_embedding(timesteps, 320)
    assert emb.shape == (3, 320)

def test_time_embedding():
    model = TimeEmbedding(time_emb_dim=320, out_dim=1280)
    x = torch.randn(2, 320)
    out = model(x)
    assert out.shape == (2, 1280)

def test_resnet_block_2d():
    # Test without temb
    block_no_temb = ResnetBlock2D(in_channels=64, out_channels=128, temb_channels=None)
    x = torch.randn(2, 64, 16, 16)
    out = block_no_temb(x)
    assert out.shape == (2, 128, 16, 16)

    # Test with temb
    block_with_temb = ResnetBlock2D(in_channels=128, out_channels=128, temb_channels=1280)
    x = torch.randn(2, 128, 16, 16)
    temb = torch.randn(2, 1280)
    out = block_with_temb(x, temb)
    assert out.shape == (2, 128, 16, 16)

def test_attention():
    attn = Attention(query_dim=64, context_dim=128, heads=4, dim_head=16)
    x = torch.randn(2, 10, 64)
    context = torch.randn(2, 5, 128)
    out = attn(x, context=context)
    assert out.shape == (2, 10, 64)

def test_transformer_2d_model():
    model = Transformer2DModel(in_channels=320, n_heads=8, d_head=40, context_dim=768)
    x = torch.randn(2, 320, 8, 8)
    context = torch.randn(2, 77, 768)
    out = model(x, context=context)
    assert out.shape == (2, 320, 8, 8)

def test_clip_text_model():
    model = CLIPTextModel()
    input_ids = torch.randint(0, 49408, (2, 77), dtype=torch.long)
    outputs = model(input_ids)
    assert isinstance(outputs, tuple)
    last_hidden_state = outputs[0]
    assert last_hidden_state.shape == (2, 77, 768)

def test_autoencoder_kl():
    model = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=[64, 128, 256, 256])
    x = torch.randn(2, 3, 64, 64)
    
    # Test encode
    posterior = model.encode(x).latent_dist
    moments = posterior.parameters
    assert moments.shape == (2, 8, 8, 8)
    
    # Test sample
    latent = posterior.sample()
    assert latent.shape == (2, 4, 8, 8)
    
    # Test decode
    decoded = model.decode(latent).sample
    assert decoded.shape == (2, 3, 64, 64)

def test_unet_2d_condition_model():
    model = UNet2DConditionModel(in_channels=4, out_channels=4, sample_size=64, cross_attention_dim=768, attention_head_dim=8)
    sample = torch.randn(2, 4, 32, 32)
    timestep = torch.tensor([10, 20])
    encoder_hidden_states = torch.randn(2, 77, 768)
    
    out = model(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample
    assert out.shape == (2, 4, 32, 32)


def test_model_weight_keys():
    from models import UNet2DConditionModel, AutoencoderKL, CLIPTextModel
    from diffusers import UNet2DConditionModel as HFUNet2DConditionModel, AutoencoderKL as HFAutoencoderKL
    from transformers import CLIPTextModel as HFCLIPTextModel
    
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    
    # 1. UNet keys verification
    unet = UNet2DConditionModel()
    hf_unet = HFUNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    res_unet = unet.load_state_dict(hf_unet.state_dict(), strict=False)
    print(f"UNet Missing: {len(res_unet.missing_keys)}, Unexpected: {len(res_unet.unexpected_keys)}")
    assert len(res_unet.missing_keys) == 0, f"UNet missing: {res_unet.missing_keys}"
    assert len(res_unet.unexpected_keys) == 0, f"UNet unexpected: {res_unet.unexpected_keys}"

    # 2. VAE keys verification
    vae = AutoencoderKL()
    hf_vae = HFAutoencoderKL.from_pretrained(model_id, subfolder="vae")
    res_vae = vae.load_state_dict(hf_vae.state_dict(), strict=False)
    print(f"VAE Missing: {len(res_vae.missing_keys)}, Unexpected: {len(res_vae.unexpected_keys)}")
    assert len(res_vae.missing_keys) == 0, f"VAE missing: {res_vae.missing_keys}"
    assert len(res_vae.unexpected_keys) == 0, f"VAE unexpected: {res_vae.unexpected_keys}"

    # 3. CLIP keys verification
    clip = CLIPTextModel()
    hf_clip = HFCLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    res_clip = clip.load_state_dict(hf_clip.state_dict(), strict=False)
    missing_clip = [k for k in res_clip.missing_keys if k != "embeddings.position_ids"]
    print(f"CLIP Missing (excl position_ids): {len(missing_clip)}, Unexpected: {len(res_clip.unexpected_keys)}")
    assert len(missing_clip) == 0, f"CLIP missing: {missing_clip}"
    assert len(res_clip.unexpected_keys) == 0, f"CLIP unexpected: {res_clip.unexpected_keys}"


def test_stable_diffusion():
    sd = StableDiffusion()
    assert sd.unet is None
    assert sd.vae is None
    assert sd.text_encoder is None
    assert sd.tokenizer is None
    assert sd.scheduler is None
    assert sd.torch_dtype == torch.float32


