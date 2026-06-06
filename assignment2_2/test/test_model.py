import os
import sys
import unittest
import torch

# Add parent directory to sys.path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet, SinusoidalPositionEmbeddings, ResnetBlock, AttentionBlock
from models.ddpm import DDPM

class TestDDPMModel(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.time_emb_dim = 128
        self.hidden_dims = [32, 64, 128, 128]
        self.batch_size = 4
        self.image_size = 32
        
        self.unet = UNet(
            in_channels=self.in_channels,
            time_emb_dim=self.time_emb_dim,
            hidden_dims=self.hidden_dims
        )
        
        self.ddpm = DDPM(
            unet=self.unet,
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_name='linear'
        )
        
    def test_time_embeddings_shape(self):
        print("\n--- Auditing Sinusoidal Position Embeddings Shape ---")
        timesteps = torch.randint(0, 100, (self.batch_size,))
        embedder = SinusoidalPositionEmbeddings(dim=self.time_emb_dim)
        embeddings = embedder(timesteps)
        self.assertEqual(embeddings.shape, (self.batch_size, self.time_emb_dim))
        print(f"Timesteps: {timesteps.tolist()} -> Embedding shape: {embeddings.shape}")
        
    def test_resnet_block_shape(self):
        print("\n--- Auditing ResnetBlock Shape ---")
        # Test shape with and without time step embeddings
        x = torch.randn(self.batch_size, 32, 16, 16)
        time_emb = torch.randn(self.batch_size, self.time_emb_dim)
        
        block_time = ResnetBlock(in_ch=32, out_ch=64, time_emb_dim=self.time_emb_dim)
        out_time = block_time(x, time_emb)
        self.assertEqual(out_time.shape, (self.batch_size, 64, 16, 16))
        print(f"ResnetBlock with time embed output shape: {out_time.shape}")
        
        block_no_time = ResnetBlock(in_ch=32, out_ch=32)
        out_no_time = block_no_time(x)
        self.assertEqual(out_no_time.shape, (self.batch_size, 32, 16, 16))
        print(f"ResnetBlock without time embed output shape: {out_no_time.shape}")
        
    def test_attention_block_shape(self):
        print("\n--- Auditing AttentionBlock Shape ---")
        x = torch.randn(self.batch_size, 64, 8, 8)
        attn_block = AttentionBlock(channels=64)
        out = attn_block(x)
        self.assertEqual(out.shape, x.shape)
        print(f"AttentionBlock input shape {x.shape} -> output shape: {out.shape}")
        
    def test_unet_layer_by_layer_shapes(self):
        print("\n=== RUNNING U-NET LAYER-BY-LAYER SHAPE AUDIT ===")
        x = torch.randn(self.batch_size, self.in_channels, self.image_size, self.image_size)
        t = torch.randint(0, 100, (self.batch_size,))
        
        # 1. Test initial projection
        t_proj = self.unet.time_mlp(t)
        self.assertEqual(t_proj.shape, (self.batch_size, self.time_emb_dim))
        print(f"Time MLP Projection shape: {t_proj.shape}")
        
        x_init = self.unet.init_conv(x)
        self.assertEqual(x_init.shape, (self.batch_size, self.hidden_dims[0], self.image_size, self.image_size))
        print(f"Initial Conv shape: {x_init.shape}")
        
        # 2. Test Down paths
        skip_connections = []
        current_x = x_init
        print("\nUNet Downsampling activation sequence:")
        for idx, (block1, block2, attn, down) in enumerate(self.unet.downs):
            current_x = block1(current_x, t_proj)
            current_x = block2(current_x, t_proj)
            current_x = attn(current_x)
            skip_connections.append(current_x)
            current_x = down(current_x)
            expected_ch = self.hidden_dims[idx + 1]
            expected_hw = self.image_size // (2 ** (idx + 1))
            self.assertEqual(current_x.shape, (self.batch_size, expected_ch, expected_hw, expected_hw))
            print(f"  - Stage {idx+1} Output shape: {current_x.shape}")
            
        # 3. Test Bottleneck
        x_mid = self.unet.mid_block1(current_x, t_proj)
        x_mid = self.unet.mid_attn(x_mid)
        x_mid = self.unet.mid_block2(x_mid, t_proj)
        self.assertEqual(x_mid.shape, current_x.shape)
        print(f"\nUNet Bottleneck Output shape: {x_mid.shape}")
        
        # 4. Test Up paths
        current_up_x = x_mid
        print("\nUNet Upsampling activation sequence:")
        for idx, (upsample, block1, block2, attn) in enumerate(self.unet.ups):
            current_up_x = upsample(current_up_x)
            skip = skip_connections.pop()
            current_up_x = torch.cat((current_up_x, skip), dim=1)
            current_up_x = block1(current_up_x, t_proj)
            current_up_x = block2(current_up_x, t_proj)
            current_up_x = attn(current_up_x)
            expected_ch = self.hidden_dims[::-1][idx + 1]
            expected_hw = expected_hw = self.image_size // (2 ** (len(self.hidden_dims) - 2 - idx))
            self.assertEqual(current_up_x.shape, (self.batch_size, expected_ch, expected_hw, expected_hw))
            print(f"  - Stage {idx+1} Output shape: {current_up_x.shape}")
            
        # 5. Test final projection
        out = self.unet.final_conv(current_up_x)
        self.assertEqual(out.shape, x.shape)
        print(f"\nFinal UNet projection output shape: {out.shape}")
        
    def test_ddpm_diffusion_loop_shapes(self):
        print("\n=== RUNNING DDPM DIFFUSION SHAPES TEST ===")
        x = torch.randn(self.batch_size, self.in_channels, self.image_size, self.image_size)
        t = torch.randint(0, 100, (self.batch_size,))
        noise = torch.randn_like(x)
        
        # 1. q_sample forward diffusion
        x_noisy = self.ddpm.q_sample(x, t, noise)
        self.assertEqual(x_noisy.shape, x.shape)
        print(f"Forward diffusion q_sample output shape: {x_noisy.shape}")
        
        # 2. Training forward loss computation
        loss = self.ddpm(x)
        self.assertEqual(loss.dim(), 0)  # Loss should be a scalar
        print(f"Model forward training loss: {loss.item():.4f}")
        
        # 3. p_sample reverse step
        x_denoised = self.ddpm.p_sample(x_noisy, t, t_index=50)
        self.assertEqual(x_denoised.shape, x.shape)
        print(f"Reverse denoising p_sample output shape: {x_denoised.shape}")

if __name__ == '__main__':
    unittest.main()
