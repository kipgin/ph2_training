import os
import sys
import unittest
import torch

# Add parent directory to sys.path to enable modules and datasets imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.baseline_vae import BaselineVAE

class TestModelClasses(unittest.TestCase):
    def setUp(self):
        # Create standard hyperparameter configurations for testing
        self.in_channels = 3
        self.hidden_dims = [32, 64, 128, 256]
        self.latent_dim = 128
        self.batch_size = 4
        self.image_size = 128
        
        self.model = BaselineVAE(
            in_channels=self.in_channels,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        )
        
    def test_layer_by_layer_shapes(self):
        print("\n=== RUNNING LAYER-BY-LAYER MODEL SHAPE AUDIT ===")
        x = torch.randn(self.batch_size, self.in_channels, self.image_size, self.image_size)
        print(f"Input shape: {x.shape}")
        
        # 1. Test Encoder layer-by-layer
        current_x = x
        print("\nEncoding activation sequence:")
        for idx, block in enumerate(self.model.encoder):
            current_x = block(current_x)
            expected_channels = self.hidden_dims[idx]
            expected_hw = self.image_size // (2 ** (idx + 1))
            self.assertEqual(
                current_x.shape, 
                (self.batch_size, expected_channels, expected_hw, expected_hw),
                f"Encoder block {idx} shape mismatch."
            )
            print(f"  - Block {idx+1} Output shape: {current_x.shape}")
            
        encoder_output = current_x
        
        # 2. Test Flattening and Projection to Latent Space
        flat_x = encoder_output.view(self.batch_size, -1)
        self.assertEqual(flat_x.shape, (self.batch_size, self.model.fc_mu.in_features))
        print(f"\nFlattened representation shape: {flat_x.shape}")
        
        mu = self.model.fc_mu(flat_x)
        logvar = self.model.fc_logvar(flat_x)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        print(f"FC Mu shape: {mu.shape}")
        print(f"FC LogVar shape: {logvar.shape}")
        
        # 3. Test Reparameterization
        z = self.model.reparameterize(mu, logvar)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
        print(f"Latent z shape: {z.shape}")
        
        # 4. Test Decoder Input Projection
        dec_input = self.model.decoder_input(z)
        self.assertEqual(dec_input.shape, (self.batch_size, self.model.fc_mu.in_features))
        print(f"\nDecoder Projection input shape: {dec_input.shape}")
        
        h = int((self.model.fc_mu.in_features // self.hidden_dims[-1]) ** 0.5)
        reshaped_dec_input = dec_input.view(self.batch_size, self.hidden_dims[-1], h, h)
        self.assertEqual(reshaped_dec_input.shape, (self.batch_size, self.hidden_dims[-1], h, h))
        print(f"Reshaped Decoder input shape: {reshaped_dec_input.shape}")
        
        # 5. Test Decoder layer-by-layer
        current_dec_x = reshaped_dec_input
        print("\nDecoding activation sequence:")
        for idx, block in enumerate(self.model.decoder):
            current_dec_x = block(current_dec_x)
            # Check shape sizes during decoder upsampling
            # First len(reversed_hidden_dims)-1 layers map between hidden dimensions,
            # final layer maps to the last hidden dim width (32) and upsamples to 128x128
            print(f"  - Block {idx+1} Output shape: {current_dec_x.shape}")
            
        self.assertEqual(
            current_dec_x.shape, 
            (self.batch_size, self.hidden_dims[0], self.image_size, self.image_size),
            f"Decoder output shape mismatch: {current_dec_x.shape}"
        )
        
        # 6. Test Final Output Layer
        recon_x = self.model.final_layer(current_dec_x)
        self.assertEqual(
            recon_x.shape, 
            (self.batch_size, self.in_channels, self.image_size, self.image_size),
            f"Final layer output shape mismatch: {recon_x.shape}"
        )
        print(f"\nFinal reconstructed shape: {recon_x.shape}")
        
    def test_full_forward_and_sample(self):
        print("\n=== RUNNING FULL FORWARD AND SAMPLE SHAPE TEST ===")
        x = torch.randn(self.batch_size, self.in_channels, self.image_size, self.image_size)
        
        # Full forward pass
        recon_x, mu, logvar = self.model(x)
        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        
        print(f"Forward output shapes: x_recon={recon_x.shape}, mu={mu.shape}, logvar={logvar.shape}")
        
        # Random Sampling from Prior
        num_samples = 8
        samples = self.model.sample(num_samples, current_device='cpu')
        self.assertEqual(samples.shape, (num_samples, self.in_channels, self.image_size, self.image_size))
        print(f"Sample output shape: {samples.shape}")

if __name__ == '__main__':
    unittest.main()
