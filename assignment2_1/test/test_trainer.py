import os
import sys
import tempfile
import shutil
import unittest
import json
from unittest.mock import patch, MagicMock
import torch

# Add parent directory to sys.path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.baseline_vae import BaselineVAE
from train.training_baseline_vae import VAETrainer

class TestTrainerClasses(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.in_channels = 3
        self.hidden_dims = [32, 64, 128, 256]
        self.latent_dim = 16
        
        self.model = BaselineVAE(
            in_channels=self.in_channels,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_gradient_backpropagation(self):
        print("\n=== RUNNING GRADIENT BACKPROPAGATION TEST ===")
        # 1. Setup dummy input batch (size 2)
        x = torch.randn(2, self.in_channels, 128, 128)
        
        # # 2. Forward pass
        # x_recon, mu, logvar = self.model(x)
        
        # 3. Calculate loss
        loss,recon_loss,kld_loss = self.model.loss(x=x,kld_weight=0.0001)
        
        # Zero out existing gradients
        self.model.zero_grad()
        
        # 4. Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero for encoder and projection layers
        encoder_weight_grad = self.model.encoder[0][0].weight.grad
        self.assertIsNotNone(encoder_weight_grad, "Encoder weights received no gradients.")
        self.assertNotEqual(encoder_weight_grad.sum().item(), 0.0, "Encoder weight gradients are all zero.")
        
        fc_mu_grad = self.model.fc_mu.weight.grad
        self.assertIsNotNone(fc_mu_grad, "fc_mu projection weights received no gradients.")
        self.assertNotEqual(fc_mu_grad.sum().item(), 0.0, "fc_mu gradients are all zero.")
        
        decoder_weight_grad = self.model.decoder[0][0].weight.grad
        self.assertIsNotNone(decoder_weight_grad, "Decoder weights received no gradients.")
        self.assertNotEqual(decoder_weight_grad.sum().item(), 0.0, "Decoder weight gradients are all zero.")
        
        print(f"Gradient checks successful. Loss: {loss.item():.4f}")
        
    def test_weight_update(self):
        print("\n=== RUNNING WEIGHT UPDATE OPTIMIZATION TEST ===")
        x = torch.randn(2, self.in_channels, 128, 128)
        
        # Save a copy of the model projection weights before optimization step
        initial_weights = self.model.fc_mu.weight.clone().detach()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        
        # Single optimization step
        optimizer.zero_grad()
        # x_recon, mu, logvar = self.model(x)
        loss,recon_loss,kld_loss = self.model.loss(x=x,kld_weight=0.0001)
        loss.backward()
        optimizer.step()
        
        # Retrieve optimized weights
        updated_weights = self.model.fc_mu.weight.clone().detach()
        
        # Verify that weights have actually changed
        weights_difference = torch.sum(torch.abs(updated_weights - initial_weights)).item()
        self.assertGreater(weights_difference, 0.0, "Model weights did not update after optimizer step.")
        print(f"Weight update verification successful. Total weight diff sum: {weights_difference:.6f}")

    @patch('train.training_baseline_vae.MNISTDataset')
    @patch('train.training_baseline_vae.wandb')
    def test_trainer_logging_and_checkpoints(self, mock_wandb, mock_mnist):
        print("\n=== RUNNING TRAINER LOGGING AND FILE CHECKPOINT TEST ===")
        # 1. Setup mock dataset wrapper returning small dummy loaders to isolate training logic
        mock_wrapper = MagicMock()
        mock_wrapper.in_channels = self.in_channels
        
        # Create a single batch of shape (2, 3, 128, 128)
        dummy_batch = (torch.randn(2, self.in_channels, 128, 128), torch.zeros(2))
        dummy_loader = [dummy_batch]
        
        mock_wrapper.get_loaders.return_value = (dummy_loader, dummy_loader, dummy_loader)
        mock_mnist.return_value = mock_wrapper
        
        # Mock wandb run object
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        # 2. Setup mock configurations pointing to temporary checkpoint folder
        config = {
            'dataset': 'mnist',
            'data_dir': self.temp_dir,
            'batch_size': 2,
            'image_size': 128,
            'num_workers': 0,
            'val_split_ratio': 0.1,
            'base_vae': {
                'latent_dim': self.latent_dim,
                'in_channels': self.in_channels,
                'hidden_dims': self.hidden_dims
            },
            'training': {
                'epochs': 1,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'device': 'cuda',
                'checkpoint_dir': os.path.join(self.temp_dir, 'checkpoints'),
                'fid_interval': 1,
                'sample_interval': 1,
                'num_fid_samples': 2,
                'kld_weight': 0.001
            }
        }
        
        # 3. Initialize Trainer and run training epoch
        trainer = VAETrainer(config)
        
        # Mock FID method on wrapper to prevent Inception download during test
        mock_wrapper.calculate_fid.return_value = 15.0
        
        # Run trainer train method (which triggers train_epoch, validate, save_samples, run_eval_metrics, save_checkpoint)
        trainer.train()
        
        # 4. Verify History Logs
        self.assertEqual(len(trainer.history['train_loss']), 1)
        self.assertEqual(len(trainer.history['val_loss']), 1)
        self.assertEqual(trainer.history['fid_samples'][0], 15.0)
        self.assertEqual(trainer.history['fid_recons'][0], 15.0)
        
        # Check that files were written to checkpoints dir
        history_file = os.path.join(config['training']['checkpoint_dir'], 'history.json')
        self.assertTrue(os.path.exists(history_file), "history.json file was not created.")
        
        # Read history.json and verify keys
        with open(history_file, 'r') as f:
            saved_history = json.load(f)
        self.assertIn('train_loss', saved_history)
        self.assertIn('val_loss', saved_history)
        
        # Check model checkpoints are saved
        latest_cp = os.path.join(config['training']['checkpoint_dir'], 'latest_checkpoint.pth')
        self.assertTrue(os.path.exists(latest_cp), "latest_checkpoint.pth was not created.")
        
        # Check generated image samples exist
        recon_sample_img = os.path.join(config['training']['checkpoint_dir'], 'samples', 'recon_epoch_1.png')
        gen_sample_img = os.path.join(config['training']['checkpoint_dir'], 'samples', 'sample_epoch_1.png')
        self.assertTrue(os.path.exists(recon_sample_img), "Reconstruction grid file was not saved.")
        self.assertTrue(os.path.exists(gen_sample_img), "Sample grid file was not saved.")
        
        # 5. Verify wandb logging calls were made
        self.assertTrue(mock_run.log.called, "wandb.log was never called.")
        self.assertTrue(mock_run.finish.called, "wandb.finish was never called.")
        print("Trainer logging and checkpoint storage verification successful.")

if __name__ == '__main__':
    unittest.main()
