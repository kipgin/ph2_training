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

from models.unet import UNet
from models.ddpm import DDPM
from train.training_ddpm import DDPMTrainer

class TestDDPMTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.in_channels = 3
        self.time_emb_dim = 128
        self.hidden_dims = [32, 64, 128, 128]
        self.latent_dim = 16
        
        self.unet = UNet(
            in_channels=self.in_channels,
            time_emb_dim=self.time_emb_dim,
            hidden_dims=self.hidden_dims
        )
        
        self.model = DDPM(
            unet=self.unet,
            num_timesteps=10,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_name='linear'
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_gradient_backpropagation(self):
        print("\n=== RUNNING DDPM GRADIENT BACKPROPAGATION TEST ===")
        # 1. Setup dummy input batch
        x = torch.randn(2, self.in_channels, 32, 32)
        
        # Zero out existing gradients
        self.model.zero_grad()
        
        # 2. Forward pass (returns loss value)
        loss = self.model(x)
        
        # 3. Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero for unet blocks
        unet_init_conv_grad = self.unet.init_conv.weight.grad
        self.assertIsNotNone(unet_init_conv_grad, "UNet initial conv received no gradients.")
        self.assertNotEqual(unet_init_conv_grad.sum().item(), 0.0, "UNet weight gradients are all zero.")
        
        unet_mid_conv_grad = self.unet.mid_block1.block1.proj.weight.grad
        self.assertIsNotNone(unet_mid_conv_grad, "UNet middle block received no gradients.")
        self.assertNotEqual(unet_mid_conv_grad.sum().item(), 0.0, "UNet bottleneck gradients are all zero.")
        
        print(f"Gradient checks successful. DDPM training loss: {loss.item():.5f}")
        
    def test_weight_update(self):
        print("\n=== RUNNING DDPM WEIGHT UPDATE OPTIMIZATION TEST ===")
        x = torch.randn(2, self.in_channels, 32, 32)
        
        initial_weights = self.unet.init_conv.weight.clone().detach()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        
        # Single step optimization
        optimizer.zero_grad()
        loss = self.model(x)
        loss.backward()
        optimizer.step()
        
        updated_weights = self.unet.init_conv.weight.clone().detach()
        
        # Assert that weight tensor values have changed
        weights_difference = torch.sum(torch.abs(updated_weights - initial_weights)).item()
        self.assertGreater(weights_difference, 0.0, "Model weights did not update after optimizer step.")
        print(f"Weight update verification successful. Total weight diff sum: {weights_difference:.6f}")

    @patch('train.training_ddpm.MNISTDataset')
    @patch('train.training_ddpm.wandb')
    def test_trainer_logging_and_checkpoints(self, mock_wandb, mock_mnist):
        print("\n=== RUNNING DDPM TRAINER LOGGING AND FILE CHECKPOINT TEST ===")
        # 1. Setup mock dataset wrapper returning small dummy loaders
        mock_wrapper = MagicMock()
        mock_wrapper.in_channels = self.in_channels
        
        dummy_batch = (torch.randn(2, self.in_channels, 32, 32), torch.zeros(2))
        dummy_loader = [dummy_batch]
        
        mock_wrapper.get_loaders.return_value = (dummy_loader, dummy_loader, dummy_loader)
        mock_mnist.return_value = mock_wrapper
        
        # Mock wandb run object
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        # 2. Setup mock configurations pointing to temporary folders
        config = {
            'dataset': 'mnist',
            'data_dir': self.temp_dir,
            'batch_size': 2,
            'image_size': 32,
            'num_workers': 0,
            'val_split_ratio': 0.1,
            'model': {
                'time_emb_dim': self.time_emb_dim,
                'in_channels': self.in_channels,
                'hidden_dims': self.hidden_dims
            },
            'diffusion': {
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'num_timesteps': 5,  # Keep it very small for test
                'schedule_name': 'linear'
            },
            'training': {
                'epochs': 1,
                'learning_rate': 0.0002,
                'weight_decay': 0.0,
                'device': 'cuda',
                'checkpoint_dir': os.path.join(self.temp_dir, 'checkpoints'),
                'fid_interval': 1,
                'sample_interval': 1,
                'num_fid_samples': 2,
                'wandb_project': 'DDPM-Testing'
            }
        }
        
        # 3. Initialize Trainer and run training epoch
        trainer = DDPMTrainer(config)
        
        # Mock FID method on wrapper to prevent Inception download during test
        mock_wrapper.calculate_fid.return_value = 22.5
        
        # Run trainer train loop (triggers training, validation, samples, FID, and checkpoint saving)
        trainer.train()
        
        # 4. Verify History Logs
        self.assertEqual(len(trainer.history['train_loss']), 1)
        self.assertEqual(len(trainer.history['val_loss']), 1)
        self.assertEqual(trainer.history['fid_samples'][0], 22.5)
        
        # Check files were created on disk
        history_file = os.path.join(config['training']['checkpoint_dir'], 'history.json')
        self.assertTrue(os.path.exists(history_file), "history.json file was not created.")
        
        with open(history_file, 'r') as f:
            saved_history = json.load(f)
        self.assertIn('train_loss', saved_history)
        self.assertIn('val_loss', saved_history)
        
        # Verify model checkpoints saved
        latest_cp = os.path.join(config['training']['checkpoint_dir'], 'latest_checkpoint.pth')
        self.assertTrue(os.path.exists(latest_cp), "latest_checkpoint.pth was not created.")
        
        # Verify sample grid files exist
        gen_sample_img = os.path.join(config['training']['checkpoint_dir'], 'samples', 'sample_epoch_1.png')
        self.assertTrue(os.path.exists(gen_sample_img), "Sample grid file was not saved.")
        
        # 5. Verify wandb logging calls were made
        self.assertTrue(mock_run.log.called, "wandb.log was never called.")
        self.assertTrue(mock_run.finish.called, "wandb.finish was never called.")
        print("DDPM trainer logging and checkpoint storage verification successful.")

if __name__ == '__main__':
    unittest.main()
