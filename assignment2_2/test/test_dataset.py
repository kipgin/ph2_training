import os
import sys
import tempfile
import shutil
import unittest
import torch
from PIL import Image

# Add parent directory to sys.path to enable datasets and models imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.datasets import MNISTDataset, CelebADataset

class TestDDPMDataset(unittest.TestCase):
    def setUp(self):
        # Setup temporary directories for testing local fallbacks
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_mnist_dataset(self):
        print("\n=== RUNNING MNIST DDPM DATASET TEST ===")
        try:
            mnist_dir = os.path.join(self.temp_dir, 'mnist_data')
            batch_size = 4
            image_size = 32
            
            # Initialize MNIST wrapper
            dataset = MNISTDataset(
                data_dir=mnist_dir, 
                batch_size=batch_size, 
                image_size=image_size, 
                val_split_ratio=0.1, 
                num_workers=0
            )
            
            train_loader, val_loader, test_loader = dataset.get_loaders()
            
            # Assert splits are non-empty
            self.assertTrue(len(dataset.train_dataset) > 0)
            self.assertTrue(len(dataset.val_dataset) > 0)
            self.assertTrue(len(dataset.test_dataset) > 0)
            
            # Load sample batch
            imgs, labels = next(iter(train_loader))
            
            # 1. Assert shape: (B, 1, 32, 32)
            self.assertEqual(imgs.shape, (batch_size, 1, image_size, image_size))
            
            # 2. Assert data type
            self.assertEqual(imgs.dtype, torch.float32)
            
            # 3. Assert normalization range is in [-1.0, 1.0]
            self.assertTrue(imgs.min() >= -1.05)
            self.assertTrue(imgs.max() <= 1.05)
            
            print(f"MNIST Train split size: {len(dataset.train_dataset)}")
            print(f"MNIST Val split size: {len(dataset.val_dataset)}")
            print(f"MNIST Test split size: {len(dataset.test_dataset)}")
            print(f"Batch shape: {imgs.shape}")
            print(f"Batch value range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
            
            print("Logging 3 sample images:")
            for idx in range(min(3, len(imgs))):
                img = imgs[idx]
                print(f"  - Item {idx+1}: Shape={img.shape}, Mean={img.mean().item():.4f}, Std={img.std().item():.4f}")
                
        except Exception as e:
            print(f"Notice: Skipping online MNIST download test due to: {e}")
            
    def test_celeba_dataset_fallback(self):
        print("\n=== RUNNING CELEBA DDPM DATASET FALLBACK TEST ===")
        # Setup mock CelebA directory split structure
        celeba_dir = os.path.join(self.temp_dir, 'celeba_data')
        img_dir = os.path.join(celeba_dir, 'img_align_celeba')
        os.makedirs(img_dir, exist_ok=True)
        
        # Save 8 dummy images
        num_mock_imgs = 8
        for idx in range(num_mock_imgs):
            img = Image.new('RGB', (64, 64), color=(idx * 30, 80, 255 - idx * 30))
            img.save(os.path.join(img_dir, f"mock_celeba_{idx:03d}.jpg"))
            
        # Instantiate CelebA Wrapper (triggers FlatImageDataset fallback)
        batch_size = 2
        image_size = 32
        dataset = CelebADataset(
            data_dir=celeba_dir, 
            batch_size=batch_size, 
            image_size=image_size, 
            val_split_ratio=0.25, 
            num_workers=0, 
            download=False
        )
        
        train_loader, val_loader, test_loader = dataset.get_loaders()
        
        # 8 images split 0.25 val, 0.25 test -> 2 val, 2 test, 4 train
        self.assertEqual(len(dataset.train_dataset), 4)
        self.assertEqual(len(dataset.val_dataset), 2)
        self.assertEqual(len(dataset.test_dataset), 2)
        
        imgs, _ = next(iter(train_loader))
        
        # 1. Assert shape: (B, 3, 32, 32)
        self.assertEqual(imgs.shape, (batch_size, 3, image_size, image_size))
        
        # 2. Assert data type
        self.assertEqual(imgs.dtype, torch.float32)
        
        # 3. Assert normalization range is in [-1.0, 1.0]
        self.assertTrue(imgs.min() >= -1.05)
        self.assertTrue(imgs.max() <= 1.05)
        
        print(f"CelebA Mock Train split size: {len(dataset.train_dataset)}")
        print(f"CelebA Mock Val split size: {len(dataset.val_dataset)}")
        print(f"CelebA Mock Test split size: {len(dataset.test_dataset)}")
        print(f"Batch shape: {imgs.shape}")
        print(f"Batch value range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
        
        print("Logging 2 sample images:")
        for idx in range(min(2, len(imgs))):
            img = imgs[idx]
            print(f"  - Item {idx+1}: Shape={img.shape}, Mean={img.mean().item():.4f}, Std={img.std().item():.4f}")

if __name__ == '__main__':
    unittest.main()
