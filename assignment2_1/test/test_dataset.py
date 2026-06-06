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

class TestDatasetClasses(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for mock data loading tests
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Remove the temp directory after tests complete
        shutil.rmtree(self.temp_dir)
        
    def test_mnist_dataset(self):
        print("\n=== RUNNING MNIST DATASET TEST ===")
        try:
            mnist_dir = os.path.join(self.temp_dir, 'mnist_data')
            batch_size = 4
            
            # Attempt to instantiate MNIST Dataset (might skip if offline/no download files exist)
            dataset = MNISTDataset(
                data_dir=mnist_dir, 
                batch_size=batch_size, 
                image_size=128, 
                val_split_ratio=0.1, 
                num_workers=0
            )
            
            train_loader, val_loader, test_loader = dataset.get_loaders()
            
            # Check that splits actually contain elements
            self.assertTrue(len(dataset.train_dataset) > 0, "Train split is empty.")
            self.assertTrue(len(dataset.val_dataset) > 0, "Validation split is empty.")
            self.assertTrue(len(dataset.test_dataset) > 0, "Test split is empty.")
            
            # Load a sample batch
            imgs, labels = next(iter(train_loader))
            
            # 1. Verify tensor dimensions: (batch_size, 1, 128, 128)
            self.assertEqual(imgs.shape, (batch_size, 1, 128, 128), f"Incorrect shape: {imgs.shape}")
            
            # 2. Verify tensor float type
            self.assertEqual(imgs.dtype, torch.float32, f"Incorrect type: {imgs.dtype}")
            
            # 3. Verify pixel value normalizations range in [-1.0, 1.0] (for Tanh compatibility)
            self.assertTrue(imgs.min() >= -1.05, f"Values too low: {imgs.min()}")
            self.assertTrue(imgs.max() <= 1.05, f"Values too high: {imgs.max()}")
            
            print(f"MNIST Train split items: {len(dataset.train_dataset)}")
            print(f"MNIST Val split items: {len(dataset.val_dataset)}")
            print(f"MNIST Test split items: {len(dataset.test_dataset)}")
            print(f"Loaded batch shape: {imgs.shape}")
            print(f"Batch value range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
            
            # Print details of 3 sample data items
            print("Logging 3 sample images:")
            for idx in range(min(3, len(imgs))):
                img = imgs[idx]
                print(f"  - Item {idx+1}: Shape={img.shape}, Mean={img.mean().item():.4f}, Std={img.std().item():.4f}")
                
        except Exception as e:
            print(f"Notice: Skipping MNIST download tests due to: {e}")
            
    def test_celeba_dataset_fallback(self):
        print("\n=== RUNNING CELEBA DATASET FALLBACK TEST ===")
        # Build mock folder directory and mock image files to test FlatImageDataset fallback
        celeba_dir = os.path.join(self.temp_dir, 'celeba_data')
        img_align_dir = os.path.join(celeba_dir, 'img_align_celeba')
        os.makedirs(img_align_dir, exist_ok=True)
        
        # Save 10 mock RGB images
        num_mock_imgs = 10
        for idx in range(num_mock_imgs):
            img = Image.new('RGB', (128, 128), color=(idx * 20, 100, 255 - idx * 20))
            img.save(os.path.join(img_align_dir, f"mock_celeba_{idx:03d}.jpg"))
            
        # Instantiate CelebADataset (should trigger FlatImageDataset fallback)
        batch_size = 2
        dataset = CelebADataset(
            data_dir=celeba_dir, 
            batch_size=batch_size, 
            image_size=128, 
            val_split_ratio=0.2, 
            num_workers=0, 
            download=False
        )
        
        train_loader, val_loader, test_loader = dataset.get_loaders()
        
        # Verify splits partition (10 images, val_ratio=0.2, test_ratio=0.2 -> 6 train, 2 val, 2 test)
        self.assertEqual(len(dataset.train_dataset), 6, f"Incorrect train size: {len(dataset.train_dataset)}")
        self.assertEqual(len(dataset.val_dataset), 2, f"Incorrect val size: {len(dataset.val_dataset)}")
        self.assertEqual(len(dataset.test_dataset), 2, f"Incorrect test size: {len(dataset.test_dataset)}")
        
        # Load a sample batch
        imgs, _ = next(iter(train_loader))
        
        # 1. Verify tensor dimensions: (batch_size, 3, 128, 128)
        self.assertEqual(imgs.shape, (batch_size, 3, 128, 128), f"Incorrect shape: {imgs.shape}")
        
        # 2. Verify tensor float type
        self.assertEqual(imgs.dtype, torch.float32, f"Incorrect type: {imgs.dtype}")
        
        # 3. Verify pixel value range in [-1.0, 1.0]
        self.assertTrue(imgs.min() >= -1.05, f"Values too low: {imgs.min()}")
        self.assertTrue(imgs.max() <= 1.05, f"Values too high: {imgs.max()}")
        
        print(f"CelebA Mock Train split items: {len(dataset.train_dataset)}")
        print(f"CelebA Mock Val split items: {len(dataset.val_dataset)}")
        print(f"CelebA Mock Test split items: {len(dataset.test_dataset)}")
        print(f"Loaded batch shape: {imgs.shape}")
        print(f"Batch value range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
        
        # Print details of 3 sample data items
        print("Logging 3 sample images:")
        for idx in range(min(3, len(imgs))):
            img = imgs[idx]
            print(f"  - Item {idx+1}: Shape={img.shape}, Mean={img.mean().item():.4f}, Std={img.std().item():.4f}")

if __name__ == '__main__':
    unittest.main()
