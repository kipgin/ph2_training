import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

def preprocess_for_fid(img_tensor):
    #map [-1, 1] to [0, 1]
    img_tensor = (img_tensor + 1.0) / 2.0
    #scale to [0, 255] and cast to uint8
    img_tensor = torch.clamp(img_tensor * 255.0, 0.0, 255.0).to(torch.uint8)
    
    #if the image is grayscale (1 channel), repeat it 3 times to make it 3-channel RGB
    if img_tensor.size(1) == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
    return img_tensor

def compute_fid_score(model, data_loader, device, num_samples=200, image_size=32, mode='sample'):
    """
    Computes the Frechet Inception Distance (FID) score comparing the real dataset images
    with the DDPM generated samples.
    
    Args:
        model: The DDPM model instance.
        data_loader: The test or validation DataLoader containing real images.
        device: The device to run computations on (e.g. 'cuda' or 'cpu').
        num_samples: Total number of images to use for calculating the metric.
        image_size: The spatial size of images (e.g., 32).
        mode: ignored for diffusion, always generates brand new samples.
    """
    model.eval()
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    real_images_list = []
    
    #input data
    with torch.no_grad():
        for imgs, _ in data_loader:
            real_images_list.append(imgs.cpu())  #images on cpu
            if sum(x.size(0) for x in real_images_list) >= num_samples:
                break
                
    real_images = torch.cat(real_images_list, dim=0)[:num_samples]
    
    # Generate new images using reverse diffusion loop
    in_channels = model.unet.init_conv.in_channels
    shape = (num_samples, in_channels, image_size, image_size)
    gen_images = model.p_sample_loop(shape, device=device).cpu()
        
    # Preprocess tensors for FID (shift scale to uint8 RGB)
    real_uint8 = preprocess_for_fid(real_images)
    gen_uint8 = preprocess_for_fid(gen_images)
    
    # Update FID metric in mini-batches to avoid GPU OOM errors
    batch_size = 64
    for i in range(0, num_samples, batch_size):
        r_batch = real_uint8[i:i+batch_size].to(device)
        g_batch = gen_uint8[i:i+batch_size].to(device)
        fid_metric.update(r_batch, real=True)
        fid_metric.update(g_batch, real=False)
        
    fid_score = fid_metric.compute().item()
    fid_metric.reset()
    return fid_score

class FlatImageDataset(Dataset):
    """
    A custom dataset to load flat directories of images or standard folders containing
    CelebA dataset images. It supports common image extensions and converts them to RGB.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = []
        
        # Search the root dir and common subdirectories for images
        search_dirs = [root_dir]
        for sub in ['img_align_celeba', 'celeba', 'images']:
            sub_path = os.path.join(root_dir, sub)
            if os.path.isdir(sub_path):
                search_dirs.append(sub_path)
                
        for d in search_dirs:
            if os.path.exists(d):
                files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(valid_extensions)]
                if len(files) > 0:
                    self.image_files = files
                    self.root_dir = d
                    break
                    
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No image files found under directory {root_dir}")
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # Return a dummy label to maintain MNIST API consistency

class MNISTDataset:
    """
    A wrapper class for standard torchvision MNIST dataset.
    Handles splitting train set into train and validation sets, dataloaders creation,
    and exposes an interface for FID evaluation.
    """
    def __init__(self, data_dir, batch_size, image_size=32, val_split_ratio=0.1, num_workers=2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_split_ratio = val_split_ratio
        self.num_workers = num_workers
        self.in_channels = 1
        
        # Normalize to [-1, 1] range which is standard for diffusion models
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load raw dataset splits
        train_val_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=self.transform, download=True)
        self.test_dataset = datasets.MNIST(root=self.data_dir, train=False, transform=self.transform, download=True)
        
        # Split train_val_dataset into train and validation
        val_size = int(len(train_val_dataset) * self.val_split_ratio)
        train_size = len(train_val_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
    def get_loaders(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader
        
    def calculate_fid(self, model, device, num_samples=200, mode='sample'):
        _, _, test_loader = self.get_loaders()
        return compute_fid_score(model, test_loader, device, num_samples, self.image_size, mode)

class CelebADataset:
    
    def __init__(self, data_dir, batch_size, image_size=32, val_split_ratio=0.1, num_workers=2, download=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_split_ratio = val_split_ratio
        self.num_workers = num_workers
        self.in_channels = 3
        
        #augumentation and scale to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        #load datasets
        try:
            self.train_dataset = datasets.CelebA(root=self.data_dir, split='train', transform=self.transform, download=download)
            self.val_dataset = datasets.CelebA(root=self.data_dir, split='valid', transform=self.transform, download=download)
            self.test_dataset = datasets.CelebA(root=self.data_dir, split='test', transform=self.transform, download=download)
            print("Successfully loaded CelebA dataset via torchvision.datasets.CelebA")
        except Exception as e:
            print(f"Warning: Standard torchvision CelebA loader failed ({e}). Loading fallback FlatImageDataset...")
            
            try:
                full_dataset = FlatImageDataset(root_dir=self.data_dir, transform=self.transform)
                
                #80/10/10 splits
                total_len = len(full_dataset)
                val_size = int(total_len * self.val_split_ratio)
                test_size = int(total_len * self.val_split_ratio)
                train_size = total_len - val_size - test_size
                
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
                )
                print(f"Successfully loaded {total_len} CelebA images using FlatImageDataset fallback.")
            except Exception as fallback_err:
                print("Error: Failed to load CelebA dataset through torchvision and custom fallback.")
                raise fallback_err
                
    def get_loaders(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader
        
    def calculate_fid(self, model, device, num_samples=200, mode='sample'):
        _, _, test_loader = self.get_loaders()
        return compute_fid_score(model, test_loader, device, num_samples, self.image_size, mode)
