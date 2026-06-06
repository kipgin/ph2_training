import os
import sys
import yaml
import torch
import argparse
from torchvision.utils import save_image, make_grid

# Add the parent folder of infer to sys.path to resolve models and dataset imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.baseline_vae import BaselineVAE
from dataset.datasets import MNISTDataset, CelebADataset

def main():
    parser = argparse.ArgumentParser(description="Inference CLI Script for Baseline VAE")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pretrained weights (.pth file)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML (if config not in checkpoint)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save output images')
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'reconstruct'], help='Inference mode')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of images to generate or reconstruct')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Inference device override')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Running inference on: {device}")
    
    # 1. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Determine if checkpoint is custom dictionary format or raw state_dict
    is_state_dict = True
    config = None
    
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            is_state_dict = False
            # Try to restore config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                print("Configuration successfully loaded from the checkpoint metadata.")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # 2. Load Configuration if not loaded from checkpoint
    if config is None:
        config_path = args.config
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), '..', args.config)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {args.config}")
                
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Loaded configuration parameters from YAML file.")
        
    # Determine number of input channels based on the configuration dataset parameter
    dataset_name = config['dataset'].lower()
    in_channels = 1 if dataset_name == 'mnist' else 3
    config['base_vae']['in_channels'] = in_channels
    
    # 3. Instantiate and setup VAE Model
    model = BaselineVAE(
        in_channels=in_channels,
        hidden_dims=config['base_vae']['hidden_dims'],
        latent_dim=config['base_vae']['latent_dim']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    print("Model initialized and checkpoint weights successfully loaded.")
    
    # Setup output folder
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. Execute Inference Mode
    if args.mode == 'sample':
        print(f"Generating {args.num_samples} samples from latent prior space...")
        with torch.no_grad():
            samples = model.sample(args.num_samples, current_device=device)
            # Normalize from [-1, 1] back to [0, 1]
            samples = (samples + 1.0) / 2.0
            
            # Save as a single image grid
            grid_cols = int(args.num_samples ** 0.5)
            grid = make_grid(samples, nrow=grid_cols if grid_cols > 0 else 1)
            
            output_path = os.path.join(args.output_dir, f"generation_samples.png")
            save_image(grid, output_path)
            print(f"Successfully generated grid saved to: {output_path}")
            
            # Optionally save individual files
            indiv_dir = os.path.join(args.output_dir, 'samples')
            os.makedirs(indiv_dir, exist_ok=True)
            for i, img in enumerate(samples):
                save_image(img, os.path.join(indiv_dir, f"sample_{i+1}.png"))
            print(f"Individual samples saved to: {indiv_dir}/")
            
    elif args.mode == 'reconstruct':
        print(f"Loading {dataset_name} dataset to perform reconstructions...")
        
        # Load dataset
        if dataset_name == 'mnist':
            dataset = MNISTDataset(
                data_dir=config['data_dir'],
                batch_size=args.num_samples,
                image_size=config['image_size'],
                val_split_ratio=config['val_split_ratio'],
                num_workers=config['num_workers']
            )
        elif dataset_name == 'celeba':
            dataset = CelebADataset(
                data_dir=config['data_dir'],
                batch_size=args.num_samples,
                image_size=config['image_size'],
                val_split_ratio=config['val_split_ratio'],
                num_workers=config['num_workers']
            )
            
        _, _, test_loader = dataset.get_loaders()
        
        # Get a single batch of images
        real_imgs, _ = next(iter(test_loader))
        real_imgs = real_imgs[:args.num_samples].to(device)
        
        print(f"Reconstructing batch of {real_imgs.size(0)} images...")
        with torch.no_grad():
            recon_imgs, _, _ = model(real_imgs)
            
            # De-normalize images
            real_imgs = (real_imgs + 1.0) / 2.0
            recon_imgs = (recon_imgs + 1.0) / 2.0
            
            # Create a side-by-side reconstruction comparison grid
            comparison = torch.cat([real_imgs, recon_imgs], dim=0)
            grid = make_grid(comparison, nrow=real_imgs.size(0))
            
            output_path = os.path.join(args.output_dir, "reconstruction_comparison.png")
            save_image(grid, output_path)
            print(f"Successfully saved reconstruction comparison grid to: {output_path}")

if __name__ == '__main__':
    main()
