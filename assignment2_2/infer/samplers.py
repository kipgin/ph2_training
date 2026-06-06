import os
import sys
import yaml
import torch
import argparse
from torchvision.utils import save_image, make_grid

# Add parent directory of infer to sys.path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet
from models.ddpm import DDPM
from models.ddpm import extract
from dataset.datasets import MNISTDataset, CelebADataset

class DDPMSampler:
    """
    Standard DDPM sampler that runs the full reverse diffusion process
    stochastically for T steps.
    """
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def sample(self, shape, device):
        self.model.eval()
        return self.model.p_sample_loop(shape, device)

class DDIMSampler:
    """
    Accelerated Denoising Diffusion Implicit Model (DDIM) sampler.
    Allows sampling on a smaller subsequence of timesteps (e.g. 50 steps instead of 1000)
    using deterministic transitions (if eta=0) or stochastic transitions (if eta > 0).
    """
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def sample(self, shape, device, steps=50, eta=0.0):
        self.model.eval()
        batch_size = shape[0]
        T = self.model.num_timesteps
        
        # Start from pure Gaussian noise
        img = torch.randn(shape, device=device)
        
        # Subsequence of timesteps
        # E.g. for steps=50, times = [-1, 19, 39, ..., 999]
        times = torch.linspace(-1, T - 1, steps + 1, dtype=torch.long)
        
        # Iterate backwards from steps down to 1
        for i in reversed(range(1, steps + 1)):
            t = torch.full((batch_size,), times[i], device=device, dtype=torch.long)
            t_prev = torch.full((batch_size,), times[i-1], device=device, dtype=torch.long)
            
            # Predict noise using U-Net
            epsilon_theta = self.model.unet(img, t)
            
            # Get cumulative products
            alphas_cumprod_t = extract(self.model.alphas_cumprod, t, img.shape)
            if times[i-1] == -1:
                alphas_cumprod_t_prev = torch.ones_like(alphas_cumprod_t)
            else:
                alphas_cumprod_t_prev = extract(self.model.alphas_cumprod, t_prev, img.shape)
                
            # 1. Estimate initial x_0:
            # \hat{x}_0 = (x_t - \sqrt{1 - \bar{\alpha}_t} * \epsilon_\theta(x_t, t)) / \sqrt{\bar{\alpha}_t}
            pred_x0 = (img - torch.sqrt(1.0 - alphas_cumprod_t) * epsilon_theta) / torch.sqrt(alphas_cumprod_t)
            
            # 2. Compute posterior variance \sigma_t
            if eta == 0.0:
                sigmas_t = torch.zeros_like(alphas_cumprod_t)
            else:
                sigmas_t = eta * torch.sqrt(
                    (1.0 - alphas_cumprod_t_prev) / (1.0 - alphas_cumprod_t) * (1.0 - alphas_cumprod_t / alphas_cumprod_t_prev)
                )
                
            # 3. Compute direction pointing to x_t
            pred_dir_xt = torch.sqrt(1.0 - alphas_cumprod_t_prev - sigmas_t**2) * epsilon_theta
            
            # 4. Generate next step sample
            # x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} * \hat{x}_0 + \text{direction} + \sigma_t * z_t
            noise = torch.randn_like(img) if eta > 0.0 else torch.zeros_like(img)
            img = torch.sqrt(alphas_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * noise
            
        return img

def main():
    parser = argparse.ArgumentParser(description="Inference CLI for DDPM and DDIM")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pre-trained weights (.pth)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML (fallback)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Folder to save output grid')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of images to generate')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddpm', 'ddim'], help='Inference sampler type')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of skip steps for DDIM')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='Stochasticity parameter eta (0=deterministic)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Override device')
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device for inference: {device}")
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Retrieve configuration
    config = None
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            if 'config' in checkpoint:
                config = checkpoint['config']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    if config is None:
        config_path = args.config
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), '..', args.config)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
    dataset_name = config['dataset'].lower()
    in_channels = 1 if dataset_name == 'mnist' else 3
    config['model']['in_channels'] = in_channels
    image_size = config['image_size']
    
    # Instantiate models
    unet = UNet(
        in_channels=in_channels,
        time_emb_dim=config['model']['time_emb_dim'],
        hidden_dims=config['model']['hidden_dims']
    )
    
    model = DDPM(
        unet=unet,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule_name=config['diffusion']['schedule_name']
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print("Pre-trained model loaded successfully.")
    
    # Select sampler
    shape = (args.num_samples, in_channels, image_size, image_size)
    if args.sampler == 'ddpm':
        print(f"Sampling {args.num_samples} images using DDPM (stochastic {config['diffusion']['num_timesteps']} steps)...")
        sampler = DDPMSampler(model)
        samples = sampler.sample(shape, device)
    else:  # ddim
        print(f"Sampling {args.num_samples} images using DDIM (deterministic {args.ddim_steps} steps)...")
        sampler = DDIMSampler(model)
        samples = sampler.sample(shape, device, steps=args.ddim_steps, eta=args.ddim_eta)
        
    # De-normalize [-1, 1] to [0, 1]
    samples = (samples + 1.0) / 2.0
    
    # Save as grid
    os.makedirs(args.output_dir, exist_ok=True)
    grid_cols = int(args.num_samples ** 0.5)
    grid = make_grid(samples, nrow=grid_cols if grid_cols > 0 else 1)
    
    filename = f"generated_{args.sampler}_s{args.num_samples}.png"
    output_path = os.path.join(args.output_dir, filename)
    save_image(grid, output_path)
    print(f"Successfully saved generation grid to: {output_path}")

if __name__ == '__main__':
    main()
