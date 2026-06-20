import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.score_net import ScoreNet
from data.datasets import Dataset8Gaussians

@torch.no_grad()
def sample_ddpm(model, scheduler, num_samples, device):
    model.eval()
    T = scheduler['steps']
    beta = scheduler['beta']
    alpha = scheduler['alpha']
    alpha_bar = scheduler['alpha_bar']

    x = torch.randn(num_samples, 2, device=device)

    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0
        
        t_tensor = torch.full((num_samples, 1), t / T, device=device)
        
        eps_theta = model(x, t_tensor)
    
        coeff1 = 1 / torch.sqrt(alpha[t])
        coeff2 = (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t])
        sigma_t = torch.sqrt(beta[t]) 
        x = coeff1 * (x - coeff2 * eps_theta) + sigma_t * z
        
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_ddpm/ddpm_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sample_size", type=int, default=10000)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load everything
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ScoreNet(input_dim=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # Generate Samples
    print(f"DDPM Inference: Generating {args.sample_size} samples...")
    samples = sample_ddpm(model, ckpt, args.sample_size, device)

    # Visualization
    ds = Dataset8Gaussians(2, "cpu")
    square_range = ds.get_square_range(samples.cpu())

    fig, ax = plt.subplots(figsize=(5, 5))
    H = ax.hist2d(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), 
                  bins=300, range=square_range)
    
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    norm = cm.colors.Normalize(vmin=0.0, vmax=cmax)
    
    ax.hist2d(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), 
              bins=300, norm=norm, range=square_range)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"DDPM Generated Samples")

    save_path = Path(args.output_dir) / "inferred_ddpm_8gaussians.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()