import torch
import torch.optim as optim
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.velocity_net import ScoreNet
from data.datasets import Dataset8Gaussians

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000, help="Total diffusion steps T")
    parser.add_argument("--output_dir", type=str, default="checkpoints_ddpm")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beta = torch.linspace(1e-4, 0.02, args.steps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    dataset = Dataset8Gaussians(dim=2, device=device)
    model = ScoreNet(input_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training DDPM on 8Gaussians...")

    for i in range(args.iters):
        optimizer.zero_grad()

        x0 = dataset.sample(args.batch_size)
        
        eps = torch.randn_like(x0)

        t_idx = torch.randint(0, args.steps, (args.batch_size,), device=device)
        a_bar = alpha_bar[t_idx].unsqueeze(-1)

        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps

        t_norm = t_idx.float().unsqueeze(-1) / args.steps
        pred_eps = model(xt, t_norm)

        loss = torch.mean((eps - pred_eps)**2)
        
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i:5d} | Loss: {loss.item():.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'beta': beta,
        'alpha': alpha,
        'alpha_bar': alpha_bar,
        'steps': args.steps
    }, os.path.join(args.output_dir, "ddpm_model.pth"))

if __name__ == "__main__":
    train()