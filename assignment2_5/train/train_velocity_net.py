
import torch
import torch.optim as optim
import argparse
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.velocity_net import VelocityNet
from data.datasets import Dataset8Gaussians

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset8Gaussians(dim=2, device=device)
    model = VelocityNet(input_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.iters} iterations...")

    for i in range(args.iters):
        optimizer.zero_grad()

        x1 = dataset.sample(args.batch_size)  
        x0 = torch.randn_like(x1)
        t = torch.rand((args.batch_size, 1), device=device)

        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0

        pred_v = model(xt, t)
        loss = torch.mean((pred_v - target_v)**2)

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i:5d} | Loss: {loss.item():.6f}")

    save_path = os.path.join(args.output_dir, "velocity_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()