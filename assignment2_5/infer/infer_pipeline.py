import torch
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.velocity_net import VelocityNet
from model.score_net import ScoreNet
from infer.infer_utils import plot_trajectories, plot_snapshots

@torch.no_grad()
def run_fm_inference(model, steps, num_samples, device):
    model.eval()
    x = torch.randn(num_samples, 2, device=device)
    dt = 1.0 / steps
    trajectory = [x.clone()]

    for i in range(steps):
        t = torch.full((num_samples, 1), i / steps, device=device)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.clone())
    
    return torch.stack(trajectory) # [Steps, B, 2]

@torch.no_grad()
def run_ddpm_inference(model, ckpt, num_samples, device):
    model.eval()
    T = ckpt['steps']
    beta, alpha, alpha_bar = ckpt['beta'], ckpt['alpha'], ckpt['alpha_bar']
    
    x = torch.randn(num_samples, 2, device=device)
    trajectory = [x.clone()]

    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0
        t_tensor = torch.full((num_samples, 1), t / T, device=device)
        eps_theta = model(x, t_tensor)
        
        coeff1 = 1 / torch.sqrt(alpha[t])
        coeff2 = (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t])
        x = coeff1 * (x - coeff2 * eps_theta) + torch.sqrt(beta[t]) * z
        trajectory.append(x.clone())
        
    return torch.stack(trajectory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['fm', 'ddpm'], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps_list", type=int, nargs='+', default=[20, 100])
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)

    if args.mode == 'fm':
        model = VelocityNet().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        for s in args.steps_list:
            trajs = run_fm_inference(model, s, 1000, device)
            plot_trajectories(trajs, f"FM Trajectories ({s} steps)", f"outputs/fm_traj_{s}.png")
            plot_snapshots(trajs, f"FM Evolution ({s} steps)", f"outputs/fm_snap_{s}.png")
    else:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = ScoreNet().to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        trajs = run_ddpm_inference(model, ckpt, 1000, device)
        plot_trajectories(trajs, "DDPM Trajectories", "outputs/ddpm_traj.png")
        plot_snapshots(trajs, "DDPM Evolution", "outputs/ddpm_snap.png")