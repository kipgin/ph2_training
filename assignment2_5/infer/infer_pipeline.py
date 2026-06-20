import torch
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.velocity_net import VelocityNet, ScoreNet
from infer.infer_utils import plot_trajectories, plot_snapshots

@torch.no_grad()
def run_fm_inference(model, steps, num_samples, device):
    model.eval()
    x = torch.randn(num_samples, 2, device=device)
    dt = 1.0 / steps
    trajectory = [x.clone()]

    for i in range(steps):
        # [0,1]
        t = torch.full((num_samples, 1), i / steps, device=device)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.clone())
    
    return torch.stack(trajectory) # [Steps+1, B, 2]

@torch.no_grad()
def run_ddpm_inference(model, ckpt, steps, num_samples, device):
    """
    Sub-sampled DDPM Inference.
    steps: The number of steps to take during inference (e.g., 50).
    ckpt['steps']: The total steps model was trained on (e.g., 1000).
    """
    model.eval()
    T_train = ckpt['steps']
    beta = ckpt['beta']
    alpha = ckpt['alpha']
    alpha_bar = ckpt['alpha_bar']
    
    x = torch.randn(num_samples, 2, device=device)
    trajectory = [x.clone()]

    indices = torch.linspace(T_train - 1, 0, steps).long().to(device)
    
    for i in range(len(indices)):
        t_idx = indices[i]
        
        t_norm = torch.full((num_samples, 1), t_idx / T_train, device=device)
        
        eps_theta = model(x, t_norm)
        
        a = alpha[t_idx]
        a_bar = alpha_bar[t_idx]
        b = beta[t_idx]
        
        z = torch.randn_like(x) if t_idx > 0 else 0
        
        coeff1 = 1 / torch.sqrt(a)
        coeff2 = (1 - a) / torch.sqrt(1 - a_bar)
        sigma_t = torch.sqrt(b)
        
        x = coeff1 * (x - coeff2 * eps_theta) + sigma_t * z
        trajectory.append(x.clone())
        
    return torch.stack(trajectory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['fm', 'ddpm'], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps_list", type=int, nargs='+', default=[20, 100], 
                        help="List of step counts for inference")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)

    if args.mode == 'fm':
        print(f"Running Flow Matching Inference with steps: {args.steps_list}")
        model = VelocityNet().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        for s in args.steps_list:
            trajs = run_fm_inference(model, s, args.num_samples, device)
            plot_trajectories(trajs, f"FM Trajectories ({s} steps)", f"outputs/fm_traj_{s}.png")
            plot_snapshots(trajs, f"FM Evolution ({s} steps)", f"outputs/fm_snap_{s}.png")
            
    elif args.mode == 'ddpm':
        print(f"Running DDPM Inference with steps: {args.steps_list}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = ScoreNet().to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        
        for s in args.steps_list:
            actual_steps = min(s, ckpt['steps'])
            trajs = run_ddpm_inference(model, ckpt, actual_steps, args.num_samples, device)
            
            plot_trajectories(trajs, f"DDPM Trajectories ({actual_steps} steps)", f"outputs/ddpm_traj_{actual_steps}.png")
            plot_snapshots(trajs, f"DDPM Evolution ({actual_steps} steps)", f"outputs/ddpm_snap_{actual_steps}.png")

    print("Inference and plotting complete. Check the 'outputs' directory.")