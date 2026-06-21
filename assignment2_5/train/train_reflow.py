import torch
import torch.optim as optim
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.velocity_net import VelocityNet
from infer.infer_pipeline import run_fm_inference
from infer.infer_utils import plot_trajectories

def train_reflow():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_path = "checkpoints/velocity_model.pth"
    
    base_model = VelocityNet().to(device)
    base_model.load_state_dict(torch.load(base_model_path))
    
    print("Gen 10k cap=============")
    num_pairs = 10000
    x0 = torch.randn(num_pairs, 2, device=device)
    trajs = run_fm_inference(base_model, steps=50, num_samples=num_pairs, device=device)
    x1 = trajs[-1] 
    
    #train
    reflow_model = VelocityNet().to(device)
    optimizer = optim.Adam(reflow_model.parameters(), lr=1e-4)
    
    print("Training Reflow Model (straightening paths)...")
    for i in range(10000):
        optimizer.zero_grad()
        
        idx = torch.randint(0, num_pairs, (512,))
        batch_x0, batch_x1 = x0[idx], x1[idx]
        
        t = torch.rand((512, 1), device=device)
        xt = (1 - t) * batch_x0 + t * batch_x1
        target_v = batch_x1 - batch_x0 
        
        loss = torch.mean((reflow_model(xt, t) - target_v)**2)
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print(f"Iter {i} | Loss: {loss.item():.6f}")

    #dung cai ben kia de in
    base_trajs = run_fm_inference(base_model, 50, 100, device)
    plot_trajectories(base_trajs, "Original FM (Curved)", "outputs/curvature_before.png")
    
    #giong
    reflow_trajs = run_fm_inference(reflow_model, 50, 100, device)
    plot_trajectories(reflow_trajs, "Reflow FM (Straight)", "outputs/curvature_after.png")
    
    torch.save(reflow_model.state_dict(), "checkpoints/reflow_model.pth")

if __name__ == "__main__":
    train_reflow()