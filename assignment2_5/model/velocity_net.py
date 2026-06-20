from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Module
from tqdm.auto import tqdm

import torch.nn as nn

class VelocityNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
    
        t_emb = self.time_mlp(t)
        
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)

class ScoreNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        # x: [B, 2], t: [B, 1]
        t_emb = self.time_mlp(t)
        # Concatenate spatial info with time info
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)