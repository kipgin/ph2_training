import torch
from torch import nn
from torch.nn import functional as F
class VQ_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        