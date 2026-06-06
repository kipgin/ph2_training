import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.
    This schedule keeps the noise rate lower in early steps to preserve image structure.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear beta schedule as proposed in Ho et al. 2020.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    """
    Helper function to gather coefficients for a batch of timesteps
    and reshape to align with spatial dimension: (B, 1, 1, 1)
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) wrapper around U-Net.
    Precalculates schedules and buffers for forward and reverse diffusion steps.
    """
    def __init__(self, unet, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_name='linear'):
        super().__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps
        
        # 1. Setup beta schedule
        if schedule_name == 'linear':
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_name == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported schedule name: {schedule_name}")
            
        self.register_buffer('betas', betas)
        
        # Calculate alpha values
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        
        # Compute cumulative product of alphas (\bar{\alpha}_t)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # Shifted cumprod for calculating posterior distributions (\bar{\alpha}_{t-1})
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for forward diffusion process: q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for reverse process posterior: q(x_{t-1} | x_t, x_0)
        # \tilde{\beta}_t = \beta_t * (1 - \bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Log calculation clamped to prevent log(0) at index 0
        self.register_buffer('posterior_log_variance_clipped', torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        ))
        
        # Reciprocal of square root of alphas for posterior mean calculation: 1 / \sqrt{\alpha}_t
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process. Adds noise to the initial image x_start:
        x_t = \sqrt{\bar{\alpha}_t} * x_start + \sqrt{1 - \bar{\alpha}_t} * \epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def p_losses(self, x_start, t, noise=None):
        """
        Forward step and training loss computation.
        Generates random noise, adds it to inputs at timestep t, and trains U-Net to predict it.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Diffuse image to step t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict the noise using U-Net
        predicted_noise = self.unet(x_noisy, t)
        
        # Standard MSE loss between predicted and ground truth noise
        return F.mse_loss(noise, predicted_noise)
        
    def p_sample(self, x, t, t_index):
        """
        Reverse denoising step (p(x_{t-1} | x_t)).
        Estimates the mean and variance to sample x_{t-1}.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise using U-Net
        predicted_noise = self.unet(x, t)
        
        # Calculate posterior mean:
        # \mu_t = \frac{1}{\sqrt{\alpha}_t} ( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} * \epsilon_\theta(x_t, t) )
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise
        )
        
        if t_index == 0:
            return model_mean
        else:
            # Sample standard normal noise and scale it by posterior standard deviation
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
            
    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        """
        Iterative generation loop. Generates full image starting from pure normal noise.
        """
        batch_size = shape[0]
        # Start from random normal distribution
        img = torch.randn(shape, device=device)
        
        # Run reverse loop backwards from T-1 to 0
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            
        return img
        
    def forward(self, x):
        # Forward pass samples a random timestep for each batch element
        batch_size = x.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t)
