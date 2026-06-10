import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # diffusers uses [sin, cos] by default (flip_sin_to_cos=False).
    # If flip_sin_to_cos is True, it swaps them to [cos, sin].
    if flip_sin_to_cos:
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    else:
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if embedding_dim % 2 == 1:
        # pad zero
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim=320, out_dim=1280):
        super().__init__()
        self.linear_1 = nn.Linear(time_emb_dim, out_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size=49408, hidden_size=768, max_position_embeddings=77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, position_ids=None):
        seq_length = input_ids.shape[-1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        inputs_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        return inputs_embeds + position_embeds
