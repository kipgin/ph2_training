import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim if context_dim is not None else query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim if context_dim is not None else query_dim, inner_dim, bias=False)
        
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0)
        ])

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        bsz, q_len, _ = q.shape
        k_len = k.shape[1]
        head_dim = q.shape[-1] // h

        # Reshape to (B, heads, seq_len, head_dim) required by SDPA
        q = q.view(bsz, q_len, h, head_dim).transpose(1, 2)   # (B, h, q_len, d)
        k = k.view(bsz, k_len, h, head_dim).transpose(1, 2)   # (B, h, k_len, d)
        v = v.view(bsz, k_len, h, head_dim).transpose(1, 2)   # (B, h, k_len, d)

        # Memory-efficient scaled dot-product attention (Flash Attention when available).
        # Never materialises the full N² attention matrix — O(N) peak memory vs O(N²)
        # for the previous manual chunked bmm loop, which still OOM-ed at q_len=4096.
        # scale=None uses the default 1/sqrt(head_dim), identical to self.scale.
        out = F.scaled_dot_product_attention(q, k, v)         # (B, h, q_len, d)

        out = out.transpose(1, 2).reshape(bsz, q_len, -1)     # (B, q_len, inner_dim)

        for layer in self.to_out:
            out = layer(out)
        return out


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([
            GEGLU(dim, inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim_out)
        ])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None):
        super().__init__()
        self.attn1 = Attention(query_dim=dim, context_dim=None, heads=n_heads, dim_head=d_head)
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn2 = Attention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head) if context_dim is not None else None
        self.norm2 = nn.LayerNorm(dim) if context_dim is not None else None
        
        self.ff = FeedForward(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = x + self.attn1(self.norm1(x))
        if self.attn2 is not None:
            x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, context_dim=None, depth=1):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-5)
        # Checkpoint stores proj_in/proj_out as Conv2d(kernel_size=1), shape (N, N, 1, 1)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim)
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        bsz, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        x = self.proj_in(x)                                    # (B, inner_dim, H, W)
        x = x.permute(0, 2, 3, 1).reshape(bsz, h * w, -1)    # (B, H*W, inner_dim)
        
        for block in self.transformer_blocks:
            x = block(x, context=context)
            
        x = x.reshape(bsz, h, w, -1).permute(0, 3, 1, 2)     # (B, inner_dim, H, W)
        x = self.proj_out(x)                                   # (B, in_channels, H, W)
        return residual + x


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        bsz, tgt_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        # QuickGELU
        x = self.fc1(hidden_states)
        x = x * torch.sigmoid(1.702 * x)
        x = self.fc2(x)
        return x


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, num_layers=12, hidden_size=768, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class VAEAttention(nn.Module):
    def __init__(self, channels, groups=32, eps=1e-5):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([
            nn.Linear(channels, channels)
        ])

    def forward(self, x):
        bsz, c, h, w = x.shape
        residual = x

        x = self.group_norm(x)
        x = x.permute(0, 2, 3, 1).reshape(bsz, h * w, c)  # (B, H*W, C)

        q = self.to_q(x)  # (B, H*W, C)
        k = self.to_k(x)
        v = self.to_v(x)

        # Treat channel dim as a single head for SDPA: (B, 1, H*W, C)
        # Memory-efficient — avoids the (B, H*W, H*W) O(N²) attention matrix.
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, 1, H*W, C)
        out = out.squeeze(1)                           # (B, H*W, C)

        for layer in self.to_out:
            out = layer(out)

        out = out.reshape(bsz, h, w, c).permute(0, 3, 1, 2)
        return residual + out
