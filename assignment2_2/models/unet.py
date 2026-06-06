import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    #time embedding
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    #resnet block
    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8):
        super().__init__()
        #time embedd->block embedd
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(in_ch, out_ch, groups=groups)
        self.block2 = Block(out_ch, out_ch, groups=groups)
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        
        #h+= time embedd
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = h + time_emb[:, :, None, None]  #(B, C, H, W)            
        h = self.block2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    # dung o bottleneck va lowresolution
    def __init__(self, channels, groups=8):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_norm = self.group_norm(x)
        qkv = self.qkv(h_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        q = q.reshape(b, c, h * w).transpose(-1, -2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(-1, -2)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        
        context = torch.bmm(attn, v)
        context = context.transpose(-1, -2).reshape(b, c, h, w)
        
        return x + self.proj(context)

class UNet(nn.Module):
    def __init__(self, in_channels=1, time_emb_dim=256, hidden_dims=[64, 128, 256, 256]):
        super().__init__()
        
        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        self.downs = nn.ModuleList([])
        in_ch = hidden_dims[0]
        for idx in range(len(hidden_dims) - 1):
            out_ch = hidden_dims[idx + 1]
            block1 = ResnetBlock(in_ch, in_ch, time_emb_dim=time_emb_dim)
            block2 = ResnetBlock(in_ch, in_ch, time_emb_dim=time_emb_dim)
            attn = AttentionBlock(in_ch) if idx >= 1 else nn.Identity()
            down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
            
            self.downs.append(nn.ModuleList([block1, block2, attn, down]))
            in_ch = out_ch
            
        #Bottleneck
        self.mid_block1 = ResnetBlock(in_ch, in_ch, time_emb_dim=time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, time_emb_dim=time_emb_dim)
        
        #upsampling
        self.ups = nn.ModuleList([])
        reversed_dims = hidden_dims[::-1]
        for idx in range(len(reversed_dims) - 1):
            in_ch = reversed_dims[idx]
            out_ch = reversed_dims[idx + 1]
            
            upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
            
            #skip connection concatenation-> dim*2
            block1 = ResnetBlock(out_ch * 2, out_ch, time_emb_dim=time_emb_dim)
            block2 = ResnetBlock(out_ch, out_ch, time_emb_dim=time_emb_dim)
            attn = AttentionBlock(out_ch) if idx <= len(reversed_dims) - 3 else nn.Identity()
            
            self.ups.append(nn.ModuleList([upsample, block1, block2, attn]))
            
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], in_channels, 3, padding=1)
        )
        
    def forward(self, x, time):
        t = self.time_mlp(time)
        
        x = self.init_conv(x)
        
        skip_connections = []
        for block1, block2, attn, down in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            skip_connections.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)
            skip = skip_connections.pop()
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
        return self.final_conv(x)
