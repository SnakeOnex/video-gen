import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads=8, causal=True, dropout=0.):
        super().__init__()
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.causal = causal

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k(x).reshape(B, T, self.n_heads, self.head_dim)
        v = self.v(x).reshape(B, T, self.n_heads, self.head_dim)
        q = rearrange(q, 'B T H D -> B H T D')
        k = rearrange(k, 'B T H D -> B H T D')
        v = rearrange(v, 'B T H D -> B H T D')

        q *= self.head_dim ** -0.5
        attn = q @ k.transpose(-2, -1)

        if self.causal:
            mask = torch.triu(torch.ones(T, T), diagonal=1).to(attn.device)
            attn.masked_fill_(mask == 1, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out(out)
        return out

class STBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.spatial_attn = SelfAttention(dim, num_heads)
        self.temporal_attn = SelfAttention(dim, num_heads, causal=True)
        self.mlp = MLP(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x):
        B, T, S, C = x.shape
        x_SC = rearrange(x, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm(x_SC))

        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(self.norm(x_TC))

        x = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        x = x + self.mlp(self.norm(x))
        return x

class STTransformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            STBlock(dim, num_heads) for _ in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    model = STTransformer(num_layers=2, dim=64, num_heads=4)
    x = torch.randn(1, 32, 64, 64)
    out = model(x)
    print(out.shape)  # torch.Size([1, 32, 64, 64])
