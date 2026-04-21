import torch
import torch.nn as nn


class Config:
    def __init__(
        self,
        vocab_size=1000,
        dim=128,
        n_layers=4,
        n_heads=4,
        hidden_dim=256,
        max_seq_len=128,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (..., dim)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.scale * x_norm


class NanoLlama(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.norm = RMSNorm(config.dim)

    def forward(self, x):
        return self.norm(x)
