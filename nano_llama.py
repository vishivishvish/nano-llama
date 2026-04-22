import math

import torch
import torch.nn as nn


def precompute_rope_frequencies(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()

    freqs = torch.outer(positions, freqs)  # (seq_len, dim/2)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin


def apply_rope(x, cos, sin):
    # x shape: (batch, seq_len, dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    cos = cos[: x.shape[1]].unsqueeze(0)
    sin = sin[: x.shape[1]].unsqueeze(0)

    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return x_rotated


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

        cos, sin = precompute_rope_frequencies(config.dim, config.max_seq_len)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        x = self.norm(x)
        x = apply_rope(x, self.cos, self.sin)
        return x
