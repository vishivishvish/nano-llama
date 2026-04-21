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


class NanoLlama(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x
