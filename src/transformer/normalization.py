
import torch
from torch import nn


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean and variance across feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Normalization
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Learnable affine transformation
        return self.gamma * x_normalized + self.beta
