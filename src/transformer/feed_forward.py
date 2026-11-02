
import torch
from torch import nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))
