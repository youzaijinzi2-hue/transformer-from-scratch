
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super().__init__()

        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError("d_model must be positive and even.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model)
        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[1]

        if N > self.pe.shape[1]:
            raise ValueError("Sequence exceeds max_len.")

        return x + self.pe[:, :N, :].to(
            device=x.device, dtype=x.dtype
        )
