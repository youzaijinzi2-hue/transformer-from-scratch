
import math

import torch
from torch import nn

from transformer.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        embedding_scale: float | None = None,
        positional_scale: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding_scale = (
            math.sqrt(d_model)
            if embedding_scale is None
            else embedding_scale
        )
        self.positional_scale = positional_scale

        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )
        self.position = PositionalEncoding(
            d_model, max_len
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        x = self.embedding_scale * x

        N = x.shape[1]
        if N > self.position.pe.shape[1]:
            raise ValueError("Sequence exceeds max_len.")

        pe = self.position.pe[:, :N, :].to(
            device=x.device,
            dtype=x.dtype,
        )

        x = x + self.positional_scale * pe

        return self.dropout(x)
