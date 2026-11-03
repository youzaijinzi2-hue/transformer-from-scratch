
import math

import torch
from torch import nn

from transformer.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """Token embedding, positional encoding and dropout."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )
        self.position = PositionalEncoding(
            d_model, max_len
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        x = x * math.sqrt(self.d_model)
        x = self.position(x)
        return self.dropout(x)
