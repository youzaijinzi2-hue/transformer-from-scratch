
import torch
from torch import nn

from transformer.attention import MultiHeadAttention
from transformer.feed_forward import FeedForward
from transformer.normalization import LayerNormalization


class EncoderBlock(nn.Module):
    """Transformer encoder block (Post-LN)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        self.ffn = FeedForward(d_model, d_ff)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attn_output, attn_weights = self.attention(
            x, mask=mask
        )

        # Residual dropout + Add & Norm
        y = self.norm1(
            x + self.dropout1(attn_output)
        )

        ffn_output = self.ffn(y)

        # Residual dropout + Add & Norm
        z = self.norm2(
            y + self.dropout2(ffn_output)
        )

        return z, attn_weights
