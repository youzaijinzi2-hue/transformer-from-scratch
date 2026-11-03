
import torch
from torch import nn

from transformer.attention import MultiHeadAttention
from transformer.embedding import TransformerEmbedding
from transformer.feed_forward import FeedForward
from transformer.normalization import LayerNormalization
from transformer.masking import make_padding_mask


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


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        if mask is None:
            mask = make_padding_mask(
                token_ids, pad_id=self.pad_id
            )

        x = self.embedding(token_ids)
        attention_maps = []

        for layer in self.layers:
            x, weights = layer(x, mask=mask)
            attention_maps.append(weights)

        return x, attention_maps

