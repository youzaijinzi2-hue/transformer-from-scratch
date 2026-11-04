
import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.feed_forward import FeedForward
from transformer.normalization import LayerNormalization
from transformer.embedding import TransformerEmbedding


class DecoderBlock(nn.Module):
    """Post-LN Transformer decoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model, num_heads
        )
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads
        )

        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. Masked Self-Attention
        self_out, self_weights = self.self_attention(
            x,
            mask=target_mask,
        )

        y1 = self.norm1(
            x + self.dropout1(self_out)
        )

        # 2. Cross-Attention
        cross_out, cross_weights = self.cross_attention(
            y1,
            context=encoder_output,
            mask=source_mask,
        )

        y2 = self.norm2(
            y1 + self.dropout2(cross_out)
        )

        # 3. Feed Forward Network
        ffn_out = self.ffn(y2)

        z = self.norm3(
            y2 + self.dropout3(ffn_out)
        )

        return z, self_weights, cross_weights


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([
            DecoderBlock(
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
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list, list]:

        x = self.embedding(token_ids)

        self_attention_maps = []
        cross_attention_maps = []

        for layer in self.layers:
            x, self_A, cross_A = layer(
                x,
                encoder_output,
                target_mask,
                source_mask,
            )

            self_attention_maps.append(self_A)
            cross_attention_maps.append(cross_A)

        return x, self_attention_maps, cross_attention_maps
