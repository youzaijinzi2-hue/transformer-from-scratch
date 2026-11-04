
import torch
import torch.nn as nn

from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder
from transformer.masking import (
    make_padding_mask,
    make_target_mask,
)


class Transformer(nn.Module):
    """Original-style encoder-decoder Transformer."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        src_pad_id: int = 0,
        tgt_pad_id: int = 0,
        max_len: int = 5000,
    ):
        super().__init__()

        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
            pad_id=src_pad_id,
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
        )

        self.output_projection = nn.Linear(
            d_model, tgt_vocab_size
        )

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:

        source_mask = make_padding_mask(
            src_tokens, pad_id=self.src_pad_id
        )
        target_mask = make_target_mask(
            tgt_tokens, pad_id=self.tgt_pad_id
        )

        # Encoder
        memory, _ = self.encoder(
            src_tokens, mask=source_mask
        )

        # Decoder
        z, _, _ = self.decoder(
            tgt_tokens,
            memory,
            target_mask,
            source_mask,
        )

        # Output projection
        logits = self.output_projection(z)

        return logits
