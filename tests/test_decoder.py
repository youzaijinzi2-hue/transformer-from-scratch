
import torch

from transformer.decoder import DecoderBlock
from transformer.masking import (
    make_padding_mask,
    make_target_mask,
)


def test_decoder_forward():
    torch.manual_seed(42)

    B, Nt, Ns, D, H = 2, 4, 6, 8, 2

    decoder = DecoderBlock(
        d_model=D,
        num_heads=H,
        d_ff=32,
        dropout=0.0,
    )

    x = torch.randn(B, Nt, D)
    memory = torch.randn(B, Ns, D)

    target_ids = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 6, 7],
    ])

    source_ids = torch.tensor([
        [1, 2, 3, 4, 0, 0],
        [5, 6, 7, 8, 9, 10],
    ])

    target_mask = make_target_mask(target_ids)
    source_mask = make_padding_mask(source_ids)

    z, self_A, cross_A = decoder(
        x,
        memory,
        target_mask,
        source_mask,
    )

    assert z.shape == (B, Nt, D)
    assert self_A.shape == (B, H, Nt, Nt)
    assert cross_A.shape == (B, H, Nt, Ns)

    # No attention to future or PAD target keys
    assert torch.all(self_A[0, :, 0, 1:] == 0)
    assert torch.all(self_A[0, :, :, 3] == 0)

    # No attention to PAD source keys
    assert torch.all(cross_A[0, :, :, 4:] == 0)

    # Attention weights should sum to one
    torch.testing.assert_close(
        self_A.sum(dim=-1),
        torch.ones_like(self_A.sum(dim=-1)),
    )
    torch.testing.assert_close(
        cross_A.sum(dim=-1),
        torch.ones_like(cross_A.sum(dim=-1)),
    )


def test_decoder_backward():
    torch.manual_seed(42)

    B, Nt, Ns, D = 2, 4, 6, 8

    decoder = DecoderBlock(
        d_model=D,
        num_heads=2,
        d_ff=32,
        dropout=0.0,
    )

    x = torch.randn(B, Nt, D, requires_grad=True)
    memory = torch.randn(B, Ns, D, requires_grad=True)

    target_ids = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 6, 7],
    ])
    source_ids = torch.tensor([
        [1, 2, 3, 4, 0, 0],
        [5, 6, 7, 8, 9, 10],
    ])

    z, _, _ = decoder(
        x,
        memory,
        make_target_mask(target_ids),
        make_padding_mask(source_ids),
    )

    target = torch.randn_like(z)
    loss = (z - target).square().mean()

    loss.backward()

    assert x.grad is not None
    assert memory.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(memory.grad).all()

    for name, param in decoder.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
