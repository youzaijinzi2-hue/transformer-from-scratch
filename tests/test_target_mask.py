
import pytest
import torch

from transformer.masking import make_target_mask
from transformer.attention import MultiHeadAttention


def test_target_mask_values():
    tokens = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 6, 7],
    ])

    mask = make_target_mask(tokens, pad_id=0)

    assert mask.shape == (2, 1, 4, 4)
    assert mask.dtype == torch.bool

    expected_first = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
    ], dtype=torch.bool)

    expected_second = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.bool)

    torch.testing.assert_close(mask[0, 0], expected_first)
    torch.testing.assert_close(mask[1, 0], expected_second)


def test_target_mask_blocks_attention():
    torch.manual_seed(42)

    tokens = torch.tensor([[1, 2, 3, 0]])

    mask = make_target_mask(tokens)

    attention = MultiHeadAttention(
        d_model=8,
        num_heads=2,
    ).double()

    x = torch.randn(1, 4, 8, dtype=torch.float64)

    _, weights = attention(x, mask=mask)

    # Future positions must be blocked
    assert torch.all(
        weights[0, :, 0, 1:] == 0
    )

    # Last key is PAD and must be blocked everywhere
    assert torch.all(
        weights[0, :, :, 3] == 0
    )

    # Attention weights remain normalized
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
    )


def test_target_mask_rejects_all_padding():
    tokens = torch.tensor([[0, 0, 0]])

    with pytest.raises(ValueError):
        make_target_mask(tokens)


def test_target_mask_rejects_empty_sequence():
    tokens = torch.empty(2, 0, dtype=torch.long)

    with pytest.raises(ValueError):
        make_target_mask(tokens)


def test_future_tokens_do_not_affect_past():
    torch.manual_seed(42)

    attention = MultiHeadAttention(8, 2).double()
    attention.eval()

    tokens = torch.tensor([[1, 2, 3, 4]])
    mask = make_target_mask(tokens)

    x1 = torch.randn(1, 4, 8, dtype=torch.float64)
    x2 = x1.clone()

    # Only change future positions
    x2[:, 2:, :] = torch.randn_like(x2[:, 2:, :])

    z1, _ = attention(x1, mask=mask)
    z2, _ = attention(x2, mask=mask)

    # Positions 0 and 1 must remain unchanged
    torch.testing.assert_close(
        z1[:, :2, :],
        z2[:, :2, :],
        rtol=1e-10,
        atol=1e-12,
    )
