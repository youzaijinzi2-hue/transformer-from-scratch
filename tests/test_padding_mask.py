
import pytest
import torch

from transformer.masking import make_padding_mask
from transformer.encoder import TransformerEncoder


def make_encoder():
    model = TransformerEncoder(
        vocab_size=100,
        d_model=16,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        dropout=0.0,
        pad_id=0,
    )
    model.eval()
    return model


def test_padding_mask_shape():
    tokens = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 8],
    ])

    mask = make_padding_mask(tokens)

    assert mask.shape == (2, 1, 1, 5)

    expected = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])

    torch.testing.assert_close(
        mask[:, 0, 0, :], expected
    )


def test_padding_attention_weights():
    torch.manual_seed(42)

    model = make_encoder()

    tokens = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 8],
    ])

    _, attention_maps = model(tokens)

    for A in attention_maps:
        # Batch 0: last two Key positions are padding
        assert torch.all(A[0, :, :, 3:] == 0)

        # Each row remains normalized (dropout=0)
        torch.testing.assert_close(
            A.sum(dim=-1),
            torch.ones_like(A.sum(dim=-1)),
        )


def test_padding_values_do_not_affect_valid_tokens():
    torch.manual_seed(42)

    model = make_encoder()

    original = torch.tensor([[1, 2, 3, 0, 0]])
    changed = torch.tensor([[1, 2, 3, 9, 10]])

    # Explicitly preserve the original padding mask
    mask = make_padding_mask(original)

    with torch.no_grad():
        y1, _ = model(original, mask=mask)
        y2, _ = model(changed, mask=mask)

    # Valid token outputs must be unaffected
    torch.testing.assert_close(
        y1[:, :3],
        y2[:, :3],
        rtol=1e-5,
        atol=1e-6,
    )


def test_all_padding_is_rejected():
    tokens = torch.tensor([[0, 0, 0]])

    with pytest.raises(ValueError):
        make_padding_mask(tokens)
