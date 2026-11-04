
import torch

from transformer.model import Transformer


def make_model():
    return Transformer(
        src_vocab_size=50,
        tgt_vocab_size=60,
        d_model=16,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        dropout=0.0,
        src_pad_id=0,
        tgt_pad_id=0,
    )


def test_transformer_forward():
    torch.manual_seed(42)

    model = make_model()

    src = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 8],
    ])

    tgt = torch.tensor([
        [1, 9, 10, 0],
        [1, 11, 12, 13],
    ])

    logits = model(src, tgt)

    assert logits.shape == (2, 4, 60)
    assert torch.isfinite(logits).all()


def test_transformer_backward():
    torch.manual_seed(42)

    model = make_model()

    src = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 8],
    ])

    tgt = torch.tensor([
        [1, 9, 10, 0],
        [1, 11, 12, 13],
    ])

    logits = model(src, tgt)

    target = torch.randn_like(logits)

    loss = (logits - target).square().mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_transformer_causality():
    torch.manual_seed(42)

    model = make_model().eval()

    src = torch.tensor([[1, 2, 3, 4]])

    tgt1 = torch.tensor([[1, 5, 6, 7]])
    tgt2 = torch.tensor([[1, 5, 8, 9]])

    with torch.no_grad():
        logits1 = model(src, tgt1)
        logits2 = model(src, tgt2)

    # Future token changes must not alter
    # the logits at positions 0 and 1.
    torch.testing.assert_close(
        logits1[:, :2, :],
        logits2[:, :2, :],
        rtol=1e-5,
        atol=1e-6,
    )
