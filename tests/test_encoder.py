
import torch

from transformer.encoder import EncoderBlock


def test_encoder_shape():
    torch.manual_seed(42)

    model = EncoderBlock(
        d_model=8,
        num_heads=2,
        d_ff=32,
    )

    x = torch.randn(2, 4, 8)

    z, weights = model(x)

    assert z.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)


def test_encoder_backward():
    torch.manual_seed(42)

    model = EncoderBlock(8, 2, 32)
    x = torch.randn(2, 4, 8, requires_grad=True)

    z, _ = model(x)

    loss = z.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_encoder_attention_normalization():
    model = EncoderBlock(8, 2, 32)
    x = torch.randn(2, 4, 8)

    _, weights = model(x)

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(2, 2, 4),
    )


def test_encoder_permutation_equivariance():
    torch.manual_seed(42)

    model = EncoderBlock(8, 2, 32).double()
    model.eval()

    x = torch.randn(2, 4, 8, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1])

    z1, _ = model(x)
    z2, _ = model(x[:, permutation, :])

    torch.testing.assert_close(
        z2,
        z1[:, permutation, :],
        rtol=1e-10,
        atol=1e-12,
    )

def test_encoder_dropout_eval():
    torch.manual_seed(42)

    model = EncoderBlock(
        d_model=8,
        num_heads=2,
        d_ff=32,
        dropout=0.1,
    )
    model.eval()

    x = torch.randn(2, 4, 8)

    with torch.no_grad():
        y1, _ = model(x)
        y2, _ = model(x)

    # evalモードでは同じ入力に同じ出力を返す
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)


def test_encoder_dropout_train():
    torch.manual_seed(42)

    model = EncoderBlock(
        d_model=8,
        num_heads=2,
        d_ff=32,
        dropout=0.5,
    )
    model.train()

    x = torch.randn(2, 4, 8)

    y1, _ = model(x)
    y2, _ = model(x)

    # 異なるDropoutマスクが使われることを確認
    assert not torch.equal(y1, y2)


def test_encoder_dropout_backward():
    model = EncoderBlock(8, 2, 32, dropout=0.1)
    model.train()

    x = torch.randn(2, 4, 8, requires_grad=True)

    y, _ = model(x)
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()