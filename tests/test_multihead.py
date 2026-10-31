
import torch

from transformer.attention import MultiHeadAttention


def test_multihead_shape():
    mha = MultiHeadAttention(
        d_model=8,
        num_heads=2
    )

    X = torch.randn(2, 4, 8)

    Z, A = mha(X)

    assert Z.shape == (2, 4, 8)
    assert A.shape == (2, 2, 4, 4)


def test_multihead_normalization():
    mha = MultiHeadAttention(8, 2)
    X = torch.randn(2, 4, 8)

    _, A = mha(X)

    torch.testing.assert_close(
        A.sum(dim=-1),
        torch.ones(2, 2, 4)
    )


def test_multihead_gradient():
    mha = MultiHeadAttention(8, 2)
    X = torch.randn(2, 4, 8)

    Z, _ = mha(X)
    loss = Z.square().mean()
    loss.backward()

    for parameter in mha.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_multihead_causal_mask():
    mha = MultiHeadAttention(8, 2)
    X = torch.randn(2, 4, 8)

    mask = torch.tril(
        torch.ones(4, 4, dtype=torch.bool)
    )

    _, A = mha(X, mask=mask)

    assert torch.all(
        A.triu(diagonal=1) == 0
    )
