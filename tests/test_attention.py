
import torch

from transformer.attention import scaled_dot_product_attention


def test_attention_shape():
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 5, 8)
    V = torch.randn(2, 5, 6)

    Z, A = scaled_dot_product_attention(Q, K, V)

    assert Z.shape == (2, 4, 6)
    assert A.shape == (2, 4, 5)


def test_attention_normalization():
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)

    _, A = scaled_dot_product_attention(Q, K, V)

    torch.testing.assert_close(
        A.sum(dim=-1),
        torch.ones(2, 4),
    )


def test_causal_mask():
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)

    mask = torch.tril(
        torch.ones(4, 4, dtype=torch.bool)
    )

    _, A = scaled_dot_product_attention(
        Q, K, V, mask=mask
    )

    future_weights = A[0].triu(diagonal=1)

    assert torch.all(future_weights == 0)
