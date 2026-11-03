
import torch
from torch import nn

from transformer.normalization import LayerNormalization


def test_layernorm_forward():
    torch.manual_seed(42)

    custom = LayerNormalization(8).double()
    reference = nn.LayerNorm(8).double()

    x = torch.randn(2, 4, 8, dtype=torch.float64)

    y_custom = custom(x)
    y_reference = reference(x)

    torch.testing.assert_close(
        y_custom,
        y_reference,
        rtol=1e-10,
        atol=1e-12,
    )


def test_layernorm_backward():
    torch.manual_seed(42)

    custom = LayerNormalization(8).double()
    reference = nn.LayerNorm(8).double()

    x1 = torch.randn(
        2, 4, 8, dtype=torch.float64,
        requires_grad=True,
    )
    x2 = x1.detach().clone().requires_grad_(True)

    upstream = torch.randn_like(x1)

    custom(x1).backward(upstream)
    reference(x2).backward(upstream)

    torch.testing.assert_close(
        x1.grad, x2.grad,
        rtol=1e-10, atol=1e-12,
    )

    torch.testing.assert_close(
        custom.gamma.grad,
        reference.weight.grad,
        rtol=1e-10, atol=1e-12,
    )

    torch.testing.assert_close(
        custom.beta.grad,
        reference.bias.grad,
        rtol=1e-10, atol=1e-12,
    )
