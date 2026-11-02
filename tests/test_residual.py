
# Residual Connectionの加算がAutogradに正しく認識されていることを確認
import torch


def test_residual_identity_gradient():
    x = torch.randn(
        2, 4, 8,
        dtype=torch.float64,
        requires_grad=True,
    )

    # F(x) = 0 なら y = x
    y = x + torch.zeros_like(x)

    torch.testing.assert_close(y, x)

    y.sum().backward()

    # dy/dx = I なので d(sum(y))/dx = 1
    torch.testing.assert_close(
        x.grad,
        torch.ones_like(x),
    )


def test_residual_linear_gradient():
    x = torch.randn(
        2, 4, 8,
        dtype=torch.float64,
        requires_grad=True,
    )

    # F(x) = 2x
    y = x + 2 * x

    y.sum().backward()

    # y = 3x なので勾配は3
    torch.testing.assert_close(
        x.grad,
        torch.full_like(x, 3.0),
    )
