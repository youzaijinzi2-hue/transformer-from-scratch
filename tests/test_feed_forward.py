
import torch
from torch import nn

from transformer.feed_forward import FeedForward


def test_ffn_forward_and_backward():
    torch.manual_seed(42)

    custom = FeedForward(8, 32).double()

    reference = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    ).double()

    # パラメータを一致させる
    reference[0].load_state_dict(custom.linear1.state_dict())
    reference[2].load_state_dict(custom.linear2.state_dict())

    x1 = torch.randn(2, 4, 8, dtype=torch.float64,
                     requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = custom(x1)
    y2 = reference(x2)

    assert y1.shape == (2, 4, 8)
    torch.testing.assert_close(y1, y2)

    upstream = torch.randn_like(y1)
    y1.backward(upstream)
    y2.backward(upstream)

    torch.testing.assert_close(x1.grad, x2.grad)

    for p1, p2 in zip(custom.parameters(), reference.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad)
