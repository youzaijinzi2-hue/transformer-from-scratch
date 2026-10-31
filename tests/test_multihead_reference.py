
import pytest
import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention


def make_models(D=8, H=2):
    # 自作モデル
    custom = MultiHeadAttention(D, H).double()

    # PyTorch公式モデル
    reference = nn.MultiheadAttention(
        embed_dim=D,
        num_heads=H,
        dropout=0.0,
        bias=False,
        batch_first=True,
    ).double()

    # 自作モデルの重みを公式モデルにコピー
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat(
                [
                    custom.W_Q.weight,
                    custom.W_K.weight,
                    custom.W_V.weight,
                ],
                dim=0,
            )
        )
        reference.out_proj.weight.copy_(
            custom.W_O.weight
        )

    return custom, reference


@pytest.mark.parametrize("causal", [False, True])
def test_forward_matches_pytorch(causal):
    torch.manual_seed(42)

    B, N, D, H = 2, 4, 8, 2
    custom, reference = make_models(D, H)

    X = torch.randn(B, N, D, dtype=torch.float64)

    allowed_mask = None
    blocked_mask = None

    if causal:
        allowed_mask = torch.ones(
            N, N, dtype=torch.bool
        ).tril()

        # PyTorch公式のマスクはTrueが禁止位置
        blocked_mask = ~allowed_mask

    Z_custom, A_custom = custom(
        X, mask=allowed_mask
    )

    Z_ref, A_ref = reference(
        X, X, X,
        attn_mask=blocked_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    torch.testing.assert_close(
        Z_custom, Z_ref,
        rtol=1e-10, atol=1e-12
    )

    torch.testing.assert_close(
        A_custom, A_ref,
        rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_pytorch(causal):
    torch.manual_seed(42)

    B, N, D, H = 2, 4, 8, 2
    custom, reference = make_models(D, H)

    X1 = torch.randn(
        B, N, D,
        dtype=torch.float64,
        requires_grad=True,
    )

    X2 = X1.detach().clone().requires_grad_(True)

    allowed_mask = None
    blocked_mask = None

    if causal:
        allowed_mask = torch.ones(
            N, N, dtype=torch.bool
        ).tril()
        blocked_mask = ~allowed_mask

    Z_custom, _ = custom(X1, mask=allowed_mask)

    Z_ref, _ = reference(
        X2, X2, X2,
        attn_mask=blocked_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    # 両モデルに同じ上流勾配を与える
    upstream = torch.randn_like(Z_custom)

    Z_custom.backward(upstream)
    Z_ref.backward(upstream)

    # 入力に対する勾配
    torch.testing.assert_close(
        X1.grad, X2.grad,
        rtol=1e-10, atol=1e-12
    )

    # Q, K, Vの重みに対する勾配
    custom_grads = torch.cat(
        [
            custom.W_Q.weight.grad,
            custom.W_K.weight.grad,
            custom.W_V.weight.grad,
        ],
        dim=0,
    )

    torch.testing.assert_close(
        custom_grads,
        reference.in_proj_weight.grad,
        rtol=1e-10, atol=1e-12,
    )

    # 出力射影の勾配
    torch.testing.assert_close(
        custom.W_O.weight.grad,
        reference.out_proj.weight.grad,
        rtol=1e-10, atol=1e-12,
    )
