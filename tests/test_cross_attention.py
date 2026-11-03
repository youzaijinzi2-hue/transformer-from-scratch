
import torch
from torch import nn

from transformer.attention import MultiHeadAttention
from transformer.masking import make_padding_mask


def test_cross_attention_matches_pytorch():
    torch.manual_seed(42)

    B, Nq, Nk, D, H = 2, 3, 5, 8, 2

    custom = MultiHeadAttention(D, H).double()

    reference = nn.MultiheadAttention(
        embed_dim=D,
        num_heads=H,
        dropout=0.0,
        bias=False,
        batch_first=True,
    ).double()

    # Copy parameters
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat([
                custom.W_Q.weight,
                custom.W_K.weight,
                custom.W_V.weight,
            ], dim=0)
        )
        reference.out_proj.weight.copy_(
            custom.W_O.weight
        )

    decoder = torch.randn(
        B, Nq, D,
        dtype=torch.float64,
        requires_grad=True,
    )
    encoder = torch.randn(
        B, Nk, D,
        dtype=torch.float64,
        requires_grad=True,
    )

    decoder_ref = decoder.detach().clone().requires_grad_(True)
    encoder_ref = encoder.detach().clone().requires_grad_(True)

    # Source sequence: second batch has two PAD tokens
    src_tokens = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 0, 0],
    ])

    allowed_mask = make_padding_mask(src_tokens)

    # Custom cross-attention
    z1, a1 = custom(
        decoder,
        context=encoder,
        mask=allowed_mask,
    )

    # PyTorch: True means blocked
    key_padding_mask = src_tokens == 0

    z2, a2 = reference(
        query=decoder_ref,
        key=encoder_ref,
        value=encoder_ref,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    # Shape checks
    assert z1.shape == (B, Nq, D)
    assert a1.shape == (B, H, Nq, Nk)

    # Forward comparison
    torch.testing.assert_close(
        z1, z2, rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        a1, a2, rtol=1e-10, atol=1e-12
    )

    # PAD key positions must receive zero attention
    assert torch.all(a1[1, :, :, 3:] == 0)

    # Backward comparison
    upstream = torch.randn_like(z1)

    z1.backward(upstream)
    z2.backward(upstream)

    torch.testing.assert_close(
        decoder.grad,
        decoder_ref.grad,
        rtol=1e-10, atol=1e-12,
    )

    torch.testing.assert_close(
        encoder.grad,
        encoder_ref.grad,
        rtol=1e-10, atol=1e-12,
    )

    # Parameter gradients
    custom_qkv_grads = torch.cat([
        custom.W_Q.weight.grad,
        custom.W_K.weight.grad,
        custom.W_V.weight.grad,
    ], dim=0)

    torch.testing.assert_close(
        custom_qkv_grads,
        reference.in_proj_weight.grad,
        rtol=1e-10, atol=1e-12,
    )

    torch.testing.assert_close(
        custom.W_O.weight.grad,
        reference.out_proj.weight.grad,
        rtol=1e-10, atol=1e-12,
    )
