
import torch

from transformer.encoder import TransformerEncoder
from transformer.positional_encoding import PositionalEncoding


def test_positional_encoding_shape():
    pe = PositionalEncoding(d_model=8, max_len=100)
    x = torch.zeros(2, 10, 8)

    y = pe(x)

    assert y.shape == (2, 10, 8)
    torch.testing.assert_close(y[0], y[1])


def test_positional_encoding_not_trainable():
    pe = PositionalEncoding(8)

    assert len(list(pe.parameters())) == 0


def test_encoder_forward():
    model = TransformerEncoder(
        vocab_size=100,
        d_model=16,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        dropout=0.0,
    )

    tokens = torch.randint(0, 100, (2, 10))

    output, attention_maps = model(tokens)

    assert output.shape == (2, 10, 16)
    assert len(attention_maps) == 2

    for A in attention_maps:
        assert A.shape == (2, 4, 10, 10)
        torch.testing.assert_close(
            A.sum(dim=-1),
            torch.ones(2, 4, 10),
        )


def test_encoder_backward():
    model = TransformerEncoder(
        vocab_size=100,
        d_model=16,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        dropout=0.0,
    )

    tokens = torch.randint(0, 100, (2, 10))

    output, _ = model(tokens)

    # 非対称な損失で勾配伝播を確認
    target = torch.randn_like(output)
    loss = (output - target).square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
