
import math

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q: (B, Nq, dk)
    K: (B, Nk, dk)
    V: (B, Nk, dv)
    mask: (Nq, Nk) or broadcastable boolean tensor
          True means the position is allowed.

    Returns:
        Z: (B, Nq, dv)
        A: (B, Nq, Nk)
    """

    # Step 1: Query-Key compatibility
    dk = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(dk)

    # Step 2: Apply optional mask
    if mask is not None:
        # ~はビット反転演算子
        # masked_fill(condition, value)は、conditionがTrueである位置を指定した値に置き換える関数。
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 3: Normalize scores
    A = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of Values
    Z = A @ V

    return Z, A

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    x:       (B, Nq, D)
    context: (B, Nk, D)
    mask:    broadcastable to (B, H, Nq, Nk)
             True = allowed
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        return x.reshape(
            B, N, self.num_heads, self.d_k
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Self-attention when context is not supplied
        if context is None:
            context = x

        B, Nq, D = x.shape
        Bk, Nk, Dk = context.shape

        if B != Bk or D != self.d_model or Dk != D:
            raise ValueError("Incompatible input dimensions.")

        # Query from x, Key and Value from context
        Q = self.split_heads(self.W_Q(x))
        K = self.split_heads(self.W_K(context))
        V = self.split_heads(self.W_V(context))

        # Q: (B, H, Nq, d_k)
        # K: (B, H, Nk, d_k)
        # V: (B, H, Nk, d_k)

        Z, A = scaled_dot_product_attention(
            Q, K, V, mask=mask
        )

        # Z: (B, H, Nq, d_k)
        # A: (B, H, Nq, Nk)

        Z = Z.transpose(1, 2).contiguous()
        Z = Z.reshape(B, Nq, D)

        return self.W_O(Z), A
