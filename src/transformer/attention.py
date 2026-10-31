
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model <= 0 or num_heads <= 0:
            raise ValueError("Dimensions must be positive.")

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learnable linear transformations
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor):
        B, N, D = x.shape

        # (B, N, D) -> (B, N, H, d_k)
        x = x.reshape(
            B, N, self.num_heads, self.d_k
        )

        # (B, N, H, d_k) -> (B, H, N, d_k)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        B, N, D = x.shape

        # Generate Q, K, V
        Q = self.split_heads(self.W_Q(x))
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))

        # Attention for each head
        Z, A = scaled_dot_product_attention(
            Q, K, V, mask=mask
        )

        # (B, H, N, d_k) -> (B, N, H, d_k)
        Z = Z.transpose(1, 2).contiguous()

        # Concatenate all heads
        Z = Z.reshape(B, N, D)

        # Output projection
        Z = self.W_O(Z)

        return Z, A
