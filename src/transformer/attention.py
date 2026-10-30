
import math

import torch
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
