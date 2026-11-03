
import torch


def make_padding_mask(
    token_ids: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Create a boolean key-padding mask.

    Args:
        token_ids: (B, N)
        pad_id: padding token ID

    Returns:
        mask: (B, 1, 1, N)
              True  = allowed
              False = masked
    """
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (B, N).")

    valid = token_ids != pad_id  # (B, N)

    if not valid.any(dim=-1).all():
        raise ValueError("Each sequence must contain a non-PAD token.")

    return valid[:, None, None, :]
