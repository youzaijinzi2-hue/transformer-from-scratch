
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


def make_target_mask(
    token_ids: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Combine causal and padding masks.

    Args:
        token_ids: (B, N)
        pad_id: padding token ID

    Returns:
        mask: (B, 1, N, N)
              True  = allowed
              False = blocked
    """
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (B, N).")

    B, N = token_ids.shape

    if N == 0:
        raise ValueError("Target sequence must not be empty.")

    # (B, 1, 1, N)
    padding_mask = make_padding_mask(
        token_ids,
        pad_id=pad_id,
    )

    # (N, N)
    causal_mask = torch.ones(
        N, N,
        dtype=torch.bool,
        device=token_ids.device,
    ).tril()

    # (B, 1, N, N)
    target_mask = padding_mask & causal_mask[None, None, :, :]

    return target_mask
