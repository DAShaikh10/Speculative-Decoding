"""
This module contains the sampling utilities used in speculative decoding.
"""

import torch


def standardize(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Cast a sampling method into a probability distribution (Section 2.2 of the paper).

    Args:
        logits: The raw model scores, shape (..., vocab_size). Operates on the last dimension.
        temperature: The sampling temperature. 0 means argmax, which gives a one-hot distribution.

    Returns:
        The probabilities, same shape as `logits`, summing to 1 along the last dimension.

    Raises:
        ValueError: If `temperature` is negative.
    """

    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")

    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(logits.dtype)

    return torch.softmax(logits / temperature, dim=-1)


def speculative_sample(p: torch.Tensor, q: torch.Tensor, x: int) -> tuple[int, bool]:
    """
    Speculative sampling for a single position (Section 2.3 of the paper).

    Given a token `x` sampled from `q`, return a token distributed according to `p`.

    Args:
        p: The target distribution, shape (vocab_size,).
        q: The draft distribution, shape (vocab_size,).
        x: The token sampled from `q`.

    Returns:
        A tuple of the resulting token and whether the draft token `x` was accepted.
    """

    # Accept x with probability min(1, p(x) / q(x)).
    r: float = torch.rand(()).item()
    if r * q[x] <= p[x]:
        return x, True

    # Otherwise, resample from the adjusted distribution norm(max(0, p - q)).
    residual: torch.Tensor = torch.clamp(p - q, min=0)
    total: torch.Tensor = residual.sum()
    if total <= 0:
        # Only reachable through floating point error when p == q; p is then the right distribution.
        residual, total = p, p.sum()

    return int(torch.multinomial(residual / total, 1).item()), False
