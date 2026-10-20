"""
This module contains the sampling utilities used in speculative decoding.
"""

import torch
from typing_extensions import deprecated


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


def speculative_sample(p: torch.Tensor, q: torch.Tensor, guesses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Speculative sampling for all the guesses of one iteration at once (Algorithm 1 of the paper).

    Keeps the guesses up to the first rejection, and samples the token that follows them: from the adjusted distribution
    norm(max(0, p - q)) at the first rejected position, or from the last row of `p` if every guess was accepted. It does
    the same as calling `scalar_speculative_sample` position by position, but never leaves the device, so it doesn't make the
    CPU wait for the GPU.

    Args:
        p: The target distributions, shape (num_guesses + 1, vocab_size). Row i is the distribution of guess i.
        q: The draft distributions, shape (num_guesses, vocab_size).
        guesses: The tokens sampled from `q`, shape (num_guesses,).

    Returns:
        The number of accepted guesses and the token that follows them, both scalar tensors on the device of `p`.
    """

    num_guesses: int = guesses.shape[0]
    positions: torch.Tensor = torch.arange(num_guesses, device=p.device)

    # Accept guess i with probability min(1, p(x) / q(x)). The comparison is strict, so that a draw of exactly 0 can't
    # accept a token the target gives probability 0.
    # The comparison is done in float32 even for bfloat16 models, whose 8 bit draws would be too coarse.
    r: torch.Tensor = torch.rand(num_guesses, device=p.device)
    accepted: torch.Tensor = r * q[positions, guesses].float() < p[positions, guesses].float()
    num_accepted: torch.Tensor = accepted.long().cumprod(dim=0).sum()  # The length of the leading run of acceptances.

    # Indexing with a tensor, rather than an int, keeps the position on the device. The zero row pads `q` so that when
    # every guess is accepted the adjusted distribution is just the last row of `p`.
    index: torch.Tensor = num_accepted.view(1)
    p_next: torch.Tensor = p.index_select(0, index)[0]
    q_next: torch.Tensor = torch.cat([q, q.new_zeros(1, p.shape[-1])]).index_select(0, index)[0]
    residual: torch.Tensor = torch.clamp(p_next - q_next, min=0)
    # The residual is only empty through floating point error when p == q; p is then the right distribution.
    distribution: torch.Tensor = torch.where(residual.sum() > 0, residual, p_next)
    token: torch.Tensor = torch.multinomial(distribution / distribution.sum(), 1)[0]

    return num_accepted, token


@deprecated("Use `speculative_sample` instead, which uses vectorization and avoids CPU-GPU synchronization.")
def scalar_speculative_sample(p: torch.Tensor, q: torch.Tensor, x: int) -> tuple[int, bool]:
    """
    Scalar speculative sampling for a single position (Section 2.3 of the paper).

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
