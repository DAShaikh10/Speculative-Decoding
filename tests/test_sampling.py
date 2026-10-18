"""
Tests for the sampling utilities. See Appendix A.1 and Corollary 3.6 of the paper.
"""

import pytest
import torch

from speculative_decoding.sampling import speculative_sample, standardize

VOCAB_SIZE: int = 5
NUM_SAMPLES: int = 100_000


def random_distribution(generator: torch.Generator) -> torch.Tensor:
    return torch.softmax(torch.randn(VOCAB_SIZE, generator=generator), dim=-1)


def test_standardize_sums_to_one() -> None:
    probs: torch.Tensor = standardize(torch.randn(3, 32), temperature=1.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))


def test_standardize_zero_temperature_is_one_hot_argmax() -> None:
    logits: torch.Tensor = torch.randn(3, 32)
    probs: torch.Tensor = standardize(logits, temperature=0.0)
    assert torch.equal(probs.argmax(dim=-1), logits.argmax(dim=-1))
    assert torch.equal(probs.max(dim=-1).values, torch.ones(3))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))


def test_standardize_rejects_negative_temperature() -> None:
    with pytest.raises(ValueError):
        standardize(torch.randn(8), temperature=-1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_speculative_sample_matches_target_distribution(seed: int) -> None:
    """The returned tokens must follow p, not q, and acceptance must average sum(min(p, q))."""
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    p: torch.Tensor = random_distribution(generator)
    q: torch.Tensor = random_distribution(generator)

    drafts: torch.Tensor = torch.multinomial(q, NUM_SAMPLES, replacement=True, generator=generator)
    counts: torch.Tensor = torch.zeros(VOCAB_SIZE)
    accepted = 0
    for x in drafts.tolist():
        token, was_accepted = speculative_sample(p, q, x)
        counts[token] += 1
        accepted += was_accepted

    assert torch.allclose(counts / NUM_SAMPLES, p, atol=0.01)
    assert accepted / NUM_SAMPLES == pytest.approx(torch.minimum(p, q).sum().item(), abs=0.01)


def test_speculative_sample_zero_temperature_agreeing_models_accepts() -> None:
    p = q = standardize(torch.tensor([1.0, 5.0, 2.0]), temperature=0.0)
    assert speculative_sample(p, q, 1) == (1, True)


def test_speculative_sample_zero_temperature_disagreeing_models_returns_target_argmax() -> None:
    p: torch.Tensor = standardize(torch.tensor([1.0, 5.0, 2.0]), temperature=0.0)  # argmax 1
    q: torch.Tensor = standardize(torch.tensor([6.0, 1.0, 2.0]), temperature=0.0)  # argmax 0
    for _ in range(20):
        assert speculative_sample(p, q, 0) == (1, False)


def test_speculative_sample_identical_distributions_always_accepts() -> None:
    torch.manual_seed(0)
    p: torch.Tensor = random_distribution(torch.Generator().manual_seed(0))
    for x in range(VOCAB_SIZE):
        assert speculative_sample(p, p.clone(), x) == (x, True)
