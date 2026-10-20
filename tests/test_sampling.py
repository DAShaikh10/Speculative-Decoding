"""
Tests for the sampling utilities. See Appendix A.1 and Corollary 3.6 of the paper.
"""

import pytest
import torch

from speculative_decoding.sampling import scalar_speculative_sample, speculative_sample, standardize

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
def test_scalar_speculative_sample_matches_target_distribution(seed: int) -> None:
    """The returned tokens must follow p, not q, and acceptance must average sum(min(p, q))."""
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    p: torch.Tensor = random_distribution(generator)
    q: torch.Tensor = random_distribution(generator)

    drafts: torch.Tensor = torch.multinomial(q, NUM_SAMPLES, replacement=True, generator=generator)
    counts: torch.Tensor = torch.zeros(VOCAB_SIZE)
    accepted = 0
    for x in drafts.tolist():
        token, was_accepted = scalar_speculative_sample(p, q, x)
        counts[token] += 1
        accepted += was_accepted

    assert torch.allclose(counts / NUM_SAMPLES, p, atol=0.01)
    assert accepted / NUM_SAMPLES == pytest.approx(torch.minimum(p, q).sum().item(), abs=0.01)


def test_scalar_speculative_sample_zero_temperature_agreeing_models_accepts() -> None:
    p = q = standardize(torch.tensor([1.0, 5.0, 2.0]), temperature=0.0)
    assert scalar_speculative_sample(p, q, 1) == (1, True)


def test_scalar_speculative_sample_zero_temperature_disagreeing_models_returns_target_argmax() -> None:
    p: torch.Tensor = standardize(torch.tensor([1.0, 5.0, 2.0]), temperature=0.0)  # argmax 1
    q: torch.Tensor = standardize(torch.tensor([6.0, 1.0, 2.0]), temperature=0.0)  # argmax 0
    for _ in range(20):
        assert scalar_speculative_sample(p, q, 0) == (1, False)


def test_scalar_speculative_sample_identical_distributions_always_accepts() -> None:
    torch.manual_seed(0)
    p: torch.Tensor = random_distribution(torch.Generator().manual_seed(0))
    for x in range(VOCAB_SIZE):
        assert scalar_speculative_sample(p, p.clone(), x) == (x, True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_speculative_sample_first_token_matches_target_distribution(seed: int) -> None:
    """With one guess, the first emitted token must follow p, and acceptance must average sum(min(p, q))."""
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    p: torch.Tensor = torch.stack([random_distribution(generator), random_distribution(generator)])
    q: torch.Tensor = random_distribution(generator).unsqueeze(0)

    drafts: torch.Tensor = torch.multinomial(q[0], NUM_SAMPLES, replacement=True, generator=generator)
    counts: torch.Tensor = torch.zeros(VOCAB_SIZE)
    accepted = 0
    for x in drafts.tolist():
        num_accepted, token = speculative_sample(p, q, torch.tensor([x]))
        # The first emitted token is the guess if it was accepted, and the replacement token otherwise.
        counts[x if num_accepted.item() else token.item()] += 1
        accepted += num_accepted.item()

    assert torch.allclose(counts / NUM_SAMPLES, p[0], atol=0.01)
    assert accepted / NUM_SAMPLES == pytest.approx(torch.minimum(p[0], q[0]).sum().item(), abs=0.01)


def test_speculative_sample_without_guesses_samples_from_target() -> None:
    torch.manual_seed(0)
    p: torch.Tensor = random_distribution(torch.Generator().manual_seed(0)).unsqueeze(0)
    q: torch.Tensor = torch.empty(0, VOCAB_SIZE)

    tokens: list[int] = [speculative_sample(p, q, torch.empty(0, dtype=torch.long))[1].item() for _ in range(20_000)]

    assert torch.allclose(torch.bincount(torch.tensor(tokens), minlength=VOCAB_SIZE) / 20_000, p[0], atol=0.02)
    assert speculative_sample(p, q, torch.empty(0, dtype=torch.long))[0].item() == 0


def test_speculative_sample_identical_distributions_accept_everything_and_sample_the_extra_token() -> None:
    torch.manual_seed(0)
    generator: torch.Generator = torch.Generator().manual_seed(0)
    p: torch.Tensor = torch.stack([random_distribution(generator) for _ in range(4)])
    q: torch.Tensor = p[:3].clone()
    guesses: torch.Tensor = torch.tensor([0, 3, 1])

    num_accepted, token = speculative_sample(p, q, guesses)

    assert num_accepted.item() == 3
    assert 0 <= token.item() < VOCAB_SIZE


def test_speculative_sample_zero_temperature_stops_at_first_disagreement() -> None:
    logits: torch.Tensor = torch.tensor([[0.0, 9.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 9.0], [0.0, 9.0, 0.0]])
    p: torch.Tensor = standardize(logits, temperature=0.0)  # The target argmaxes are 1, 0, 2, 1.
    q: torch.Tensor = standardize(torch.tensor([[0.0, 9.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]), temperature=0.0)
    guesses: torch.Tensor = torch.tensor([1, 1, 2])  # The second guess is wrong, so the third is never reached.

    for _ in range(20):
        num_accepted, token = speculative_sample(p, q, guesses)
        assert (num_accepted.item(), token.item()) == (1, 0)


def test_speculative_sample_zero_temperature_all_correct_takes_the_target_extra_token() -> None:
    p: torch.Tensor = standardize(torch.tensor([[0.0, 9.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 9.0]]), temperature=0.0)
    q: torch.Tensor = p[:2].clone()

    for _ in range(20):
        num_accepted, token = speculative_sample(p, q, torch.tensor([1, 0]))
        assert (num_accepted.item(), token.item()) == (2, 2)


def algorithm_1_accepted_count(p: torch.Tensor, q: torch.Tensor, guesses: list[int], r: torch.Tensor) -> int:
    """
    The number of accepted guesses n, transcribed literally from Algorithm 1 of the paper (1-based in the paper):
    n = min({i - 1 | 1 <= i <= gamma, r_i > p_i(x_i) / q_i(x_i)} + {gamma}).
    """

    gamma: int = len(guesses)
    rejected: list[int] = [
        i for i in range(1, gamma + 1) if r[i - 1] > p[i - 1][guesses[i - 1]] / q[i - 1][guesses[i - 1]]
    ]
    return min([i - 1 for i in rejected] + [gamma])


@pytest.mark.parametrize("gamma", [1, 3, 6])
def test_speculative_sample_matches_algorithm_1_with_the_same_random_draws(gamma: int) -> None:
    """Fed the same uniform draws, the vectorized version must accept exactly the guesses Algorithm 1 accepts."""
    generator: torch.Generator = torch.Generator().manual_seed(gamma)
    for trial in range(300):
        p: torch.Tensor = torch.stack([random_distribution(generator) for _ in range(gamma + 1)])
        q: torch.Tensor = torch.stack([random_distribution(generator) for _ in range(gamma)])
        guesses: torch.Tensor = torch.stack([torch.multinomial(q[i], 1, generator=generator)[0] for i in range(gamma)])

        torch.manual_seed(trial)
        expected: int = algorithm_1_accepted_count(p, q, guesses.tolist(), torch.rand(gamma))
        torch.manual_seed(trial)  # speculative_sample draws the same r first.
        num_accepted, _ = speculative_sample(p, q, guesses)

        assert num_accepted.item() == expected


def test_speculative_sample_replacement_follows_the_adjusted_distribution() -> None:
    """After a rejection at position n+1 the token must follow norm(max(0, p_{n+1} - q_{n+1})), as in Algorithm 1."""
    torch.manual_seed(0)
    generator: torch.Generator = torch.Generator().manual_seed(0)
    p: torch.Tensor = torch.stack([random_distribution(generator) for _ in range(2)])
    q: torch.Tensor = random_distribution(generator).unsqueeze(0)
    guess: torch.Tensor = torch.tensor(
        [int(torch.argmax(q[0] - p[0]))]
    )  # A token the draft over-weights, so it is often rejected.

    adjusted: torch.Tensor = torch.clamp(p[0] - q[0], min=0)
    adjusted = adjusted / adjusted.sum()

    counts: torch.Tensor = torch.zeros(VOCAB_SIZE)
    rejections: int = 0
    for _ in range(60_000):
        num_accepted, token = speculative_sample(p, q, guess)
        if num_accepted.item() == 0:
            counts[token.item()] += 1
            rejections += 1

    assert rejections > 5_000
    assert torch.allclose(counts / rejections, adjusted, atol=0.015)
