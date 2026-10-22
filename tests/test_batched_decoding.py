"""
Tests for batched decoding. They use tiny randomly initialised T5 models (see tiny_models.py), so they need neither a
GPU nor a download.
"""

import copy
from collections import Counter

import pytest
import torch
from tiny_models import make_t5, pad_batch, perturbed_copy
from transformers import T5ForConditionalGeneration

from speculative_decoding.batched_decoding import (
    BatchedDecoderState,
    BatchedDecodingStats,
    batched_autoregressive_decode,
    batched_speculative_decode,
    batched_speculative_sample,
)
from speculative_decoding.decoding import autoregressive_decode
from speculative_decoding.metrics import expected_tokens_per_iteration

PROMPTS: list[list[int]] = [[5, 6, 7, 1], [8, 9, 1], [3, 4, 5, 6, 7, 8, 9, 1], [12, 2, 30, 31, 1], [7, 1]]


@pytest.fixture(scope="module")
def target() -> T5ForConditionalGeneration:
    return make_t5(seed=1)


@pytest.fixture(scope="module")
def draft(target: T5ForConditionalGeneration) -> T5ForConditionalGeneration:
    return perturbed_copy(target, noise=0.08)


# The decoder state ------------------------------------------------------------------------------------------------


@torch.inference_mode()
def test_decoder_state_matches_recomputing_everything(target: T5ForConditionalGeneration) -> None:
    """Growing and rolling back the KV cache must give the logits of a full pass without any cache, with padding."""

    ids, mask = pad_batch(PROMPTS)
    state: BatchedDecoderState = BatchedDecoderState(target, ids, mask)

    def check(first_new_position: int) -> None:
        logits: torch.Tensor = state.logits()
        full: torch.Tensor = target(
            input_ids=ids, attention_mask=mask, decoder_input_ids=state.buffer[:, : state.length]
        ).logits
        torch.testing.assert_close(logits, full[:, first_new_position:], atol=1e-4, rtol=1e-4)

    state.extend(torch.randint(2, 40, (len(PROMPTS), 5)))
    check(0)
    state.truncate(3)  # Rolls the cache back.
    state.extend(torch.randint(2, 40, (len(PROMPTS), 4)))
    check(3)
    state.truncate(5)
    state.extend(torch.randint(2, 40, (len(PROMPTS), 0)))  # Extending by nothing leaves it as it is.
    state.extend(torch.randint(2, 40, (len(PROMPTS), 3)))
    check(5)
    state.truncate(len(state) + 2)  # Truncating to a longer length than the sequence only sets the length.
    assert len(state) == 10


def test_decoder_state_rejects_bad_input(target: T5ForConditionalGeneration) -> None:
    ids, mask = pad_batch(PROMPTS)
    with pytest.raises(ValueError):
        BatchedDecoderState(target, ids, mask[:, :-1])
    state = BatchedDecoderState(target, ids, mask)
    state.logits()
    with pytest.raises(ValueError):  # Nothing new to process.
        state.logits()


# Greedy decoding is deterministic, so the decoders must agree exactly ----------------------------------------------


@torch.inference_mode()
def test_batched_baseline_matches_the_batch_size_1_decoder(target: T5ForConditionalGeneration) -> None:
    ids, mask = pad_batch(PROMPTS)
    alone = [autoregressive_decode(target, torch.tensor([prompt]), 0.0, 16, True, False) for prompt in PROMPTS]
    assert batched_autoregressive_decode(target, ids, mask, 0.0, 16) == alone  # Padded rows, so the mask matters.
    assert len({tuple(tokens) for tokens in alone}) > 1  # The test isn't vacuous: the prompts give different outputs.
    assert len({token for tokens in alone for token in tokens}) > 3


@pytest.mark.parametrize("gamma", [1, 3, 5])
@pytest.mark.parametrize("batch_size", [1, 3, 5])
@torch.inference_mode()
def test_greedy_speculative_decoding_equals_the_baseline(
    target: T5ForConditionalGeneration, draft: T5ForConditionalGeneration, gamma: int, batch_size: int
) -> None:
    ids, mask = pad_batch(PROMPTS[:batch_size])
    baseline = batched_autoregressive_decode(target, ids, mask, 0.0, 20)
    outputs, stats = batched_speculative_decode(target, draft, ids, mask, 0.0, gamma, 20)

    assert outputs == baseline
    assert 0 < stats.acceptance_rate < 1  # Both accepting and rejecting happened.


@torch.inference_mode()
def test_rows_that_finish_at_different_times(
    target: T5ForConditionalGeneration, draft: T5ForConditionalGeneration
) -> None:
    """Finished rows stay in the batch, must not limit the others, and must not leak tokens into their outputs."""

    ids, mask = pad_batch(PROMPTS)
    probe = batched_autoregressive_decode(target, ids, mask, 0.0, 12)
    # Make the token that the second row emits as its 4th the end-of-sequence token, so rows end at different positions.
    end: int = probe[1][3]
    original = target.config.eos_token_id
    target.config.eos_token_id = end
    try:
        baseline = batched_autoregressive_decode(target, ids, mask, 0.0, 12)
        outputs, _ = batched_speculative_decode(target, draft, ids, mask, 0.0, 4, 12)
    finally:
        target.config.eos_token_id = original

    assert len({len(tokens) for tokens in baseline}) > 1, "the rows should finish at different positions"
    assert outputs == baseline
    for tokens in outputs:
        assert end not in tokens[:-1]  # The end-of-sequence token only ends a sequence.


@torch.inference_mode()
def test_generation_limit_and_sequence_ends_when_sampling(target: T5ForConditionalGeneration, draft) -> None:
    ids, mask = pad_batch(PROMPTS)
    for seed in range(5):
        torch.manual_seed(seed)
        for max_new_tokens in (1, 2, 7):
            outputs, stats = batched_speculative_decode(target, draft, ids, mask, 1.0, 4, max_new_tokens)
            for tokens in outputs:
                assert 1 <= len(tokens) <= max_new_tokens
                assert 1 not in tokens[:-1]  # Token 1 is the end-of-sequence token.
            assert stats.positions <= max_new_tokens


@torch.inference_mode()
def test_identical_models_accept_every_guess(target: T5ForConditionalGeneration) -> None:
    ids, mask = pad_batch(PROMPTS)
    torch.manual_seed(0)
    outputs, stats = batched_speculative_decode(target, copy.deepcopy(target), ids, mask, 1.0, 4, 12)

    assert stats.rejected == 0
    assert stats.acceptance_rate == 1.0
    assert stats.accepted == stats.drafted
    assert all(len(tokens) >= 1 for tokens in outputs)


def test_decoding_rejects_bad_arguments(target: T5ForConditionalGeneration, draft) -> None:
    ids, mask = pad_batch(PROMPTS)
    with pytest.raises(ValueError):
        batched_speculative_decode(target, draft, ids, mask, 0.0, 0, 10)
    with pytest.raises(ValueError):
        batched_speculative_decode(target, draft, ids, mask, 0.0, 3, 0)
    with pytest.raises(ValueError):
        batched_autoregressive_decode(target, ids, mask, 0.0, 0)


# The batched sampler ----------------------------------------------------------------------------------------------


def test_sampler_accepts_everything_when_the_distributions_are_equal() -> None:
    torch.manual_seed(0)
    q = torch.softmax(torch.randn(6, 4, 9), dim=-1)
    p = torch.cat([q, torch.softmax(torch.randn(6, 1, 9), dim=-1)], dim=1)
    guesses = torch.multinomial(q.reshape(-1, 9), 1).reshape(6, 4)

    accepted, token = batched_speculative_sample(p, q, guesses)

    assert accepted.tolist() == [4] * 6
    assert token.shape == (6,)


def test_sampler_rejects_a_token_the_target_gives_no_probability() -> None:
    q = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]] * 3)  # Shape (3, 1, 4).
    p = torch.tensor([[[0.0, 0.0, 0.5, 0.5]] * 2] * 3)  # Disjoint from q, shape (3, 2, 4).
    guesses = torch.tensor([[0], [1], [0]])

    for _ in range(20):
        accepted, token = batched_speculative_sample(p, q, guesses)
        assert accepted.tolist() == [0, 0, 0]
        assert set(token.tolist()) <= {2, 3}  # Resampled from the target, never from the draft.


def test_sampler_without_guesses_samples_from_the_target() -> None:
    p = torch.tensor([[[0.0, 1.0, 0.0]]] * 4)  # Shape (4, 1, 3).
    accepted, token = batched_speculative_sample(p, torch.empty((4, 0, 3)), torch.empty((4, 0), dtype=torch.long))

    assert accepted.tolist() == [0] * 4
    assert token.tolist() == [1] * 4


def test_sampler_with_one_hot_distributions_is_greedy_verification() -> None:
    eye = torch.eye(5)
    p = eye[torch.tensor([[2, 3, 4], [2, 3, 4]])]  # The target's argmaxes: (2, 3, 5).
    q = eye[torch.tensor([[2, 3], [2, 0]])]  # The draft agrees on 2 guesses in row 0 and on 1 in row 1.
    guesses = torch.tensor([[2, 3], [2, 0]])

    accepted, token = batched_speculative_sample(p, q, guesses)

    assert accepted.tolist() == [2, 1]
    assert token.tolist() == [4, 3]  # The target's own token at the first disagreement, or after the last guess.


def test_sampler_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError):
        batched_speculative_sample(torch.ones(2, 3, 5), torch.ones(2, 3, 5), torch.zeros((2, 3), dtype=torch.long))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_lockstep_advance_follows_equation_1_with_alpha_to_the_batch_size(batch_size: int) -> None:
    """
    If every sequence accepts each guess independently with probability alpha, the batch advances by 1 + min(n_b), whose
    mean is Equation (1) at alpha^B: P(min >= k) = alpha^(k B).
    """

    gamma, trials = 4, 40000
    p_row = torch.tensor([0.5, 0.3, 0.2, 0.0])
    q_row = torch.tensor([0.3, 0.5, 0.0, 0.2])
    alpha: float = torch.minimum(p_row, q_row).sum().item()  # 0.6

    torch.manual_seed(0)
    rows: int = trials * batch_size
    q = q_row.expand(rows, gamma, 4)
    p = p_row.expand(rows, gamma + 1, 4)
    guesses = torch.multinomial(q_row.expand(rows * gamma, 4), 1).reshape(rows, gamma)

    accepted, _ = batched_speculative_sample(p, q, guesses)
    advance = 1 + accepted.reshape(trials, batch_size).min(dim=1).values.float().mean().item()

    assert advance == pytest.approx(expected_tokens_per_iteration(alpha**batch_size, gamma), abs=0.02)
    # The acceptance rate itself doesn't depend on the batch size.
    assert (accepted.float().mean().item()) == pytest.approx(sum(alpha**k for k in range(1, gamma + 1)), abs=0.02)


# The output distribution ------------------------------------------------------------------------------------------


def total_variation(first: Counter, second: Counter) -> float:
    keys = set(first) | set(second)
    n_first, n_second = sum(first.values()), sum(second.values())
    return 0.5 * sum(abs(first[key] / n_first - second[key] / n_second) for key in keys)


@torch.inference_mode()
def test_speculative_outputs_have_the_distribution_of_the_target() -> None:
    """
    Sample many sequences with the baseline, with speculative decoding in lockstep and with the draft alone, and compare
    the empirical distributions of the first two tokens jointly and of the later positions one by one. The draft's own
    distribution is far from the target's, so a sampler that leaked it would fail.
    """

    target = make_t5(seed=1, vocab_size=6, scale=0.3)
    draft = perturbed_copy(target, noise=0.6)
    batch_size, calls, max_new_tokens = 16, 700, 4
    prompt = torch.tensor([[2, 3, 4, 5, 1]]).repeat(batch_size, 1)
    mask = torch.ones_like(prompt)

    def sample(decode) -> list[tuple[int, ...]]:
        torch.manual_seed(1234)
        sequences: list[tuple[int, ...]] = []
        for _ in range(calls):
            sequences += [tuple(tokens) for tokens in decode()]
        return sequences

    baseline = sample(lambda: batched_autoregressive_decode(target, prompt, mask, 1.0, max_new_tokens))
    draft_only = sample(lambda: batched_autoregressive_decode(draft, prompt, mask, 1.0, max_new_tokens))
    stats = BatchedDecodingStats()

    def speculative():
        outputs, batch_stats = batched_speculative_decode(target, draft, prompt, mask, 1.0, 3, max_new_tokens)
        nonlocal stats
        stats += batch_stats
        return outputs

    lockstep = sample(speculative)

    def joint(sequences):
        return Counter(sequence[:2] for sequence in sequences)

    def position(sequences, index):
        return Counter(sequence[index] if len(sequence) > index else -1 for sequence in sequences)

    # The noise of the distance between two empirical distributions of this size is about 0.02 for the joint ones.
    assert total_variation(joint(lockstep), joint(baseline)) < 0.05
    for index in (2, 3):
        assert total_variation(position(lockstep, index), position(baseline, index)) < 0.05

    # The test has teeth: the draft alone would be told apart, and the lockstep did accept and reject guesses.
    assert total_variation(joint(draft_only), joint(baseline)) > 0.15
    assert 0.2 < stats.acceptance_rate < 0.95
