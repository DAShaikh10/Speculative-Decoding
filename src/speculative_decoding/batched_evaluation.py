"""
This module contains the benchmark that compares batched speculative decoding to the batched autoregressive baseline at
a given batch size. It is the batched counterpart of `evaluation.py`, which it leaves untouched.

Besides the speedup it reports where the speedup comes from, in the terms of the paper:

- alpha, the acceptance rate (Corollary 3.6), measured per sequence.
- The tokens per iteration of the batch, against Equation (1) for one sequence and against Equation (1) with alpha^B,
  which is what the lockstep batch of B sequences yields if every sequence accepts each guess with the probability alpha
  independently (the batch only advances as far as the sequence that rejects first; see `batched_decoding.py`).
- With `predict`, the cost coefficient c(B) and the verification ratio r(B) of `batch_scaling.py`, and the speedup they
  predict: tokens per iteration / (gamma * c(B) + r(B)). It is not in the paper; it is the proof of Theorem 3.8 with
  the measured step times. It ignores the encoders, the padding of the inputs and the sequences that are already
  finished but still in the batch.
"""

import logging
import statistics
import time
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .batch_scaling import Timing, generalised_speedup, time_decoder_steps
from .batched_decoding import BatchedDecodingStats, batched_autoregressive_decode, batched_speculative_decode
from .metrics import expected_speedup, expected_tokens_per_iteration, synchronize

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BatchedEvaluationResult:
    """
    The aggregated result of a benchmark at one batch size. The speedup is the baseline walltime over the speculative
    walltime, summed over the same batches.
    """

    batch_size: int
    num_examples: int  # The examples decoded, a multiple of the batch size.
    num_batches: int
    gamma: int
    temperature: float
    alpha: float  # The measured acceptance rate, per sequence.
    tokens_per_iteration: float  # The tokens every unfinished sequence gains per target model call, measured.
    expected_tokens_per_iteration: float  # Equation (1) at the measured alpha: what a batch size of 1 would yield.
    expected_lockstep_tokens_per_iteration: float  # Equation (1) at alpha^B, for the average number of unfinished rows.
    lockstep_efficiency: float  # Measured over Equation (1) at alpha: how much of the gain the lockstep keeps.
    generated_tokens: int  # The tokens of the speculative decoding outputs, up to each end-of-sequence token.
    speculative_seconds: float
    baseline_seconds: float | None
    speedup: float | None
    speculative_tokens_per_second: float
    baseline_tokens_per_second: float | None
    identical_outputs: int | None  # How many outputs equal the baseline's. Meaningful at temperature 0.
    cost_coefficient: float | None  # c(B), with `predict`.
    verification_ratio: float | None  # r(B), with `predict`.
    predicted_speedup_theorem_3_8: float | None  # Theorem 3.8 at c(B), with the batch size 1 Equation (1).
    predicted_speedup: float | None  # With r(B), and Equation (1) at alpha. An upper bound for the lockstep batch.
    predicted_speedup_lockstep: float | None  # With r(B) and the measured tokens per iteration.


def _timed(device: torch.device, function, *args, **kwargs):
    synchronize(device)
    begin: float = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize(device)
    return result, time.perf_counter() - begin


def make_batches(
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    batch_size: int,
    max_input_length: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenize the inputs into batches of `batch_size`, in order, each padded to its longest input.

    Args:
        tokenizer: The tokenizer, which must pad on the right (the T5 tokenizers do).
        inputs: The input texts. A last incomplete batch is dropped, so that every batch has the same size.
        batch_size: The number of inputs per batch.
        max_input_length: The inputs are truncated to this many tokens.
        device: The device to put the tensors on.

    Returns:
        The input ids and the attention mask of every batch.

    Raises:
        ValueError: If there are fewer inputs than one batch.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if len(inputs) < batch_size:
        raise ValueError(f"{len(inputs)} inputs don't fill a batch of {batch_size}")

    usable: int = len(inputs) // batch_size * batch_size
    if usable < len(inputs):
        logger.info("Batch size %d: dropping the last %d of %d inputs", batch_size, len(inputs) - usable, len(inputs))

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for begin in range(0, usable, batch_size):
        encoded = tokenizer(
            inputs[begin : begin + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        batches.append((encoded.input_ids.to(device), encoded.attention_mask.to(device)))
    return batches


def evaluate_batched(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    batch_size: int,
    temperature: float,
    gamma: int,
    max_new_tokens: int,
    max_input_length: int,
    run_baseline: bool = True,
    predict: bool = False,
) -> BatchedEvaluationResult:
    """
    Run batched speculative decoding, and the batched baseline unless disabled, on every batch of the inputs.

    The first batch is decoded once beforehand as a warmup that isn't counted. Both decoders run back to back on the
    same batch, so that the ratio isn't skewed by the machine state.

    Args:
        target: The target model, an encoder-decoder model in eval mode.
        draft: The draft model, an encoder-decoder model in eval mode, on the same device.
        tokenizer: The tokenizer of the models.
        inputs: The input texts. Only a multiple of `batch_size` of them is decoded.
        batch_size: The number of sequences decoded together.
        temperature: The sampling temperature for both models.
        gamma: The number of guesses per iteration.
        max_new_tokens: The generation limit.
        max_input_length: The inputs are truncated to this many tokens.
        run_baseline: Whether to run the baseline, which the speedup needs.
        predict: Whether to also measure c(B) and r(B) and the speedup they predict. It uses the average padded input
            length and a decoder prefix of half of `max_new_tokens`.

    Returns:
        The aggregated result.
    """

    device: torch.device = next(target.parameters()).device
    batches: list[tuple[torch.Tensor, torch.Tensor]] = make_batches(
        tokenizer, inputs, batch_size, max_input_length, device
    )

    # Warm up on the first batch.
    first_ids, first_mask = batches[0]
    batched_speculative_decode(target, draft, first_ids, first_mask, temperature, gamma, max_new_tokens)
    if run_baseline:
        batched_autoregressive_decode(target, first_ids, first_mask, temperature, max_new_tokens)

    stats: BatchedDecodingStats = BatchedDecodingStats()
    generated_tokens: int = 0
    baseline_tokens: int = 0
    speculative_seconds: float = 0.0
    baseline_seconds: float = 0.0
    identical: int = 0

    for input_ids, attention_mask in batches:
        (speculative_outputs, batch_stats), seconds = _timed(
            device,
            batched_speculative_decode,
            target,
            draft,
            input_ids,
            attention_mask,
            temperature,
            gamma,
            max_new_tokens,
        )
        speculative_seconds += seconds
        generated_tokens += sum(len(tokens) for tokens in speculative_outputs)
        stats += batch_stats

        if run_baseline:
            baseline_outputs, seconds = _timed(
                device, batched_autoregressive_decode, target, input_ids, attention_mask, temperature, max_new_tokens
            )
            baseline_seconds += seconds
            baseline_tokens += sum(len(tokens) for tokens in baseline_outputs)
            identical += sum(a == b for a, b in zip(speculative_outputs, baseline_outputs))

    alpha: float = stats.acceptance_rate
    average_rows: float = stats.row_iterations / stats.iterations
    expected: float = expected_tokens_per_iteration(alpha, gamma)

    cost_coefficient: float | None = None
    verification_ratio: float | None = None
    if predict:
        input_length: int = max(1, round(statistics.mean(ids.shape[1] for ids, _ in batches)))
        prefix_length: int = max(1, max_new_tokens // 2)
        target_timings: dict[int, Timing] = time_decoder_steps(
            target, batch_size, [1, gamma + 1], input_length, prefix_length
        )
        draft_timings: dict[int, Timing] = time_decoder_steps(draft, batch_size, [1], input_length, prefix_length)
        cost_coefficient = draft_timings[1].median / target_timings[1].median
        verification_ratio = target_timings[gamma + 1].median / target_timings[1].median

    examples: int = len(batches) * batch_size
    return BatchedEvaluationResult(
        batch_size=batch_size,
        num_examples=examples,
        num_batches=len(batches),
        gamma=gamma,
        temperature=temperature,
        alpha=alpha,
        tokens_per_iteration=stats.tokens_per_iteration,
        expected_tokens_per_iteration=expected,
        expected_lockstep_tokens_per_iteration=expected_tokens_per_iteration(alpha**average_rows, gamma),
        lockstep_efficiency=stats.tokens_per_iteration / expected,
        generated_tokens=generated_tokens,
        speculative_seconds=speculative_seconds,
        baseline_seconds=baseline_seconds if run_baseline else None,
        speedup=baseline_seconds / speculative_seconds if run_baseline else None,
        speculative_tokens_per_second=generated_tokens / speculative_seconds,
        baseline_tokens_per_second=baseline_tokens / baseline_seconds if run_baseline else None,
        identical_outputs=identical if run_baseline else None,
        cost_coefficient=cost_coefficient,
        verification_ratio=verification_ratio,
        predicted_speedup_theorem_3_8=(
            expected_speedup(alpha, gamma, cost_coefficient) if cost_coefficient is not None else None
        ),
        predicted_speedup=(
            generalised_speedup(alpha, gamma, cost_coefficient, verification_ratio)
            if cost_coefficient is not None
            else None
        ),
        predicted_speedup_lockstep=(
            stats.tokens_per_iteration / (gamma * cost_coefficient + verification_ratio)
            if cost_coefficient is not None
            else None
        ),
    )
