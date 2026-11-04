"""
This module contains the benchmark that compares batched speculative decoding to the batched autoregressive baseline ata given batch size.
"""

import logging
import statistics
import time
from dataclasses import dataclass, field

import torch
from torch._dynamo.utils import counters
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .batch_scaling import (
    Timing,
    generalised_speedup,
    time_decoder_steps,
    time_runner_loop,
    time_runner_steps,
)
from .batched_decoding import (
    BatchedDecodingStats,
    batched_autoregressive_decode,
    batched_speculative_decode,
    eos_token_ids,
)
from .batched_runner import BatchedModelRunner, reset_compilation
from .metrics import expected_speedup, expected_tokens_per_iteration, synchronize

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BatchedEvaluationResult:

    batch_size: int
    num_examples: int
    num_batches: int
    gamma: int
    temperature: float
    static_cache: bool
    compile_mode: str | None
    fuse_draft: bool
    alpha: float
    alpha_analytic: float
    example_counts: list[list[int]]
    tokens_per_iteration: float
    generated_tokens_per_iteration: float
    expected_tokens_per_iteration: float
    expected_lockstep_tokens_per_iteration: float
    lockstep_efficiency: float
    generated_tokens: int
    truncated_generations: int
    cudagraph_skips: (
        int
    )
    speculative_seconds: float
    baseline_seconds: float | None
    speedup: float | None
    speculative_tokens_per_second: float
    baseline_tokens_per_second: float | None
    identical_outputs: int | None
    cost_coefficient: float | None
    verification_ratio: float | None
    draft_loop_ratio: (
        float | None
    )
    predicted_speedup_theorem_3_8: float | None
    predicted_speedup: float | None
    predicted_speedup_lockstep: float | None
    speculative_seconds_runs: list[float] = field(default_factory=list)
    baseline_seconds_runs: list[float] = field(default_factory=list)
    speedup_runs: list[float] = field(default_factory=list)


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
    static_cache: bool = False,
    compile_mode: str | None = None,
    encoder_bucket: int = 64,
    compile_backend: str = "inductor",
    fuse_draft: bool = False,
    repeats: int = 1,
) -> BatchedEvaluationResult:

    if fuse_draft and compile_mode is None:
        raise ValueError("Fusing the drafting loop needs the runners, so a compile mode")

    device: torch.device = next(target.parameters()).device
    batches: list[tuple[torch.Tensor, torch.Tensor]] = make_batches(
        tokenizer, inputs, batch_size, max_input_length, device
    )

    vocab_size: int = min(len(tokenizer), target.config.vocab_size, draft.config.vocab_size)
    skips_before: int = counters["inductor"]["cudagraph_skips"]

    if compile_mode is None:
        return _benchmark(
            target,
            draft,
            batches,
            batch_size,
            temperature,
            gamma,
            max_new_tokens,
            run_baseline,
            predict,
            static_cache,
            compile_mode,
            fuse_draft,
            vocab_size,
            skips_before,
            repeats,
        )

    runner_mode: str | None = None if compile_mode == "eager" else compile_mode
    capacity: int = 1 + max_new_tokens + gamma + 1
    target_runner: BatchedModelRunner = BatchedModelRunner(
        target, batch_size, capacity, runner_mode, compile_backend, encoder_bucket
    )
    draft_runner: BatchedModelRunner = BatchedModelRunner(
        draft, batch_size, capacity, runner_mode, compile_backend, encoder_bucket
    )
    try:
        examples: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
            target_runner.bucket_for(ids.shape[1]): (ids, mask) for ids, mask in batches
        }
        for bucket, (ids, mask) in sorted(examples.items()):
            logger.info("Batch size %d: warming up the bucket of %d input tokens", batch_size, bucket)
            draft_runner.warmup(
                ids, mask, [1, 2]
            )
            if fuse_draft:
                draft_runner.warmup_loop(ids, mask, [1, 2], gamma, temperature, vocab_size)
            target_runner.warmup(ids, mask, [1, gamma + 1])

        return _benchmark(
            target_runner,
            draft_runner,
            batches,
            batch_size,
            temperature,
            gamma,
            max_new_tokens,
            run_baseline,
            predict,
            static_cache,
            compile_mode,
            fuse_draft,
            vocab_size,
            skips_before,
            repeats,
        )
    finally:
        del target_runner, draft_runner
        reset_compilation()


def _benchmark(
    target: PreTrainedModel | BatchedModelRunner,
    draft: PreTrainedModel | BatchedModelRunner,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    temperature: float,
    gamma: int,
    max_new_tokens: int,
    run_baseline: bool,
    predict: bool,
    static_cache: bool,
    compile_mode: str | None,
    fuse_draft: bool,
    vocab_size: int,
    skips_before: int,
    repeats: int = 1,
) -> BatchedEvaluationResult:

    model: PreTrainedModel = target.model if isinstance(target, BatchedModelRunner) else target
    device: torch.device = next(model.parameters()).device
    eos: set[int] = eos_token_ids(target)

    first_ids, first_mask = batches[0]
    batched_speculative_decode(
        target, draft, first_ids, first_mask, temperature, gamma, max_new_tokens, static_cache, vocab_size, fuse_draft
    )
    if run_baseline:
        batched_autoregressive_decode(
            target, first_ids, first_mask, temperature, max_new_tokens, static_cache, vocab_size
        )

    stats: BatchedDecodingStats = BatchedDecodingStats()
    generated_tokens: int = 0
    baseline_tokens: int = 0
    truncated: int = 0
    speculative_seconds: float = 0.0
    baseline_seconds: float = 0.0
    identical: int = 0

    speculative_runs: list[float] = []
    baseline_runs: list[float] = []
    for repeat in range(repeats):
        first_pass: bool = repeat == 0
        speculative_seconds = 0.0
        baseline_seconds = 0.0
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
                static_cache,
                vocab_size,
                fuse_draft,
            )
            speculative_seconds += seconds
            if first_pass:
                generated_tokens += sum(len(tokens) for tokens in speculative_outputs)
                truncated += sum(
                    len(tokens) >= max_new_tokens and tokens[-1] not in eos for tokens in speculative_outputs
                )
                stats += batch_stats

            if run_baseline:
                baseline_outputs, seconds = _timed(
                    device,
                    batched_autoregressive_decode,
                    target,
                    input_ids,
                    attention_mask,
                    temperature,
                    max_new_tokens,
                    static_cache,
                    vocab_size,
                )
                baseline_seconds += seconds
                if first_pass:
                    baseline_tokens += sum(len(tokens) for tokens in baseline_outputs)
                    identical += sum(a == b for a, b in zip(speculative_outputs, baseline_outputs))
        speculative_runs.append(speculative_seconds)
        if run_baseline:
            baseline_runs.append(baseline_seconds)
        logger.info(
            "Batch size %d, repeat %d/%d: speculative %.2fs, baseline %s",
            batch_size,
            repeat + 1,
            repeats,
            speculative_seconds,
            f"{baseline_seconds:.2f}s" if run_baseline else "n/a",
        )

    speculative_seconds = statistics.mean(speculative_runs)
    baseline_seconds = statistics.mean(baseline_runs) if run_baseline else 0.0

    cudagraph_skips: int = counters["inductor"]["cudagraph_skips"] - skips_before
    if cudagraph_skips:
        logger.warning("Inductor skipped CUDA graphs %d times", cudagraph_skips)

    alpha: float = stats.acceptance_rate
    average_rows: float = stats.row_iterations / stats.iterations
    expected: float = expected_tokens_per_iteration(alpha, gamma)

    cost_coefficient: float | None = None
    verification_ratio: float | None = None
    draft_loop_ratio: float | None = None
    if predict:
        prefix_length: int = max(1, max_new_tokens // 2)
        mean_length: float = statistics.mean(ids.shape[1] for ids, _ in batches)
        if isinstance(target, BatchedModelRunner):
            buckets: set[int] = {target.bucket_for(ids.shape[1]) for ids, _ in batches}
            input_length: int = min(buckets, key=lambda bucket: abs(bucket - mean_length))
            target_timings: dict[int, Timing] = time_runner_steps(target, [1, gamma + 1], input_length, prefix_length)
            draft_timings: dict[int, Timing] = time_runner_steps(draft, [1], input_length, prefix_length)
            if fuse_draft:
                loop_timing: Timing = time_runner_loop(
                    draft, gamma, temperature, vocab_size, input_length, prefix_length
                )
                draft_loop_ratio = loop_timing.median / target_timings[1].median
        else:
            input_length = max(1, round(mean_length))
            target_timings = time_decoder_steps(target, batch_size, [1, gamma + 1], input_length, prefix_length)
            draft_timings = time_decoder_steps(draft, batch_size, [1], input_length, prefix_length)
        cost_coefficient = draft_timings[1].median / target_timings[1].median
        verification_ratio = target_timings[gamma + 1].median / target_timings[1].median

    drafting_cost: float | None = None
    if cost_coefficient is not None:
        drafting_cost = draft_loop_ratio if draft_loop_ratio is not None else gamma * cost_coefficient

    examples: int = len(batches) * batch_size
    return BatchedEvaluationResult(
        batch_size=batch_size,
        num_examples=examples,
        num_batches=len(batches),
        gamma=gamma,
        temperature=temperature,
        static_cache=static_cache,
        compile_mode=compile_mode,
        fuse_draft=fuse_draft,
        alpha=alpha,
        alpha_analytic=stats.analytic_alpha,
        example_counts=stats.example_counts,
        tokens_per_iteration=stats.tokens_per_iteration,
        generated_tokens_per_iteration=generated_tokens / stats.row_iterations,
        expected_tokens_per_iteration=expected,
        expected_lockstep_tokens_per_iteration=expected_tokens_per_iteration(alpha**average_rows, gamma),
        lockstep_efficiency=stats.tokens_per_iteration / expected,
        generated_tokens=generated_tokens,
        truncated_generations=truncated,
        cudagraph_skips=cudagraph_skips,
        speculative_seconds=speculative_seconds,
        baseline_seconds=baseline_seconds if run_baseline else None,
        speedup=baseline_seconds / speculative_seconds if run_baseline else None,
        speculative_tokens_per_second=generated_tokens / speculative_seconds,
        baseline_tokens_per_second=baseline_tokens / baseline_seconds if run_baseline else None,
        identical_outputs=identical if run_baseline else None,
        cost_coefficient=cost_coefficient,
        verification_ratio=verification_ratio,
        draft_loop_ratio=draft_loop_ratio,
        predicted_speedup_theorem_3_8=(
            expected_speedup(alpha, gamma, cost_coefficient) if cost_coefficient is not None else None
        ),
        predicted_speedup=(
            generalised_speedup(alpha, gamma, drafting_cost / gamma, verification_ratio)
            if drafting_cost is not None
            else None
        ),
        predicted_speedup_lockstep=(
            stats.tokens_per_iteration / (drafting_cost + verification_ratio) if drafting_cost is not None else None
        ),
        speculative_seconds_runs=speculative_runs,
        baseline_seconds_runs=baseline_runs,
        speedup_runs=[b / s for b, s in zip(baseline_runs, speculative_runs)],
    )
