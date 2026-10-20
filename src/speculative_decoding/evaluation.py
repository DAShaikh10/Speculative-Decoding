"""
This module contains the benchmark that compares speculative decoding to the autoregressive baseline.
"""

import logging
import time
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .decoding import DecodingStats, autoregressive_decode, speculative_decode
from .metrics import expected_speedup, expected_tokens_per_iteration, measure_cost_coefficient, synchronize
from .runner import ModelRunner

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    The aggregated result of a benchmark run. The speedup is the baseline walltime over the speculative walltime.
    """

    num_examples: int
    gamma: int
    temperature: float
    alpha: float  # The measured acceptance rate.
    tokens_per_iteration: float  # The measured number of tokens produced per target model call.
    expected_tokens_per_iteration: float  # Equation (1) at the measured alpha.
    use_cache: bool
    static_cache: bool
    compile_mode: str | None  # None when not run through the runner, "eager" for the runner without compiling.
    cost_coefficient: float  # c, Definition 3.7.
    speculative_seconds: float
    baseline_seconds: float | None
    speedup: float | None
    expected_speedup: float  # Theorem 3.8 at the measured alpha and c.
    identical_outputs: int | None  # How many outputs equal the baseline's. Meaningful at temperature 0.


def _timed(device: torch.device, function, *args, **kwargs):
    synchronize(device)
    begin: float = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize(device)
    return result, time.perf_counter() - begin


def evaluate(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    temperature: float,
    gamma: int,
    max_new_tokens: int,
    max_input_length: int,
    run_baseline: bool = True,
    use_cache: bool = True,
    static_cache: bool = False,
    compile_mode: str | None = None,
    encoder_bucket: int = 64,
    compile_backend: str = "inductor",
) -> EvaluationResult:
    """
    Run speculative decoding, and the baseline unless disabled, on every input with a batch size of 1.

    The first input is decoded once beforehand as a warmup that isn't counted. With `compile_mode` both models are run
    through runners, which pad the inputs to buckets and compile the decoding steps (the baseline and the speculative
    decoding use the same compiled target steps). Every bucket and step length is compiled during the warmup, so that
    no compiling is timed. Only encoder-decoder models are supported then.

    Args:
        compile_mode: None to run the models as they are. "eager" to run them through the runners without compiling.
            Otherwise the `mode` of `torch.compile`, such as "default" or "reduce-overhead".
        encoder_bucket: With a runner, the inputs are padded to a multiple of this length.
        compile_backend: With a runner, the `torch.compile` backend. Only tests change it.
    """

    if compile_mode is not None and not use_cache:
        raise ValueError("The runners need the KV cache")

    device: torch.device = next(target.parameters()).device
    encoded: list[torch.Tensor] = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length).input_ids.to(device)
        for text in inputs
    ]

    if compile_mode is not None:
        # One position for the start token, the generated tokens, and the guesses that can run past the limit.
        capacity: int = 1 + max_new_tokens + gamma + 1
        runner_mode: str | None = None if compile_mode == "eager" else compile_mode
        target = ModelRunner(target, capacity, runner_mode, compile_backend, encoder_bucket)
        draft = ModelRunner(draft, capacity, runner_mode, compile_backend, encoder_bucket)

        # Compile and record the graphs of every bucket that occurs, before anything is timed.
        examples: dict[int, torch.Tensor] = {target.bucket_for(ids.shape[1]): ids for ids in encoded}
        for bucket, ids in sorted(examples.items()):
            logger.info("Warming up the bucket of %d input tokens", bucket)
            draft.warmup(ids, [1, 2])  # One new token after a rejection, two after every guess was accepted.
            target.warmup(ids, [1, gamma + 1])  # The baseline's steps and the verification steps.

    # Warm up, and estimate the cost coefficient on the first input.
    speculative_decode(target, draft, encoded[0], temperature, gamma, max_new_tokens, use_cache, static_cache)
    if run_baseline:
        autoregressive_decode(target, encoded[0], temperature, max_new_tokens, use_cache, static_cache)
    cost_coefficient: float = measure_cost_coefficient(target, draft, encoded[0], use_cache, static_cache=static_cache)

    stats: DecodingStats = DecodingStats()
    generated_tokens: int = 0
    speculative_seconds: float = 0.0
    baseline_seconds: float = 0.0
    identical: int = 0

    for input_ids in encoded:
        # Run both decoders back to back on the same input, so that the ratio is not skewed by the machine state.
        (speculative_tokens, example_stats), seconds = _timed(
            device,
            speculative_decode,
            target,
            draft,
            input_ids,
            temperature,
            gamma,
            max_new_tokens,
            use_cache,
            static_cache,
        )
        speculative_seconds += seconds
        generated_tokens += len(speculative_tokens)
        stats += example_stats

        if run_baseline:
            baseline_tokens, seconds = _timed(
                device, autoregressive_decode, target, input_ids, temperature, max_new_tokens, use_cache, static_cache
            )
            baseline_seconds += seconds
            identical += baseline_tokens == speculative_tokens

    alpha: float = stats.acceptance_rate
    return EvaluationResult(
        num_examples=len(inputs),
        gamma=gamma,
        temperature=temperature,
        alpha=alpha,
        tokens_per_iteration=generated_tokens / stats.iterations,
        expected_tokens_per_iteration=expected_tokens_per_iteration(alpha, gamma),
        use_cache=use_cache,
        static_cache=static_cache,
        compile_mode=compile_mode,
        cost_coefficient=cost_coefficient,
        speculative_seconds=speculative_seconds,
        baseline_seconds=baseline_seconds if run_baseline else None,
        speedup=baseline_seconds / speculative_seconds if run_baseline else None,
        expected_speedup=expected_speedup(alpha, gamma, cost_coefficient),
        identical_outputs=identical if run_baseline else None,
    )
