"""
This script measures how the cost model of "Fast Inference of Transformers via Speculative Decoding"
(https://arxiv.org/abs/2211.17192) changes with the batch size. It doesn't decode anything and touches no existing code.

The paper only evaluates a batch size of 1. Its Section 3.3 assumes that "we can run gamma + 1 concurrent evaluations of
Mp in parallel without increasing the walltime", and its Discussion says the method is "not helpful for configurations
where additional computation resources are not available". A larger batch uses up that spare compute, so this script
measures how much of the assumption survives, per batch size B:

- c(B) = T_q(B, 1) / T_p(B, 1), the cost coefficient of Definition 3.7 at batch size B.
- r(B) = T_p(B, gamma + 1) / T_p(B, 1), how much more verifying gamma + 1 positions costs than one position. The paper
  assumes r = 1.

T_m(B, K) is the time of one decoder step of model m that feeds K new tokens to each of B sequences, on a KV cache that
already holds a prefix. These steps are exactly what `speculative_decode` runs: one target step of gamma + 1 tokens (the
last token of the previous iteration plus the gamma guesses) and gamma draft steps of 1 token. (After an iteration that
accepted every guess the first draft step has 2 tokens, which is ignored here.)

The proof of Theorem 3.8 with "verifying costs the same as one step" replaced by the measured r(B) gives

    speedup(B) = (1 - alpha^(gamma+1)) / ((1 - alpha) * (gamma * c(B) + r(B)))

With r = 1 it is exactly Theorem 3.8. This generalisation is not in the paper. Two assumptions are not in the paper
either: alpha is taken from a batch size 1 run, so it is assumed not to depend on the batch size, and every sequence
is assumed to produce the full expected number of tokens per iteration. That makes `speedup` an upper bound for any
real batched implementation, whose sequences have to wait for each other.

Usage:
    python batch_scaling.py --target google-t5/t5-3b --draft google-t5/t5-small --results results.json
    python batch_scaling.py -t google-t5/t5-3b -d google-t5/t5-small -g 5 --alpha 0.65 --batch-sizes 1 4 16 64

`results.json` is the output of `evaluate.py`, from which `alpha` and `gamma` are read. Explicit flags override it.
"""
import sys
# 1. Temporarily remove the script's directory from the import path
# so Python doesn't accidentally load your local "types.py"
_script_dir = sys.path.pop(0)

import json
import logging
import statistics
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, DynamicCache, EncoderDecoderCache, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from speculative_decoding.metrics import expected_speedup, expected_tokens_per_iteration, synchronize

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64]
MAX_TOKEN_ID: int = (
    32000  # The synthetic inputs only use ids below this, which are regular tokens of the T5 vocabulary.
)


@dataclass
class Timing:
    """
    The time of one step, in seconds. Each trial times `repeats` back-to-back steps and divides by `repeats`.
    """

    median: float  # The median over the trials, which is the number used in the results.
    minimum: float
    maximum: float


@dataclass
class BatchPoint:
    """
    The measurements and predictions for one batch size.
    """

    batch_size: int
    target_step: Timing  # T_p(B, 1)
    target_verify: Timing  # T_p(B, gamma + 1)
    draft_step: Timing  # T_q(B, 1)
    cost_coefficient: float  # c(B), Definition 3.7.
    verification_ratio: float  # r(B). The paper assumes 1.
    speedup_theorem_3_8: float  # Theorem 3.8 at c(B), so with r = 1: what the paper's formula predicts.
    speedup: float  # With the measured r(B). An upper bound for a real batched implementation.


def generalised_speedup(alpha: float, gamma: int, c: float, r: float) -> float:
    """
    Theorem 3.8 with the cost of the target step that verifies gamma + 1 positions being r times that of a single step.

    The expected cost of an iteration is gamma * c * T + r * T, and it produces Equation (1) tokens, while the baseline
    pays T per token. With r = 1 this is `metrics.expected_speedup`.

    Args:
        alpha: The acceptance rate, in [0, 1].
        gamma: The number of guesses per iteration, at least 1.
        c: The cost coefficient, T_q / T_p for a single token step.
        r: The cost of the verification step relative to a single token step of the target, T_p(gamma + 1) / T_p(1).

    Returns:
        The expected walltime improvement factor.

    Raises:
        ValueError: If an argument is out of range.
    """

    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if gamma < 1:
        raise ValueError(f"gamma must be at least 1, got {gamma}")
    if c < 0 or r <= 0:
        raise ValueError(f"c must be non-negative and r positive, got c={c}, r={r}")

    return expected_tokens_per_iteration(alpha, gamma) / (gamma * c + r)


def _time_calls(device: torch.device, call, repeats: int, trials: int, warmup: int) -> Timing:
    """
    Time a function that runs one step. Steps run back to back and the device is only synchronized around a trial, like
    `metrics.measure_cost_coefficient` does, so that the launch of a step can overlap with the previous one.
    """

    for _ in range(warmup):
        call()

    means: list[float] = []
    for _ in range(trials):
        synchronize(device)
        begin: float = time.perf_counter()
        for _ in range(repeats):
            call()
        synchronize(device)
        means.append((time.perf_counter() - begin) / repeats)

    return Timing(median=statistics.median(means), minimum=min(means), maximum=max(means))


def _decoder_step(
    model: PreTrainedModel, encoder_outputs: BaseModelOutput, cache: EncoderDecoderCache, tokens: torch.Tensor
) -> None:
    """
    One timed step: feed `tokens` to the decoder, then roll the KV cache back to where it was.
    """

    model(encoder_outputs=encoder_outputs, decoder_input_ids=tokens, past_key_values=cache, use_cache=True)
    cache.crop(-tokens.shape[1])  # Cache.crop takes a negative number of tokens to remove.


@torch.inference_mode()
def time_decoder_steps(
    model: PreTrainedModel,
    batch_size: int,
    step_lengths: Iterable[int],
    input_length: int,
    prefix_length: int,
    repeats: int = 20,
    trials: int = 5,
    warmup: int = 5,
) -> dict[int, Timing]:
    """
    Time decoder steps of an encoder-decoder model with a batch of sequences, on a KV cache that holds a prefix.

    The batch is `batch_size` synthetic inputs and decoder prefixes of random tokens. The cost of a step doesn't depend
    on the values of the tokens. Each timed step appends the new tokens to the KV cache and then rolls the cache back to
    the prefix, which is what a decoding iteration does after a rejection, so every step sees the same cache length.

    Args:
        model: An encoder-decoder model in eval mode, on its device.
        batch_size: The number of sequences decoded together.
        step_lengths: The numbers of new tokens per sequence to time a step for.
        input_length: The length of the encoder input of every sequence, so the length of the cross-attention.
        prefix_length: The number of decoder positions, including the start token, the KV cache holds before a step.
        repeats: The number of steps per trial.
        trials: The number of trials.
        warmup: The number of untimed steps before the first trial.

    Returns:
        The timing of a step, by number of new tokens.

    Raises:
        ValueError: If the model isn't an encoder-decoder model in eval mode or an argument is out of range.
        RuntimeError: If the KV cache doesn't grow and roll back by the expected number of positions.
    """

    config = model.config
    if not config.is_encoder_decoder:
        raise ValueError("Only encoder-decoder models are supported")
    if model.training:
        raise ValueError("The model must be in eval mode")
    if min(batch_size, input_length, prefix_length, repeats, trials) < 1 or warmup < 0:
        raise ValueError("The batch size, lengths, repeats and trials must be positive")

    device: torch.device = next(model.parameters()).device
    high: int = min(config.vocab_size, MAX_TOKEN_ID)
    encoder_ids: torch.Tensor = torch.randint(3, high, (batch_size, input_length), device=device)
    prefix: torch.Tensor = torch.randint(3, high, (batch_size, prefix_length), device=device)
    prefix[:, 0] = config.decoder_start_token_id

    encoder_outputs = model.get_encoder()(input_ids=encoder_ids)
    # The same cache as `DecoderState` uses without a static capacity.
    cache: EncoderDecoderCache = EncoderDecoderCache(DynamicCache(config=config), DynamicCache(config=config))
    model(encoder_outputs=encoder_outputs, decoder_input_ids=prefix, past_key_values=cache, use_cache=True)
    if cache.get_seq_length() != prefix_length:
        raise RuntimeError(f"The cache holds {cache.get_seq_length()} positions after the prefix, not {prefix_length}")

    timings: dict[int, Timing] = {}
    for num_tokens in sorted(set(step_lengths)):
        tokens: torch.Tensor = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=device)

        # Check once that a step grows the cache by the new tokens and that the roll back restores the prefix. If
        # either failed silently, the timings would be of another cache length.
        model(encoder_outputs=encoder_outputs, decoder_input_ids=tokens, past_key_values=cache, use_cache=True)
        if cache.get_seq_length() != prefix_length + num_tokens:
            raise RuntimeError(f"A step of {num_tokens} tokens left {cache.get_seq_length()} positions in the cache")
        cache.crop(-num_tokens)
        if cache.get_seq_length() != prefix_length:
            raise RuntimeError(f"The roll back left {cache.get_seq_length()} positions, not {prefix_length}")

        step = partial(_decoder_step, model, encoder_outputs, cache, tokens)
        timings[num_tokens] = _time_calls(device, step, repeats, trials, warmup)

    return timings


def measure_batch_scaling(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    batch_sizes: Iterable[int],
    gamma: int,
    alpha: float,
    input_length: int = 64,
    prefix_length: int = 64,
    repeats: int = 20,
    trials: int = 5,
    warmup: int = 5,
) -> list[BatchPoint]:
    """
    Measure c(B) and r(B) for every batch size, and predict the speedup from them.

    Batch sizes run in increasing order. At the first one that runs out of device memory the measurement stops and the
    larger ones are skipped.

    Args:
        target: The target model Mp, in eval mode.
        draft: The draft model Mq, in eval mode, on the same device.
        batch_sizes: The batch sizes to measure.
        gamma: The number of guesses per iteration.
        alpha: The acceptance rate, measured at a batch size of 1.
        input_length: The length of the encoder inputs.
        prefix_length: The length of the decoder prefix in the KV caches.
        repeats: The number of steps per trial.
        trials: The number of trials, whose median is used.
        warmup: The number of untimed steps before the first trial.

    Returns:
        One point per batch size that fit in memory, in increasing order of the batch size.
    """

    points: list[BatchPoint] = []
    for batch_size in sorted(set(batch_sizes)):
        out_of_memory: bool = False
        try:
            target_timings: dict[int, Timing] = time_decoder_steps(
                target, batch_size, [1, gamma + 1], input_length, prefix_length, repeats, trials, warmup
            )
            draft_timings: dict[int, Timing] = time_decoder_steps(
                draft, batch_size, [1], input_length, prefix_length, repeats, trials, warmup
            )
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True

        if out_of_memory:
            logger.warning("Batch size %d ran out of memory, skipping it and the larger ones", batch_size)
            torch.cuda.empty_cache()
            break

        target_step: Timing = target_timings[1]
        target_verify: Timing = target_timings[gamma + 1]
        draft_step: Timing = draft_timings[1]
        c: float = draft_step.median / target_step.median
        r: float = target_verify.median / target_step.median
        points.append(
            BatchPoint(
                batch_size=batch_size,
                target_step=target_step,
                target_verify=target_verify,
                draft_step=draft_step,
                cost_coefficient=c,
                verification_ratio=r,
                speedup_theorem_3_8=expected_speedup(alpha, gamma, c),
                speedup=generalised_speedup(alpha, gamma, c, r),
            )
        )
        logger.info(
            "B=%-4d c=%.3f r=%.3f  Theorem 3.8: %.2fx  with r: %.2fx",
            batch_size,
            c,
            r,
            points[-1].speedup_theorem_3_8,
            points[-1].speedup,
        )

    return points


def parse_arguments() -> Namespace:
    """
    Read the command-line arguments.

    Returns:
        Namespace: The parsed arguments, with `alpha` and `gamma` resolved from the flags and `--results`.
    """

    argparser: ArgumentParser = ArgumentParser("Measure how the speculative decoding cost model scales with batch size")

    argparser.add_argument("--draft", "-d", required=True, type=str, help="The draft smaller model.")
    argparser.add_argument("--target", "-t", required=True, type=str, help="The target larger model.")
    argparser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="The JSON output of evaluate.py, to read alpha and gamma from. The flags below override it.",
    )
    argparser.add_argument("--gamma", "-g", type=int, default=None, help="The number of draft tokens per iteration.")
    argparser.add_argument(
        "--alpha", type=float, default=None, help="The acceptance rate, measured at a batch size of 1."
    )
    argparser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help=f"The batch sizes to measure. Default: {' '.join(map(str, DEFAULT_BATCH_SIZES))}",
    )
    argparser.add_argument("--input-length", type=int, default=64, help="The length of the encoder inputs. Default: 64")
    argparser.add_argument(
        "--prefix-length",
        type=int,
        default=64,
        help="The number of decoder positions in the KV cache before a step, about the middle of a generation. "
        "Default: 64",
    )
    argparser.add_argument("--repeats", type=int, default=20, help="The number of steps per trial. Default: 20")
    argparser.add_argument(
        "--trials", type=int, default=5, help="The number of trials, of which the median is used. Default: 5"
    )
    argparser.add_argument("--warmup", type=int, default=5, help="The number of untimed steps. Default: 5")
    argparser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="The model precision. T5 v1.1 overflows in float16. Default: float32",
    )
    argparser.add_argument("--seed", type=int, default=0, help="The random seed. Default: 0")
    argparser.add_argument("--output", type=Path, default=None, help="Where to write the results as JSON.")

    args: Namespace = argparser.parse_args()

    args.reported_cost_coefficient = None
    if args.results is not None:
        reported: dict = json.loads(args.results.read_text())
        args.alpha = reported["alpha"] if args.alpha is None else args.alpha
        args.gamma = reported["gamma"] if args.gamma is None else args.gamma
        args.reported_cost_coefficient = reported.get("cost_coefficient")
    if args.alpha is None or args.gamma is None:
        argparser.error("give --alpha and --gamma, or --results to read them from")
    if not 0 <= args.alpha <= 1:
        argparser.error(f"--alpha must be in [0, 1], got {args.alpha}")
    if args.gamma < 1:
        argparser.error(f"--gamma must be at least 1, got {args.gamma}")
    if min(args.batch_sizes) < 1:
        argparser.error("--batch-sizes must be positive")
    if min(args.input_length, args.prefix_length, args.repeats, args.trials) < 1 or args.warmup < 0:
        argparser.error("--input-length, --prefix-length, --repeats and --trials must be positive")
    return args


def pick_device() -> torch.device:
    """
    Device selection logic, the same as in evaluate.py.

    Returns:
        torch.device: The device to use for model inference.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    args: Namespace = parse_arguments()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")  # The same TF32 setting as evaluate.py, so the timings are comparable.

    device: torch.device = pick_device()
    dtype: torch.dtype = getattr(torch, args.dtype)

    target = AutoModelForSeq2SeqLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    draft = AutoModelForSeq2SeqLM.from_pretrained(args.draft, dtype=dtype).to(device).eval()
    if target.config.vocab_size != draft.config.vocab_size:
        raise SystemExit("The draft and target models must share a vocabulary.")

    points: list[BatchPoint] = measure_batch_scaling(
        target=target,
        draft=draft,
        batch_sizes=args.batch_sizes,
        gamma=args.gamma,
        alpha=args.alpha,
        input_length=args.input_length,
        prefix_length=args.prefix_length,
        repeats=args.repeats,
        trials=args.trials,
        warmup=args.warmup,
    )

    report: dict = {
        "target": args.target,
        "draft": args.draft,
        "device": device.type,
        "dtype": args.dtype,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "expected_tokens_per_iteration": expected_tokens_per_iteration(args.alpha, args.gamma),
        "input_length": args.input_length,
        "prefix_length": args.prefix_length,
        "repeats": args.repeats,
        "trials": args.trials,
        "results_cost_coefficient": args.reported_cost_coefficient,  # evaluate.py's c, for comparing with c(1).
        "points": [asdict(point) for point in points],
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
