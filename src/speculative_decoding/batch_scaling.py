import sys

if __name__ == "__main__":
    sys.path.pop(0)

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
from transformers import AutoModelForSeq2SeqLM, DynamicCache, EncoderDecoderCache, PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from speculative_decoding.batched_runner import BatchedModelRunner, reset_compilation
from speculative_decoding.metrics import expected_speedup, expected_tokens_per_iteration, synchronize

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64]
MAX_TOKEN_ID: int = (
    32000
)


@dataclass
class Timing:

    median: float
    minimum: float
    maximum: float


@dataclass
class BatchPoint:

    batch_size: int
    target_step: Timing
    target_verify: Timing
    draft_step: Timing
    cost_coefficient: float
    verification_ratio: float
    speedup_theorem_3_8: float
    speedup: float


def generalised_speedup(alpha: float, gamma: int, c: float, r: float) -> float:

    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if gamma < 1:
        raise ValueError(f"gamma must be at least 1, got {gamma}")
    if c < 0 or r <= 0:
        raise ValueError(f"c must be non-negative and r positive, got c={c}, r={r}")

    return expected_tokens_per_iteration(alpha, gamma) / (gamma * c + r)


def _time_calls(device: torch.device, call, repeats: int, trials: int, warmup: int) -> Timing:

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

    model(encoder_outputs=encoder_outputs, decoder_input_ids=tokens, past_key_values=cache, use_cache=True)
    cache.crop(-tokens.shape[1])


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
    cache: EncoderDecoderCache = EncoderDecoderCache(DynamicCache(config=config), DynamicCache(config=config))
    model(encoder_outputs=encoder_outputs, decoder_input_ids=prefix, past_key_values=cache, use_cache=True)
    if cache.get_seq_length() != prefix_length:
        raise RuntimeError(f"The cache holds {cache.get_seq_length()} positions after the prefix, not {prefix_length}")

    timings: dict[int, Timing] = {}
    for num_tokens in sorted(set(step_lengths)):
        tokens: torch.Tensor = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=device)

        model(encoder_outputs=encoder_outputs, decoder_input_ids=tokens, past_key_values=cache, use_cache=True)
        if cache.get_seq_length() != prefix_length + num_tokens:
            raise RuntimeError(f"A step of {num_tokens} tokens left {cache.get_seq_length()} positions in the cache")
        cache.crop(-num_tokens)
        if cache.get_seq_length() != prefix_length:
            raise RuntimeError(f"The roll back left {cache.get_seq_length()} positions, not {prefix_length}")

        step = partial(_decoder_step, model, encoder_outputs, cache, tokens)
        timings[num_tokens] = _time_calls(device, step, repeats, trials, warmup)

    return timings


def synthetic_inputs(config: PreTrainedConfig, batch_size: int, input_length: int, device: torch.device):

    ids: torch.Tensor = torch.randint(
        3, min(config.vocab_size, MAX_TOKEN_ID), (batch_size, input_length), device=device
    )
    return ids, torch.ones_like(ids)


def _prefilled_slot(runner: BatchedModelRunner, input_length: int, prefix_length: int):

    config: PreTrainedConfig = runner.config
    device: torch.device = runner.model.device
    ids, mask = synthetic_inputs(config, runner.batch_size, input_length, device)
    slot = runner.prepare(ids, mask)
    prefix: torch.Tensor = torch.randint(
        3, min(config.vocab_size, MAX_TOKEN_ID), (runner.batch_size, prefix_length), device=device
    )
    prefix[:, 0] = config.decoder_start_token_id
    for position in range(prefix_length):
        runner.forward(slot, prefix[:, position : position + 1], position)
    return slot


@torch.inference_mode()
def time_runner_steps(
    runner: BatchedModelRunner,
    step_lengths: Iterable[int],
    input_length: int,
    prefix_length: int,
    repeats: int = 20,
    trials: int = 5,
    warmup: int = 5,
) -> dict[int, Timing]:

    lengths: list[int] = sorted(set(step_lengths))
    if min(input_length, prefix_length, repeats, trials) < 1 or warmup < 0 or not lengths or min(lengths) < 1:
        raise ValueError("The lengths, repeats and trials must be positive")
    if prefix_length + lengths[-1] > runner.capacity:
        raise ValueError(f"A prefix of {prefix_length} and a step of {lengths[-1]} don't fit in {runner.capacity}")

    device: torch.device = runner.model.device
    slot = _prefilled_slot(runner, input_length, prefix_length)
    timings: dict[int, Timing] = {}
    for num_tokens in lengths:
        tokens: torch.Tensor = torch.zeros((runner.batch_size, num_tokens), dtype=torch.long, device=device)
        timings[num_tokens] = _time_calls(
            device, partial(runner.forward, slot, tokens, prefix_length), repeats, trials, warmup
        )
    return timings


@torch.inference_mode()
def time_runner_loop(
    runner: BatchedModelRunner,
    num_guesses: int,
    temperature: float,
    vocab_size: int,
    input_length: int,
    prefix_length: int,
    repeats: int = 20,
    trials: int = 5,
    warmup: int = 5,
) -> Timing:

    if min(num_guesses, input_length, prefix_length, repeats, trials) < 1 or warmup < 0:
        raise ValueError("The lengths, repeats and trials must be positive")
    if prefix_length + num_guesses > runner.capacity:
        raise ValueError(f"A prefix of {prefix_length} and {num_guesses} guesses don't fit in {runner.capacity}")

    device: torch.device = runner.model.device
    slot = _prefilled_slot(runner, input_length, prefix_length)
    tokens: torch.Tensor = torch.zeros((runner.batch_size, 1), dtype=torch.long, device=device)
    call = partial(runner.draft_loop, slot, tokens, prefix_length, num_guesses, temperature, vocab_size)
    return _time_calls(device, call, repeats, trials, warmup)


def _time_batch_size(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    batch_size: int,
    gamma: int,
    input_length: int,
    prefix_length: int,
    repeats: int,
    trials: int,
    warmup: int,
    compile_mode: str | None,
    encoder_bucket: int,
    compile_backend: str,
) -> tuple[dict[int, Timing], dict[int, Timing]]:

    if compile_mode is None:
        return (
            time_decoder_steps(
                target, batch_size, [1, gamma + 1], input_length, prefix_length, repeats, trials, warmup
            ),
            time_decoder_steps(draft, batch_size, [1], input_length, prefix_length, repeats, trials, warmup),
        )

    runner_mode: str | None = None if compile_mode == "eager" else compile_mode
    capacity: int = prefix_length + 2 * (gamma + 1)
    target_runner = BatchedModelRunner(target, batch_size, capacity, runner_mode, compile_backend, encoder_bucket)
    draft_runner = BatchedModelRunner(draft, batch_size, capacity, runner_mode, compile_backend, encoder_bucket)
    try:
        device: torch.device = next(target.parameters()).device
        ids, mask = synthetic_inputs(target.config, batch_size, input_length, device)
        target_runner.warmup(ids, mask, [1, gamma + 1])
        draft_runner.warmup(ids, mask, [1])
        return (
            time_runner_steps(target_runner, [1, gamma + 1], input_length, prefix_length, repeats, trials, warmup),
            time_runner_steps(draft_runner, [1], input_length, prefix_length, repeats, trials, warmup),
        )
    finally:
        del target_runner, draft_runner
        reset_compilation()


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
    compile_mode: str | None = None,
    encoder_bucket: int = 64,
    compile_backend: str = "inductor",
) -> list[BatchPoint]:

    points: list[BatchPoint] = []
    for batch_size in sorted(set(batch_sizes)):
        out_of_memory: bool = False
        try:
            target_timings, draft_timings = _time_batch_size(
                target,
                draft,
                batch_size,
                gamma,
                input_length,
                prefix_length,
                repeats,
                trials,
                warmup,
                compile_mode,
                encoder_bucket,
                compile_backend,
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
    argparser.add_argument(
        "--compile",
        choices=["eager", "default", "reduce-overhead"],
        default=None,
        help="Time the steps through runners with static caches and padded inputs, compiled with torch.compile in this "
        "mode, like evaluate.py --compile. 'reduce-overhead' uses CUDA graphs. 'eager' runs the runners uncompiled.",
    )
    argparser.add_argument(
        "--encoder-bucket",
        type=int,
        default=64,
        help="With --compile, inputs are padded to a multiple of this length. Default: 64",
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
    if args.encoder_bucket < 1:
        argparser.error("--encoder-bucket must be positive")
    return args


def pick_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    args: Namespace = parse_arguments()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

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
        compile_mode=args.compile,
        encoder_bucket=args.encoder_bucket,
    )

    report: dict = {
        "target": args.target,
        "draft": args.draft,
        "device": device.type,
        "dtype": args.dtype,
        "compile_mode": args.compile,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "expected_tokens_per_iteration": expected_tokens_per_iteration(args.alpha, args.gamma),
        "input_length": args.input_length,
        "prefix_length": args.prefix_length,
        "repeats": args.repeats,
        "trials": args.trials,
        "results_cost_coefficient": args.reported_cost_coefficient,
        "points": [asdict(point) for point in points],
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
