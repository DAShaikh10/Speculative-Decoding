"""
This script evaluates batched speculative decoding against the batched autoregressive baseline, for several batch sizes.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from speculative_decoding import Task
from speculative_decoding.batch_scaling import pick_device
from speculative_decoding.batched_evaluation import BatchedEvaluationResult, evaluate_batched
from speculative_decoding.batched_runner import reset_compilation
from speculative_decoding.tasks import load_inputs

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32]


def parse_arguments() -> Namespace:

    argparser: ArgumentParser = ArgumentParser("Evaluate batched Speculative Decoding against the batched baseline")

    argparser.add_argument("--draft", "-d", required=True, type=str, help="The draft smaller model.")
    argparser.add_argument(
        "--gamma", "-g", required=True, type=int, help="The number of draft tokens to sample for each target token."
    )
    argparser.add_argument("--target", "-t", required=True, type=str, help="The target larger model.")
    argparser.add_argument(
        "--task",
        "-k",
        required=True,
        type=Task,
        choices=list(Task),
        help="The task to evaluate. Must be a valid 'Task' enum value.",
    )
    argparser.add_argument(
        "--temperature",
        required=True,
        type=float,
        help="The temperature for sampling from both the draft and target models.",
    )
    argparser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help=f"The batch sizes to evaluate. Default: {' '.join(map(str, DEFAULT_BATCH_SIZES))}",
    )
    argparser.add_argument(
        "--num-examples",
        default=64,
        type=int,
        help="The number of test inputs. A batch size decodes the largest multiple of it that fits, so use a multiple "
        "of every batch size to compare them on the same inputs. Default: 64",
    )
    argparser.add_argument("--max-new-tokens", type=int, default=128, help="The generation limit. Default: 128")
    argparser.add_argument("--max-input-length", type=int, default=512, help="The input truncation. Default: 512")
    argparser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="The model precision. T5 v1.1 overflows in float16. Default: float32",
    )
    argparser.add_argument("--seed", type=int, default=0, help="The random seed. Default: 0")
    argparser.add_argument(
        "--static-cache",
        action="store_true",
        help="Preallocate the KV caches for the whole generation instead of growing them. A runner (--compile) always "
        "does.",
    )
    argparser.add_argument(
        "--compile",
        choices=["eager", "default", "reduce-overhead"],
        default=None,
        help="Run the models through runners with static caches and padded inputs, and compile the decoding steps "
        "with torch.compile in this mode. 'reduce-overhead' uses CUDA graphs. 'eager' runs the runners uncompiled.",
    )
    argparser.add_argument(
        "--encoder-bucket",
        type=int,
        default=64,
        help="With --compile, inputs are padded to a multiple of this length, and each length is compiled "
        "separately. Default: 64",
    )
    argparser.add_argument(
        "--fuse-draft",
        action="store_true",
        help="Sample all the guesses of the draft model in one call of its runner, which is one CUDA graph launch with "
        "--compile reduce-overhead. Needs --compile.",
    )
    argparser.add_argument(
        "--skip-baseline", action="store_true", help="Skip the autoregressive baseline, for quick debugging runs."
    )
    argparser.add_argument(
        "--predict",
        action="store_true",
        help="Also measure c(B) and r(B) at every batch size (see batch_scaling.py) and the speedup they predict.",
    )
    argparser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Time the pass over all batches this many times, to get the run-to-run spread of the speedup. Only the "
        "timed decoding repeats; loading, compiling and warming up happen once. Default: 1",
    )
    argparser.add_argument("--output", type=Path, default=None, help="Where to write the results as JSON.")

    args: Namespace = argparser.parse_args()
    if args.gamma < 1:
        argparser.error(f"--gamma must be at least 1, got {args.gamma}")
    if args.temperature < 0:
        argparser.error(f"--temperature must be non-negative, got {args.temperature}")
    if min(args.batch_sizes) < 1:
        argparser.error("--batch-sizes must be positive")
    if args.num_examples < min(args.batch_sizes):
        argparser.error("--num-examples must be at least the smallest batch size")
    if min(args.max_new_tokens, args.max_input_length) < 1:
        argparser.error("--max-new-tokens and --max-input-length must be positive")
    if args.fuse_draft and not args.compile:
        argparser.error("--fuse-draft needs --compile")
    if args.repeats < 1:
        argparser.error("--repeats must be at least 1")
    if args.encoder_bucket < 1:
        argparser.error("--encoder-bucket must be positive")
    if args.compile and args.max_new_tokens < args.gamma + 1:
        argparser.error("--compile needs --max-new-tokens of at least --gamma + 1")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    args: Namespace = parse_arguments()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device: torch.device = pick_device()
    dtype: torch.dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if AutoTokenizer.from_pretrained(args.draft).get_vocab() != tokenizer.get_vocab():
        raise SystemExit("The draft and target models must share a vocabulary.")
    target = AutoModelForSeq2SeqLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    draft = AutoModelForSeq2SeqLM.from_pretrained(args.draft, dtype=dtype).to(device).eval()

    inputs: list[str] = load_inputs(args.task, args.num_examples)

    results: list[BatchedEvaluationResult] = []
    for batch_size in sorted(set(args.batch_sizes)):
        if batch_size > len(inputs):
            logger.warning("Batch size %d is larger than the %d inputs, skipping it", batch_size, len(inputs))
            continue

        out_of_memory: bool = False
        try:
            result: BatchedEvaluationResult = evaluate_batched(
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                inputs=inputs,
                batch_size=batch_size,
                temperature=args.temperature,
                gamma=args.gamma,
                max_new_tokens=args.max_new_tokens,
                max_input_length=args.max_input_length,
                run_baseline=not args.skip_baseline,
                predict=args.predict,
                static_cache=args.static_cache,
                compile_mode=args.compile,
                encoder_bucket=args.encoder_bucket,
                fuse_draft=args.fuse_draft,
                repeats=args.repeats,
            )
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True

        if out_of_memory:
            logger.warning("Batch size %d ran out of memory, skipping it and the larger ones", batch_size)
            reset_compilation()
            break

        results.append(result)
        logger.info(
            "B=%-4d alpha=%.2f tokens/iter=%.2f (B=1 would give %.2f) speedup=%s",
            batch_size,
            result.alpha,
            result.tokens_per_iteration,
            result.expected_tokens_per_iteration,
            f"{result.speedup:.2f}x" if result.speedup is not None else "n/a",
        )

    report: dict = {
        "task": args.task.value,
        "target": args.target,
        "draft": args.draft,
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
