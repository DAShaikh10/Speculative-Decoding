"""
This script evaluates batched speculative decoding against the batched autoregressive baseline, for several batch sizes.

The paper only evaluates a batch size of 1 (Section 4.1), so this is an extension. It is the batched counterpart of
`evaluate.py`, with the same tasks, models and flags, and doesn't touch it. See `batched_decoding.py` for the lockstep
algorithm and why its output distribution is exact.

Usage:
    python -m speculative_decoding.batched_evaluate -t google-t5/t5-3b -d google-t5/t5-small -g 5 \\
        -k translation --temperature 0 --batch-sizes 1 2 4 8 16 --num-examples 64 --predict --output batched.json
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
from speculative_decoding.tasks import load_inputs

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32]


def parse_arguments() -> Namespace:
    """
    Read the command-line arguments.

    Returns:
        Namespace: The parsed arguments.
    """

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
        "--skip-baseline", action="store_true", help="Skip the autoregressive baseline, for quick debugging runs."
    )
    argparser.add_argument(
        "--predict",
        action="store_true",
        help="Also measure c(B) and r(B) at every batch size (see batch_scaling.py) and the speedup they predict.",
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
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    args: Namespace = parse_arguments()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")  # Allow TF32 tensor cores for float32 matmuls, like evaluate.py.

    device: torch.device = pick_device()
    dtype: torch.dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForSeq2SeqLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    draft = AutoModelForSeq2SeqLM.from_pretrained(args.draft, dtype=dtype).to(device).eval()
    if target.config.vocab_size != draft.config.vocab_size:
        raise SystemExit("The draft and target models must share a vocabulary.")

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
            )
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True

        if out_of_memory:
            logger.warning("Batch size %d ran out of memory, skipping it and the larger ones", batch_size)
            torch.cuda.empty_cache()
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
        "dtype": args.dtype,
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
