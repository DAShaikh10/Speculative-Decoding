"""
This script replicates the experiment in "Fast Inference of Transformers via Speculative Decoding"
(https://arxiv.org/abs/2211.17192)
"""

import json
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from speculative_decoding import Task
from speculative_decoding.evaluation import EvaluationResult, evaluate
from speculative_decoding.tasks import load_inputs


def parse_arguments() -> Namespace:
    """
    Read the command-line arguments.

    Returns:
        Namespace: The parsed arguments.
    """

    argparser: ArgumentParser = ArgumentParser("Evaluate model inference speed via Speculative Decoding")

    argparser.add_argument(
        "--draft",
        "-d",
        required=True,
        type=str,
        help="The draft smaller model.",
    )
    argparser.add_argument(
        "--gamma", "-g", required=True, type=int, help="The number of draft tokens to sample for each target token."
    )
    argparser.add_argument(
        "--target",
        "-t",
        required=True,
        type=str,
        help="The target larger model.",
    )
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
        "--num-examples", default=100, type=int, help="The number of test inputs to decode. Default: 100"
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
        "--no-cache", action="store_true", help="Disable the KV cache, so every call re-processes the whole sequence."
    )
    argparser.add_argument(
        "--static-cache",
        action="store_true",
        help="Preallocate the KV caches for the whole generation instead of growing them. Needs the KV cache.",
    )
    argparser.add_argument("--output", type=Path, default=None, help="Where to write the results as JSON.")

    args: Namespace = argparser.parse_args()
    if args.static_cache and args.no_cache:
        argparser.error("--static-cache needs the KV cache, so it can't be combined with --no-cache")
    return args


def pick_device() -> torch.device:
    """
    Device selection logic.

    Returns:
        torch.device: The device to use for model inference.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    args: Namespace = parse_arguments()
    torch.manual_seed(args.seed)

    device: torch.device = pick_device()
    dtype: torch.dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForSeq2SeqLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    draft = AutoModelForSeq2SeqLM.from_pretrained(args.draft, dtype=dtype).to(device).eval()
    if target.config.vocab_size != draft.config.vocab_size:
        raise SystemExit("The draft and target models must share a vocabulary.")

    inputs: list[str] = load_inputs(args.task, args.num_examples)
    result: EvaluationResult = evaluate(
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        inputs=inputs,
        temperature=args.temperature,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        run_baseline=not args.skip_baseline,
        use_cache=not args.no_cache,
        static_cache=args.static_cache,
    )

    report: dict = {"task": args.task.value, "target": args.target, "draft": args.draft, "device": device.type}
    report.update(asdict(result))
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
