"""
This script replicates the experiment in "Fast Inference of Transformers via Speculative Decoding"
(https://arxiv.org/abs/2211.17192)
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel

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
        "--num-examples",
        default=1000,
        type=int,
        help="The pool of test inputs, of which --min-generated-tokens decodes as many as it needs. Every input "
        "length bucket in the pool is compiled up front, so keep it small. Default: 1000",
    )
    argparser.add_argument(
        "--min-generated-tokens",
        default=10_000,
        type=int,
        help="Stop after the input that brings the generated tokens to this many. The paper measures alpha on 10K "
        "tokens. 0 decodes every input. Default: 10000",
    )
    argparser.add_argument("--max-new-tokens", type=int, default=512, help="The generation limit. Default: 512")
    argparser.add_argument("--max-input-length", type=int, default=512, help="The input truncation. Default: 512")
    argparser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="The model precision. T5 v1.1 overflows in float16, use bfloat16 for large decoder-only models. "
        "Default: float32",
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
    argparser.add_argument(
        "--compile",
        choices=["eager", "default", "reduce-overhead"],
        default=None,
        help="Run the models through runners with static caches and padded inputs, and compile the decoding steps "
        "with torch.compile in this mode. 'reduce-overhead' uses CUDA graphs. 'eager' runs the runners uncompiled. "
        "Needs the KV cache.",
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
    argparser.add_argument("--output", type=Path, default=None, help="Where to write the results as JSON.")

    args: Namespace = argparser.parse_args()
    if (args.static_cache or args.compile) and args.no_cache:
        argparser.error("--static-cache and --compile need the KV cache, so they can't be combined with --no-cache")
    if args.fuse_draft and not args.compile:
        argparser.error("--fuse-draft needs --compile")
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


def load_model(name: str, dtype: torch.dtype, device: torch.device) -> PreTrainedModel:
    """
    Load a model onto the device, an encoder-decoder or a decoder-only one depending on its config.
    """

    config: AutoConfig = AutoConfig.from_pretrained(name)
    auto = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    return auto.from_pretrained(name, dtype=dtype).to(device).eval()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    args: Namespace = parse_arguments()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")  # Allow TF32 tensor cores for float32 matmuls.

    device: torch.device = pick_device()
    dtype: torch.dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if AutoTokenizer.from_pretrained(args.draft).get_vocab() != tokenizer.get_vocab():
        raise SystemExit("The draft and target models must share a vocabulary.")
    target = load_model(args.target, dtype, device)
    draft = load_model(args.draft, dtype, device)
    if target.config.is_encoder_decoder != draft.config.is_encoder_decoder:
        raise SystemExit("The draft and target models must both be encoder-decoder or both be decoder-only.")

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
        compile_mode=args.compile,
        encoder_bucket=args.encoder_bucket,
        min_generated_tokens=args.min_generated_tokens or None,
        fuse_draft=args.fuse_draft,
    )

    report: dict = {
        "task": args.task.value,
        "target": args.target,
        "draft": args.draft,
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "min_generated_tokens": args.min_generated_tokens,
    }
    report.update(asdict(result))
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
