"""
This script measures the acceptance rate alpha of a draft model against a target model (Corollary 3.6 of the paper):
the mean over positions of sum_x min(p(x), q(x)), where p and q are the target and draft distributions given the same
prefix.

The prefixes are the outputs of the target model for test inputs, decoded at temperature 0 (argmax) or with pure
sampling otherwise. Both models are then run teacher-forced over each output, so that each position is one sample of
alpha. Positions are counted up to and including the first EOS.

    python -m finetune.alpha --target runs/large-translation/best --draft runs/small-translation/best \
        --task translation --temperature 0
"""

import json
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel

from speculative_decoding.sampling import standardize
from speculative_decoding.types import Task

from .data import iter_pairs


@dataclass
class AlphaResult:
    task: str
    temperature: float
    num_examples: int
    num_tokens: int
    alpha_analytic: float


@torch.inference_mode()
def measure_alpha(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    tokenizer,
    inputs: list[str],
    temperature: float,
    min_tokens: int = 10_000,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    max_length: int = 512,
) -> tuple[float, int, int]:
    """
    Returns:
        Alpha, the number of positions it was averaged over, and the number of inputs used. Inputs are consumed until
        `min_tokens` positions are counted.
    """

    if temperature == 0:
        sampling: dict = {"do_sample": False}
    else:
        # Pure sampling. The HF default top_k=50 would truncate the target's distribution.
        sampling = {"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0}

    total: float = 0.0
    count: int = 0
    used: int = 0
    for start in range(0, len(inputs), batch_size):
        batch = tokenizer(
            inputs[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(target.device)
        generated: torch.Tensor = target.generate(
            **batch, max_new_tokens=max_new_tokens, **sampling
        )  # Starts with the start token.

        decoder_input: torch.Tensor = generated[:, :-1]
        outputs: torch.Tensor = generated[:, 1:]
        eos: torch.Tensor = outputs == target.config.eos_token_id
        # True up to and including the first EOS. Outputs that hit the cap have no EOS and count in full.
        valid: torch.Tensor = (eos.cumsum(dim=1) - eos.long()) == 0

        p: torch.Tensor = standardize(target(**batch, decoder_input_ids=decoder_input).logits.float(), temperature)
        q: torch.Tensor = standardize(draft(**batch, decoder_input_ids=decoder_input).logits.float(), temperature)
        overlap: torch.Tensor = torch.minimum(p, q).sum(dim=-1)

        total += overlap[valid].sum().item()
        count += int(valid.sum())
        used += batch["input_ids"].shape[0]
        if count >= min_tokens:
            break

    return total / count, count, used


def parse_arguments() -> Namespace:
    parser: ArgumentParser = ArgumentParser("Measure the acceptance rate alpha of a draft model")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--task", required=True, type=Task, choices=list(Task))
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--source", default="tfds", choices=["tfds", "hf"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--num-examples", type=int, default=1000, help="The pool of test inputs.")
    parser.add_argument("--min-tokens", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments: Namespace = parse_arguments()
    torch.manual_seed(arguments.seed)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(arguments.target)
    target = AutoModelForSeq2SeqLM.from_pretrained(arguments.target, dtype=dtype).to(device).eval()
    draft = AutoModelForSeq2SeqLM.from_pretrained(arguments.draft, dtype=dtype).to(device).eval()

    inputs = [
        source
        for source, _ in iter_pairs(
            arguments.task, "test", arguments.source, arguments.data_dir, arguments.num_examples
        )
    ]
    alpha, tokens, used = measure_alpha(
        target,
        draft,
        tokenizer,
        inputs,
        arguments.temperature,
        arguments.min_tokens,
        arguments.batch_size,
        arguments.max_new_tokens,
    )
    print(json.dumps(asdict(AlphaResult(str(arguments.task), arguments.temperature, used, tokens, alpha))))


if __name__ == "__main__":
    main()
