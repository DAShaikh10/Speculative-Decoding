"""
This script scores a fine-tuned checkpoint with BLEU (translation) or ROUGE (summarization). It only checks that the
pipeline works, the acceptance rate is measured by `finetune.alpha`.

    python -m finetune.eval --checkpoint runs/small-translation/best --task translation
"""

import json
from argparse import ArgumentParser, Namespace

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from speculative_decoding.types import Task

from .data import iter_pairs


def generate_greedy(
    model, tokenizer, inputs: list[str], batch_size: int, max_new_tokens: int, max_length: int
) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(inputs), batch_size):
        batch = tokenizer(
            inputs[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens)
        outputs.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return outputs


def score(task: Task, predictions: list[str], references: list[str]) -> dict[str, float]:
    if task == Task.TRANSLATION:
        import sacrebleu

        return {"bleu": sacrebleu.corpus_bleu(predictions, [references]).score}

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(reference, prediction) for prediction, reference in zip(predictions, references)]
    return {name: 100 * sum(s[name].fmeasure for s in scores) / len(scores) for name in ("rouge1", "rouge2", "rougeL")}


def parse_arguments() -> Namespace:
    parser: ArgumentParser = ArgumentParser("Score a fine-tuned checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, type=Task, choices=list(Task))
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--source", default="tfds", choices=["tfds", "hf"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    arguments: Namespace = parse_arguments()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(arguments.checkpoint).to(device).eval()

    pairs = list(
        iter_pairs(arguments.task, arguments.split, arguments.source, arguments.data_dir, arguments.num_examples)
    )
    predictions = generate_greedy(
        model,
        tokenizer,
        [source for source, _ in pairs],
        arguments.batch_size,
        arguments.max_new_tokens,
        arguments.max_length,
    )
    print(
        json.dumps(
            {
                "task": str(arguments.task),
                "examples": len(pairs),
                **score(arguments.task, predictions, [t for _, t in pairs]),
            }
        )
    )


if __name__ == "__main__":
    main()
