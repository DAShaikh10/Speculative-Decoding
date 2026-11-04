"""
Fit the unigram and bigram draft models of the paper's GPT-like experiment: count the tokens of a text corpus, tokenized
with the target model's tokenizer. One file serves both models, evaluate it with `-d unigram:<file>` or `-d bigram:<file>`.
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from speculative_decoding.ngram import CountAccumulator, NGramCounts, documents_from

logger: logging.Logger = logging.getLogger("fit_ngram")


def parse_arguments() -> Namespace:
    argparser: ArgumentParser = ArgumentParser("Fit unigram and bigram draft models on a text corpus")
    argparser.add_argument(
        "--target",
        "-t",
        required=True,
        type=str,
        help="The target model, whose tokenizer and vocabulary size are used.",
    )
    argparser.add_argument("--output", type=Path, required=True, help="Where to save the counts.")
    argparser.add_argument(
        "--dataset",
        default="monology/pile-uncopyrighted",
        help="The Hugging Face dataset to count, streamed. Pythia was trained on the Pile. Default: %(default)s",
    )
    argparser.add_argument("--dataset-config", default=None, help="The configuration of the dataset, if it has any.")
    argparser.add_argument("--split", default="train", help="The split to count. Default: %(default)s")
    argparser.add_argument("--text-column", default="text", help="The column that holds the text. Default: %(default)s")
    argparser.add_argument(
        "--num-tokens",
        type=int,
        default=100_000_000,
        help="Stop after the document that brings the count to this many tokens. Default: %(default)s",
    )
    argparser.add_argument(
        "--max-document-tokens",
        type=int,
        default=1024,
        help="Cut longer documents, which are then not counted as ended. Default: %(default)s",
    )
    return argparser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args: Namespace = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    vocab_size: int = AutoConfig.from_pretrained(args.target).vocab_size
    start_id: int = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    end_id: int = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else start_id

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    texts = (example[args.text_column] for example in dataset)
    accumulator = CountAccumulator(vocab_size, start_id, end_id, args.max_document_tokens)

    next_report: int = 0
    for documents in documents_from(texts, tokenizer):
        accumulator.add(documents)
        if accumulator.num_tokens >= next_report:
            logger.info("Counted %d tokens", accumulator.num_tokens)
            next_report += 10_000_000
        if accumulator.num_tokens >= args.num_tokens:
            break

    source: str = f"{args.dataset} {args.split}, tokenizer of {args.target}"
    counts: NGramCounts = accumulator.finish(len(tokenizer), source)
    counts.save(args.output)
    logger.info("Saved %d tokens and %d distinct pairs to %s", counts.num_tokens, len(counts.bigram_codes), args.output)


if __name__ == "__main__":
    main()
