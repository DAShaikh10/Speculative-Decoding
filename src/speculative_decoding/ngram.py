"""
This module contains the unigram and bigram draft models of the paper's GPT-like experiment (Table 3).

An n-gram model is a table of next-token log probabilities, fitted by counting tokens in a text corpus, so a draft step
is a lookup and the cost coefficient c is close to 0. The model has no state: the next token depends on the previous
token alone (bigram) or on nothing (unigram), so there is no KV cache and rolling back rejected guesses costs nothing.

`NGramRunner` has the interface of the runners of `runner.py`, so it can be passed to the decoding functions in place
of a draft model.
"""

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from .runner import BaseRunner

logger: logging.Logger = logging.getLogger(__name__)

# The prefixes that name an n-gram draft on the command line: "unigram:<path>" or "bigram:<path>".
ORDERS: dict[str, int] = {"unigram": 1, "bigram": 2}

# The number of table entries that are built at once, which is 256 MB of float32.
CHUNK_ENTRIES: int = 2**26

# The number of token pairs that are held before they are merged into the counts.
PENDING_PAIRS: int = 20_000_000


def parse_ngram_spec(name: str) -> tuple[int, Path] | None:
    """
    Read a draft name such as "bigram:/cache/ngram/pile.pt" into its order and the path of the fitted counts.

    Returns:
        None if the name is not an n-gram draft but a model name.
    """

    prefix, separator, path = name.partition(":")
    if not separator or prefix not in ORDERS:
        return None
    return ORDERS[prefix], Path(path)


@dataclass
class NGramCounts:
    """
    The counts of a corpus, from which both the unigram and the bigram model are built.
    """

    vocab_size: int  # The size of the output layer of the target model, which the draft's distribution must match.
    tokenizer_size: int  # The number of tokens of the tokenizer, to check that the counts belong to the target.
    unigram: torch.Tensor  # (vocab_size,) how often each token follows a token of the same document.
    bigram_codes: torch.Tensor  # (pairs,) the sorted codes `previous * vocab_size + next` of the pairs that occur.
    bigram_counts: torch.Tensor  # (pairs,) how often each of them occurs.
    num_tokens: int = 0  # The number of tokens that were predicted, which is what the counts add up to.
    source: str = ""  # Where the counts come from, for the record.

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: Path) -> "NGramCounts":
        return cls(**torch.load(path, map_location="cpu", weights_only=True))


class CountAccumulator:
    """
    Counts the tokens and the pairs of consecutive tokens of documents, without pairing across documents.

    A document is wrapped in the start token and the end token, so that the bigram model learns how a text begins, which
    is the context of unconditional generation, and how it ends.
    """

    def __init__(self, vocab_size: int, start_id: int, end_id: int, max_document_tokens: int | None = None) -> None:
        self.vocab_size: int = vocab_size
        self.start_id: int = start_id
        self.end_id: int = end_id
        self.max_document_tokens: int | None = max_document_tokens

        self.unigram: torch.Tensor = torch.zeros(vocab_size, dtype=torch.long)
        self.codes: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.counts: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.num_tokens: int = 0
        self._pending: list[torch.Tensor] = []
        self._pending_size: int = 0

    def add(self, documents: Iterable[list[int]]) -> int:
        """
        Count a batch of documents, given as lists of token ids.

        Returns:
            The number of tokens that were counted.
        """

        previous: list[torch.Tensor] = []
        following: list[torch.Tensor] = []
        for document in documents:
            if self.max_document_tokens is not None and len(document) > self.max_document_tokens:
                document = document[: self.max_document_tokens]
                ids = torch.tensor([self.start_id, *document])  # A cut document has no end.
            else:
                ids = torch.tensor([self.start_id, *document, self.end_id])
            previous.append(ids[:-1])
            following.append(ids[1:])

        if not previous:
            return 0
        previous_ids: torch.Tensor = torch.cat(previous)
        following_ids: torch.Tensor = torch.cat(following)
        if int(following_ids.max()) >= self.vocab_size:
            raise ValueError(f"A token id is not below the vocabulary size {self.vocab_size}")

        self.unigram += torch.bincount(following_ids, minlength=self.vocab_size)
        self._pending.append(previous_ids * self.vocab_size + following_ids)
        self._pending_size += len(previous_ids)
        self.num_tokens += len(following_ids)
        if self._pending_size >= PENDING_PAIRS:
            self._merge()
        return len(following_ids)

    def _merge(self) -> None:
        if not self._pending:
            return
        codes: torch.Tensor = torch.cat([self.codes, *self._pending])
        unique, inverse = torch.unique(codes, return_inverse=True)
        pending_counts: torch.Tensor = torch.ones(codes.shape[0] - len(self.codes), dtype=torch.long)
        counts: torch.Tensor = torch.cat([self.counts, pending_counts])
        self.counts = torch.zeros(len(unique), dtype=torch.long).index_add_(0, inverse, counts)
        self.codes = unique
        self._pending, self._pending_size = [], 0

    def finish(self, tokenizer_size: int, source: str = "") -> NGramCounts:
        self._merge()
        return NGramCounts(
            vocab_size=self.vocab_size,
            tokenizer_size=tokenizer_size,
            unigram=self.unigram,
            bigram_codes=self.codes,
            bigram_counts=self.counts,
            num_tokens=self.num_tokens,
            source=source,
        )


def build_log_probs(
    counts: NGramCounts,
    order: int,
    smoothing: float = 1.0,
    backoff_strength: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    The table of next-token log probabilities of the unigram or the bigram model.

    The unigram model is add-`smoothing`: every token has at least `smoothing` pseudo counts, so that no probability is
    zero, which would break sampling and the acceptance test.

    The bigram model backs off to the unigram model: p(y | x) = (count(x, y) + backoff_strength * unigram(y)) /
    (count(x) + backoff_strength). A token that never occurred in the corpus gets the unigram distribution.

    Returns:
        Shape (1, vocab_size) for the unigram model, and (vocab_size, vocab_size) for the bigram model, whose row x holds
        the log probabilities of the tokens that follow x. A bigram table is 4 bytes per entry in float32, 10 GB for a
        vocabulary of 50k tokens.
    """

    if order not in (1, 2):
        raise ValueError("Only unigram (1) and bigram (2) models are supported")

    size: int = counts.vocab_size
    unigram: torch.Tensor = counts.unigram.to(device=device, dtype=torch.float64) + smoothing
    unigram = (unigram / unigram.sum()).to(torch.float32)
    if order == 1:
        return unigram.log().to(dtype)[None]

    codes: torch.Tensor = counts.bigram_codes.to(device)
    values: torch.Tensor = counts.bigram_counts.to(device=device, dtype=torch.float32)
    rows: torch.Tensor = codes // size
    columns: torch.Tensor = codes % size

    table: torch.Tensor = torch.empty((size, size), dtype=dtype, device=device)
    step: int = max(1, CHUNK_ENTRIES // size)
    for first in range(0, size, step):
        last: int = min(first + step, size)
        # The codes are sorted, so the pairs of these rows are one slice.
        begin, end = torch.searchsorted(rows, torch.tensor([first, last], device=device)).tolist()
        chunk: torch.Tensor = torch.zeros((last - first, size), dtype=torch.float32, device=device)
        chunk.index_put_((rows[begin:end] - first, columns[begin:end]), values[begin:end], accumulate=True)
        context_counts: torch.Tensor = chunk.sum(dim=1, keepdim=True)
        chunk = (chunk + backoff_strength * unigram) / (context_counts + backoff_strength)
        table[first:last] = chunk.log().to(dtype)
    return table


@dataclass
class NGramSlot:
    """The state of a decoding run of an n-gram model, which is none."""

    cache: None = None


class NGramRunner(BaseRunner):
    """
    A unigram or bigram model with the interface of the runners, to be used as the draft model.

    Its `forward` returns log probabilities as logits. The decoding functions turn logits into distributions with a
    softmax, which gives the probabilities back, and the temperature then sharpens or flattens them like the logits of a
    neural network. At temperature 0 the draft picks the most probable token.
    """

    def __init__(self, table: torch.Tensor, order: int, capacity: int = 2**31 - 1) -> None:
        """
        Args:
            table: The log probabilities from `build_log_probs`.
            order: 1 for a unigram model, 2 for a bigram model.
            capacity: The longest sequence the runner serves. An n-gram model has no cache, so there is no limit.
        """

        model = SimpleNamespace(
            config=SimpleNamespace(is_encoder_decoder=False, vocab_size=table.shape[-1], eos_token_id=None),
            generation_config=SimpleNamespace(eos_token_id=None),
            dtype=table.dtype,
        )
        super().__init__(model, capacity)
        self.table: torch.Tensor = table
        self.order: int = order

    @classmethod
    def from_file(
        cls,
        path: Path,
        order: int,
        dtype: torch.dtype,
        device: torch.device | str,
        tokenizer_size: int | None = None,
        smoothing: float = 1.0,
        backoff_strength: float = 1.0,
    ) -> "NGramRunner":
        """
        Build the model from fitted counts.

        Args:
            tokenizer_size: The number of tokens of the target's tokenizer, which the counts must have been fitted with.
        """

        counts: NGramCounts = NGramCounts.load(path)
        if tokenizer_size is not None and counts.tokenizer_size != tokenizer_size:
            raise ValueError(
                f"{path} was fitted with a tokenizer of {counts.tokenizer_size} tokens, the target's has {tokenizer_size}"
            )
        logger.info("Building the order-%d model from %d tokens of %s", order, counts.num_tokens, counts.source)
        return cls(build_log_probs(counts, order, smoothing, backoff_strength, dtype, device), order)

    @torch.inference_mode()
    def prepare(self, input_ids: torch.Tensor) -> NGramSlot:  # pylint: disable=unused-argument
        return NGramSlot()

    @torch.inference_mode()
    def prefill(self, slot: NGramSlot, token_ids: torch.Tensor) -> None:
        """Nothing to process, a prompt has no effect but through its last token."""

    @torch.inference_mode()
    def forward(self, slot: NGramSlot, token_ids: torch.Tensor, position: int) -> torch.Tensor:  # pylint: disable=unused-argument
        """
        Args:
            token_ids: The new tokens, shape (1, num_tokens).

        Returns:
            The log probabilities of the token after each of them, shape (num_tokens, vocab_size).
        """

        if self.order == 1:
            return self.table.expand(token_ids.shape[1], -1)
        return self.table[token_ids[0]]

    def warmup(self, input_ids: torch.Tensor, step_lengths: Iterable[int], repeats: int = 3) -> None:  # pylint: disable=unused-argument
        """Nothing to compile."""

    def _forward(self, *args) -> torch.Tensor:
        """The runner base class wants a step to compile, and there is none to compile."""

        raise NotImplementedError("An n-gram model has no decoding step")

    @property
    def dtype(self) -> torch.dtype:
        return self.table.dtype


def documents_from(texts: Iterable[str], tokenizer, batch_size: int = 256) -> Iterator[list[list[int]]]:
    """
    Tokenize texts in batches, without special tokens, which `CountAccumulator` adds to every document.
    """

    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            yield tokenizer(batch, add_special_tokens=False)["input_ids"]
            batch = []
    if batch:
        yield tokenizer(batch, add_special_tokens=False)["input_ids"]
