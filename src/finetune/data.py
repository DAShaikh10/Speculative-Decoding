"""
This module loads, tokenizes and batches the fine-tuning data of a task.
"""

import logging
import queue
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from speculative_decoding.tasks import PREFIXES
from speculative_decoding.types import Task

logger: logging.Logger = logging.getLogger(__name__)

IGNORE_INDEX: int = -100
# Seconds between progress bar updates. A terminal can redraw often, a log file (k8s) would fill with updates.
PROGRESS_INTERVAL: float = 1.0 if sys.stderr.isatty() else 30.0
POOL_SIZE: int = 20_000  # Examples sorted by length together when forming batches. Larger pools pad less.

# The TFDS name of each task, and the TFDS split of each of our split names.
TFDS_NAMES: dict[Task, str] = {
    Task.TRANSLATION: "wmt_t2t_translate/de-en:1.0.0",
    Task.SUMMARIZATION: "cnn_dailymail:3.1.0",
}
TFDS_SPLITS: dict[str, str] = {"train": "train", "validation": "validation", "test": "test"}


def _iter_tfds(task: Task, split: str, data_dir: str | None) -> Iterator[tuple[str, str]]:
    import tensorflow_datasets as tfds  # Optional, see the `finetune` dependency group.

    # Downloads and prepares the dataset on first use, and is a no-op once it is in `data_dir`.
    dataset = tfds.load(TFDS_NAMES[task], split=TFDS_SPLITS[split], data_dir=data_dir, download=True)
    for example in tfds.as_numpy(dataset):
        if task == Task.TRANSLATION:
            yield example["en"].decode(), example["de"].decode()
        else:
            yield example["article"].decode(), example["highlights"].decode()


def _iter_hf(task: Task, split: str, data_dir: str | None) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    if task == Task.TRANSLATION:
        # WMT14 de-en of HF: europarl_v7, commoncrawl and newscommentary_v9 (TFDS has v13), newstest2013 and 2014.
        dataset = load_dataset("wmt/wmt14", "de-en", split=split, cache_dir=data_dir, streaming=split != "train")
        for example in dataset:
            yield example["translation"]["en"], example["translation"]["de"]
    else:
        dataset = load_dataset(
            "abisee/cnn_dailymail", "3.0.0", split=split, cache_dir=data_dir, streaming=split != "train"
        )
        for example in dataset:
            yield example["article"], example["highlights"]


def iter_pairs(
    task: Task, split: str, source: str = "tfds", data_dir: str | None = None, limit: int | None = None
) -> Iterator[tuple[str, str]]:
    """
    Iterate over the (model input, target) texts of a split, with the T5 task prefix on the input.

    Args:
        split: "train", "validation" or "test".
        source: "tfds" for the datasets of the plan (downloaded and prepared into `data_dir` on first use), or "hf" for the HF hub datasets, which differ slightly (see `_iter_hf`).
        limit: Stop after this many pairs.
    """

    if split not in TFDS_SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    pairs = _iter_tfds(task, split, data_dir) if source == "tfds" else _iter_hf(task, split, data_dir)
    prefix: str = PREFIXES[task]
    return ((prefix + text, target) for text, target in islice(pairs, limit))


@dataclass
class TokenizedPairs:
    """
    Tokenized examples. Both sides end with EOS, and are truncated to the maximum length.
    """

    inputs: list[np.ndarray]
    targets: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.inputs)

    @cached_property
    def lengths(self) -> np.ndarray:
        """
        The longer side of each example, which sets how much it pads. Counted once, not at every epoch.
        """

        inputs: np.ndarray = np.fromiter((len(x) for x in self.inputs), dtype=np.int64, count=len(self))
        targets: np.ndarray = np.fromiter((len(y) for y in self.targets), dtype=np.int64, count=len(self))
        return np.maximum(inputs, targets)

    def save(self, path: Path) -> None:
        """
        Save as four flat arrays, which is fast, unlike pickling millions of small ones. The file appears only once it
        is complete, so an interrupted run doesn't leave a broken cache.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path = path.with_name(path.name + ".tmp")
        with open(temporary, "wb") as file:
            np.savez(
                file,
                inputs=np.concatenate(self.inputs),
                input_lengths=np.fromiter((len(x) for x in self.inputs), dtype=np.int32, count=len(self)),
                targets=np.concatenate(self.targets),
                target_lengths=np.fromiter((len(y) for y in self.targets), dtype=np.int32, count=len(self)),
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TokenizedPairs":
        with np.load(path) as arrays:
            return cls(
                np.split(arrays["inputs"], np.cumsum(arrays["input_lengths"])[:-1]),
                np.split(arrays["targets"], np.cumsum(arrays["target_lengths"])[:-1]),
            )


def tokenize_pairs(
    pairs: Iterator[tuple[str, str]], tokenizer: PreTrainedTokenizerBase, max_length: int, chunk: int = 4096
) -> TokenizedPairs:
    """
    Tokenize pairs of texts in chunks. The tokenizer appends EOS, and truncation keeps it.
    """

    result: TokenizedPairs = TokenizedPairs([], [])
    with tqdm(desc="Tokenizing", unit="ex", mininterval=PROGRESS_INTERVAL, dynamic_ncols=True) as progress:
        while batch := list(islice(pairs, chunk)):
            sources, targets = zip(*batch)
            for texts, store in ((sources, result.inputs), (targets, result.targets)):
                ids = tokenizer(list(texts), max_length=max_length, truncation=True)["input_ids"]
                store.extend(np.asarray(sequence, dtype=np.int32) for sequence in ids)
            progress.update(len(batch))
    return result


def load_tokenized(
    task: Task,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    source: str = "tfds",
    data_dir: str | None = None,
    cache_dir: str | None = None,
    limit: int | None = None,
) -> TokenizedPairs:
    """
    Load a split tokenized, through a cache file in `cache_dir` if given.
    """

    cache: Path | None = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{task}-{split}-{source}-{max_length}-{limit}.npz"
        if cache.exists():
            logger.info("Loading the tokenized data from %s", cache)
            return TokenizedPairs.load(cache)

    data: TokenizedPairs = tokenize_pairs(iter_pairs(task, split, source, data_dir, limit), tokenizer, max_length)
    logger.info("Tokenized %d %s examples of %s", len(data), split, task)
    if cache is not None:
        data.save(cache)
        logger.info("Saved the tokenized data to %s (%.0f MB)", cache, cache.stat().st_size / 2**20)
    return data


@dataclass
class Batch:
    """
    A padded batch, with the token counts the trainer logs and normalizes the loss by.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor  # Padded with `IGNORE_INDEX`.
    input_tokens: int  # Real tokens, without padding.
    target_tokens: int

    @property
    def real_tokens(self) -> int:
        return self.input_tokens + self.target_tokens

    @property
    def padded_tokens(self) -> int:
        return self.input_ids.numel() + self.labels.numel()

    def pinned(self) -> "Batch":
        """
        The same batch in page-locked memory, which the GPU copies from without a staging step.
        """

        return Batch(
            self.input_ids.pin_memory(),
            self.attention_mask.pin_memory(),
            self.labels.pin_memory(),
            self.input_tokens,
            self.target_tokens,
        )

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            self.input_ids.to(device, non_blocking=True),
            self.attention_mask.to(device, non_blocking=True),
            self.labels.to(device, non_blocking=True),
            self.input_tokens,
            self.target_tokens,
        )


def make_batches(data: TokenizedPairs, batch_tokens: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    """
    Group examples of similar length into batches of at most `batch_tokens` padded tokens, counting the longer side of
    each example. With `rng` the examples and then the batches are shuffled, otherwise the order is kept.
    """

    lengths: np.ndarray = data.lengths
    order: np.ndarray = rng.permutation(len(data)) if rng is not None else np.arange(len(data))

    batches: list[np.ndarray] = []
    for start in range(0, len(order), POOL_SIZE):
        pool: np.ndarray = order[start : start + POOL_SIZE]
        pool = pool[np.argsort(lengths[pool], kind="stable")]
        pooled: np.ndarray = lengths[pool]
        begin: int = 0
        while begin < len(pool):
            # Sorted ascending, so the last example of a batch is its longest, and the padded size k * length grows with
            # k. A batch of k examples can't be larger than batch_tokens // (the shortest length), which bounds the
            # search to a window, and the largest k that fits is found by binary search. At least one example is kept.
            window: int = min(len(pool) - begin, max(1, batch_tokens // int(pooled[begin])))
            sizes: np.ndarray = np.arange(1, window + 1) * pooled[begin : begin + window]
            count: int = max(1, int(np.searchsorted(sizes, batch_tokens, side="right")))
            batches.append(pool[begin : begin + count])
            begin += count

    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def _pad(sequences: list[np.ndarray], value: int) -> tuple[torch.Tensor, np.ndarray]:
    lengths: np.ndarray = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    padded: np.ndarray = np.full((len(sequences), int(lengths.max())), value, dtype=np.int64)
    # The mask selects the cells of the real tokens in row order, which is the order of the concatenated sequences.
    padded[np.arange(padded.shape[1])[None, :] < lengths[:, None]] = np.concatenate(sequences)
    return torch.from_numpy(padded), lengths


def collate(data: TokenizedPairs, indices: np.ndarray, pad_token_id: int = 0) -> Batch:
    """
    Pad examples into a batch. Labels are padded with `IGNORE_INDEX`.
    """

    input_ids, input_lengths = _pad([data.inputs[i] for i in indices], pad_token_id)
    labels, target_lengths = _pad([data.targets[i] for i in indices], IGNORE_INDEX)
    attention_mask: torch.Tensor = (
        torch.arange(input_ids.shape[1])[None, :] < torch.from_numpy(input_lengths)[:, None]
    ).long()
    return Batch(input_ids, attention_mask, labels, int(input_lengths.sum()), int(target_lengths.sum()))


def train_batches(data: TokenizedPairs, batch_tokens: int, rank: int, world_size: int, seed: int) -> Iterator[Batch]:
    """
    Endless stream of training batches for one rank. Every epoch is reshuffled, and every rank gets the same number of
    batches. The batches of the next epoch are formed in the background while the current one is consumed, so an epoch
    boundary doesn't stall the stream.
    """

    def form(epoch: int) -> list[np.ndarray]:
        batches: list[np.ndarray] = make_batches(data, batch_tokens, np.random.default_rng(seed + epoch))
        usable: int = len(batches) // world_size * world_size
        return batches[:usable][rank::world_size]

    with ThreadPoolExecutor(max_workers=1) as pool:
        epoch: int = 0
        upcoming = pool.submit(form, epoch)
        while True:
            batches = upcoming.result()
            upcoming = pool.submit(form, epoch + 1)
            for indices in batches:
                yield collate(data, indices)
            epoch += 1


def prefetch(batches: Iterator[Batch], depth: int = 4, pin_memory: bool = False) -> Iterator[Batch]:
    """
    Produce batches on a background thread, ahead of the training loop, so that collating them overlaps with the GPU
    step. An error in the producer is raised in the consumer.
    """

    items: queue.Queue = queue.Queue(maxsize=depth)
    finished: object = object()

    def produce() -> None:
        try:
            for batch in batches:
                items.put(batch.pinned() if pin_memory else batch)
        except Exception as error:  # noqa: BLE001
            items.put(error)
        else:
            items.put(finished)

    threading.Thread(target=produce, daemon=True, name="prefetch").start()
    while True:
        item = items.get()
        if item is finished:
            return
        if isinstance(item, Exception):
            raise item
        yield item
