"""
This script fine-tunes one T5 v1.1 model on one task.

    python -m finetune.train --set model=google/t5-v1_1-small task=translation output_dir=runs/small-translation
    torchrun --nproc-per-node 4 -m finetune.train --set model=google/t5-v1_1-xxl task=translation strategy=fsdp ...

On the cluster use the Jobs of k8s/finetune-<size>-<task>.yaml.tmpl.
"""

import json
import logging
import math
import os
import random
import time
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from speculative_decoding.types import Task

from .data import (
    IGNORE_INDEX,
    PROGRESS_INTERVAL,
    Batch,
    TokenizedPairs,
    collate,
    load_tokenized,
    make_batches,
    prefetch,
    train_batches,
)
from .optim import ShardedAdafactor

logger: logging.Logger = logging.getLogger("finetune")


@dataclass
class Config:
    """
    The settings of one run. Defaults are the recipe of PLAN.md, the rest is set per run by the k8s Jobs (k8s/finetune-*.yaml.tmpl).
    """

    model: str
    task: Task
    output_dir: str
    source: str = "tfds"  # "tfds" or "hf", see `data.iter_pairs`.
    data_dir: str | None = None
    cache_dir: str | None = None  # Where the tokenized data is cached.
    strategy: str = "none"  # "none" (one device), "ddp" or "fsdp" (for XXL).
    precision: str = "bf16"  # "bf16" or "fp32". Parameters and optimizer state are always fp32.
    gradient_checkpointing: bool = False
    max_length: int = 512  # Inputs and targets.
    batch_tokens: int = 65_536  # Per optimizer step, across devices.
    micro_batch_tokens: int = 16_384  # Per device and forward pass.
    learning_rate: float = 1e-3
    clip_threshold: float = 1.0
    dropout: float = 0.1
    z_loss: float = 1e-4
    max_steps: int = 20_000  # Ours: the paper's recipe has no fixed budget, the best validation checkpoint is kept.
    eval_every: int = 1_000
    eval_examples: int = 2_000
    patience: int = 5  # Evaluations without improvement before stopping. 0 never stops early.
    log_every: int = 50
    seed: int = 0

    @classmethod
    def load(cls, path: str | None, overrides: list[str]) -> "Config":
        values: dict = yaml.safe_load(Path(path).read_text()) if path else {}
        for override in overrides:
            key, _, value = override.partition("=")
            values[key] = yaml.safe_load(value)
        known: set[str] = {field.name for field in fields(cls)}
        if unknown := set(values) - known:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        values["task"] = Task(values["task"])
        return cls(**values)


def seq2seq_loss(
    logits: torch.Tensor, labels: torch.Tensor, z_loss: float, normalizer: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The T5 loss: cross entropy plus `z_loss` times the squared log normalizer, summed over the target tokens and divided
    by `normalizer`, which is their count if not given.

    Args:
        logits: shape (batch, length, vocab_size).
        labels: shape (batch, length), `IGNORE_INDEX` marks padding.
        normalizer: What to divide the sums by. Training passes the number of target tokens of the whole optimizer step,
            so that every token weighs the same however the step is split into micro-batches.

    Returns:
        The loss and the cross entropy alone (the validation metric).
    """

    mask: torch.Tensor = labels != IGNORE_INDEX
    logits = logits.float()
    log_z: torch.Tensor = torch.logsumexp(logits, dim=-1)
    gold: torch.Tensor = logits.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    nll: torch.Tensor = (log_z - gold) * mask
    count: torch.Tensor | float = mask.sum().clamp_min(1) if normalizer is None else normalizer
    return (nll + z_loss * log_z.square() * mask).sum() / count, nll.sum() / count


def make_contiguous(model: torch.nn.Module) -> int:
    """
    Store every parameter in row-major order. Some checkpoints, like the hub copy of T5 v1.1 XXL, load transposed
    weights (a stride of (1, rows)), which FSDP refuses to shard. One parameter is copied at a time.

    Returns:
        How many parameters were not contiguous.
    """

    fixed: int = 0
    for parameter in model.parameters():
        if not parameter.is_contiguous():
            parameter.data = parameter.data.contiguous()
            fixed += 1
    return fixed


def shift_right(labels: torch.Tensor, start_token_id: int, pad_token_id: int) -> torch.Tensor:
    """
    The decoder input: the labels moved one position to the right behind the start token.
    """

    shifted: torch.Tensor = labels.new_full(labels.shape, pad_token_id)
    shifted[:, 0] = start_token_id
    shifted[:, 1:] = labels[:, :-1]
    return shifted.masked_fill(shifted == IGNORE_INDEX, pad_token_id)


class Trainer:
    """
    One fine-tuning run.
    """

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.rank: int = int(os.environ.get("RANK", "0"))
        self.world_size: int = int(os.environ.get("WORLD_SIZE", "1"))
        distributed: bool = self.world_size > 1

        if torch.cuda.is_available():
            self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if distributed:
            logger.info("Rank %d: joining the process group of %d", self.rank, self.world_size)
            # The other ranks wait at a barrier while rank 0 downloads and tokenizes, which outlasts the 10 minute default.
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo", timeout=timedelta(hours=3))
        if config.strategy in ("ddp", "fsdp") and not distributed:
            raise ValueError(f"strategy {config.strategy!r} needs torchrun with more than one process")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed + self.rank)  # Dropout differs across ranks.

        if distributed:
            logger.info("Rank %d: joined the process group", self.rank)
        self.output_dir: Path = Path(config.output_dir)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model)
        logger.info("Rank %d: loading %s (the first time this downloads it)", self.rank, config.model)
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            config.model, dropout_rate=config.dropout, dtype=torch.float32
        )
        self.model.config.use_cache = False
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pad_id: int = self.model.config.pad_token_id
        self.start_id: int = self.model.config.decoder_start_token_id

        logger.info("Rank %d: model loaded, moving it to %s", self.rank, self.device)
        self.model.to(self.device)
        if fixed := make_contiguous(self.model):
            logger.info("Rank %d: made %d non-contiguous parameters contiguous", self.rank, fixed)
        logger.info("Rank %d: model on the device, wrapping with strategy %s", self.rank, config.strategy)
        self.engine: torch.nn.Module = self._wrap(self.model)
        self.optimizer: torch.optim.Optimizer = self._optimizer()

        per_step: int = config.micro_batch_tokens * self.world_size
        self.accumulation: int = max(1, round(config.batch_tokens / per_step))
        if self.rank == 0:
            logger.info(
                "%d device(s) x %d micro-batches x ~%d tokens = ~%d tokens per step",
                self.world_size,
                self.accumulation,
                config.micro_batch_tokens,
                self.accumulation * per_step,
            )

    def _optimizer(self) -> torch.optim.Optimizer:
        config: Config = self.config
        if config.strategy == "fsdp":  # HF Adafactor can't update DTensors, see optim.py.
            return ShardedAdafactor(
                self.engine.parameters(), lr=config.learning_rate, clip_threshold=config.clip_threshold
            )
        return Adafactor(
            self.engine.parameters(),
            lr=config.learning_rate,
            clip_threshold=config.clip_threshold,
            beta1=None,
            scale_parameter=True,
            relative_step=False,
            warmup_init=False,
        )

    def _wrap(self, model: PreTrainedModel) -> torch.nn.Module:
        strategy: str = self.config.strategy
        if strategy == "ddp":
            return DistributedDataParallel(model, device_ids=[self.device.index])
        if strategy == "fsdp":
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for stack in (model.encoder, model.decoder):
                for block in stack.block:
                    fully_shard(block, mp_policy=policy)
            fully_shard(model, mp_policy=policy)
        return model

    def _autocast(self):
        if self.config.precision == "bf16" and self.config.strategy != "fsdp":  # FSDP casts through its policy.
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _loss(self, batch: Batch, normalizer: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device)
        decoder_input_ids: torch.Tensor = shift_right(batch.labels, self.start_id, self.pad_id)
        with self._autocast():
            logits: torch.Tensor = self.engine(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
        return seq2seq_loss(logits, batch.labels, self.config.z_loss, normalizer)

    def _sync_gradients(self, enabled: bool):
        """
        Context of one micro-batch: gradients are reduced across devices only on the last of an accumulation.
        """

        if enabled:
            return nullcontext()
        if self.config.strategy == "ddp":
            return self.engine.no_sync()
        if self.config.strategy == "fsdp":
            self.engine.set_requires_gradient_sync(False)
        return nullcontext()

    def _step_target_tokens(self, micro_batches: list[Batch]) -> float:
        """
        The number of target tokens in an optimizer step, over all devices.
        """

        tokens: torch.Tensor = torch.tensor(
            [float(sum(batch.target_tokens for batch in micro_batches))], device=self.device, dtype=torch.float64
        )
        if self.world_size > 1:
            dist.all_reduce(tokens)
        return max(tokens.item(), 1.0)

    def train_step(self, micro_batches: list[Batch]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One optimizer step over the micro-batches, which together are the batch of the step.

        Returns:
            The loss of the step and its cross entropy, on the device and not read back.
        """

        # Every target token of the step weighs the same, however the step is split into micro-batches and across
        # devices. Devices average their gradients, so each divides by its share of the step's tokens.
        normalizer: float = self._step_target_tokens(micro_batches) / self.world_size

        loss_total: torch.Tensor = torch.zeros((), device=self.device)
        nll_total: torch.Tensor = torch.zeros((), device=self.device)
        for micro, batch in enumerate(micro_batches):
            last: bool = micro == len(micro_batches) - 1
            if self.config.strategy == "fsdp":
                self.engine.set_requires_gradient_sync(last)
            with self._sync_gradients(last):
                loss, nll = self._loss(batch, normalizer)
                loss.backward()
            loss_total += loss.detach()
            nll_total += nll.detach()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss_total, nll_total

    @torch.no_grad()  # Not inference_mode: FSDP2 updates its unsharded parameters in place.
    def validate(self, data: TokenizedPairs) -> float:
        """
        The mean cross entropy per target token over the validation examples, the same on every rank.
        """

        self.engine.eval()
        batches = make_batches(data, self.config.micro_batch_tokens, None)
        total: torch.Tensor = torch.zeros(2, device=self.device, dtype=torch.float64)
        # Sharded models need every rank to run the same number of forwards. A rank that runs out of batches repeats
        # one without counting it.
        for start in tqdm(
            range(0, len(batches), self.world_size),
            desc="Validation",
            unit="batch",
            leave=False,
            disable=self.rank != 0,
            mininterval=PROGRESS_INTERVAL,
        ):
            position: int = start + self.rank
            batch = collate(data, batches[position if position < len(batches) else 0], self.pad_id)
            _, nll = self._loss(batch)
            tokens: int = batch.target_tokens if position < len(batches) else 0
            total += torch.tensor([nll.item() * tokens, tokens], device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(total)
        self.engine.train()
        return (total[0] / total[1]).item()

    def save(self, directory: Path) -> None:
        """
        Save the weights, gathered into one full copy for FSDP, and the tokenizer.
        """

        state_dict = None
        if self.config.strategy == "fsdp":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            state_dict = get_model_state_dict(
                self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
        if self.rank == 0:
            self.model.config.use_cache = True
            self.model.save_pretrained(directory, state_dict=state_dict)
            self.tokenizer.save_pretrained(directory)
            self.model.config.use_cache = False
        if self.world_size > 1:
            dist.barrier()

    def log(self, **values) -> None:
        if self.rank == 0:
            logger.info(json.dumps(values))
            with open(self.output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
                file.write(json.dumps(values) + "\n")

    def run(self) -> None:
        config: Config = self.config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            (self.output_dir / "config.json").write_text(json.dumps({**vars(config), "task": str(config.task)}))

        load = {
            "task": config.task,
            "tokenizer": self.tokenizer,
            "max_length": config.max_length,
            "source": config.source,
            "data_dir": config.data_dir,
            "cache_dir": config.cache_dir,
        }
        # Rank 0 downloads and tokenizes into the cache, the other ranks then read it.
        if self.rank != 0:
            dist.barrier()
        train: TokenizedPairs = load_tokenized(split="train", **load)
        validation: TokenizedPairs = load_tokenized(split="validation", limit=config.eval_examples, **load)
        if self.rank == 0 and self.world_size > 1:
            dist.barrier()
        logger.info("Data ready: %d train and %d validation examples", len(train), len(validation))
        stream = prefetch(
            train_batches(train, config.micro_batch_tokens, self.rank, self.world_size, config.seed),
            pin_memory=self.device.type == "cuda",
        )

        parameters: int = sum(p.numel() for p in self.model.parameters())
        self.log(
            event="start",
            model=config.model,
            task=str(config.task),
            parameters=parameters,
            devices=self.world_size,
            strategy=config.strategy,
            train_examples=len(train),
            validation_examples=len(validation),
            micro_batches_per_step=self.accumulation,
            max_steps=config.max_steps,
        )

        best: float = float("inf")
        stale: int = 0
        began: float = time.perf_counter()
        window_began: float = began
        window_steps: int = 0
        window_real_tokens: int = 0
        window_padded_tokens: int = 0
        window_wait: float = 0.0
        loss_total: torch.Tensor = torch.zeros((), device=self.device)
        nll_total: torch.Tensor = torch.zeros((), device=self.device)
        self.engine.train()
        with (
            logging_redirect_tqdm(),
            tqdm(
                total=config.max_steps,
                desc="Training",
                unit="step",
                disable=self.rank != 0,
                mininterval=PROGRESS_INTERVAL,
                dynamic_ncols=True,
            ) as progress,
        ):
            for step in range(1, config.max_steps + 1):
                waited: float = time.perf_counter()
                micro_batches: list[Batch] = [next(stream) for _ in range(self.accumulation)]
                window_wait += time.perf_counter() - waited

                loss, nll = self.train_step(micro_batches)
                loss_total += loss  # Read back only when logging, to not sync the GPU.
                nll_total += nll
                for batch in micro_batches:
                    window_real_tokens += batch.real_tokens
                    window_padded_tokens += batch.padded_tokens
                progress.update()
                window_steps += 1

                if step % config.log_every == 0:
                    now: float = time.perf_counter()
                    loss_value: float = loss_total.item() / window_steps
                    nll_value: float = nll_total.item() / window_steps
                    seconds: float = (now - window_began) / window_steps
                    tokens_per_second: float = window_real_tokens * self.world_size / (now - window_began)
                    progress.set_postfix(loss=f"{loss_value:.3f}", tok_s=f"{tokens_per_second:,.0f}")
                    self.log(
                        step=step,
                        loss=round(loss_value, 4),  # Cross entropy plus the z-loss.
                        nll=round(nll_value, 4),  # Cross entropy alone, comparable to val_loss.
                        step_seconds=round(seconds, 3),
                        data_wait_ms=round(1000 * window_wait / window_steps),  # Time the step waited for a batch.
                        tokens_per_second=round(tokens_per_second),
                        padding_fraction=round(1 - window_real_tokens / window_padded_tokens, 3),
                        gpu_memory_gb=(
                            round(torch.cuda.max_memory_allocated(self.device) / 2**30, 1)
                            if self.device.type == "cuda"
                            else None
                        ),
                        elapsed_minutes=round((now - began) / 60, 1),
                        eta_minutes=round(seconds * (config.max_steps - step) / 60, 1),
                    )
                    loss_total.zero_()
                    nll_total.zero_()
                    window_began, window_steps, window_wait = now, 0, 0.0
                    window_real_tokens = window_padded_tokens = 0

                if step % config.eval_every == 0 or step == config.max_steps:
                    paused: float = time.perf_counter()
                    val_loss: float = self.validate(validation)
                    improved: bool = val_loss < best
                    self.log(
                        step=step,
                        val_loss=round(val_loss, 4),
                        val_perplexity=round(math.exp(val_loss), 3),
                        best_so_far=min(best, val_loss),
                        improved=improved,
                    )
                    progress.set_postfix(val_loss=f"{val_loss:.3f}")
                    if improved:
                        best, stale = val_loss, 0
                        self.save(self.output_dir / "best")
                        self.log(step=step, saved=str(self.output_dir / "best"))
                    else:
                        stale += 1
                        self.log(step=step, evals_without_improvement=stale, patience=config.patience)
                        if config.patience and stale >= config.patience:
                            self.log(step=step, stopped="patience")
                            break
                    window_began += time.perf_counter() - paused  # Keep validation and saving out of the throughput.

        self.log(event="done", best_val_loss=best, minutes=round((time.perf_counter() - began) / 60, 1))

        if dist.is_initialized():
            dist.destroy_process_group()


def parse_arguments() -> Namespace:
    parser: ArgumentParser = ArgumentParser("Fine-tune one T5 v1.1 model on one task")
    parser.add_argument("--config", help="A yaml file of config values. The k8s Jobs pass them with --set instead.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Overrides of config values.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # One INFO line per hub request otherwise.
    arguments: Namespace = parse_arguments()
    Trainer(Config.load(arguments.config, arguments.set)).run()


if __name__ == "__main__":
    main()
