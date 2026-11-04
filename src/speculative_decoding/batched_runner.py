"""
This module contains the runner that executes the decoding steps of an encoder-decoder model for a batch of sequences with static shapes, so they can be compiled.
"""

import gc
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from transformers import EncoderDecoderCache, PreTrainedConfig, PreTrainedModel, StaticCache
from transformers.modeling_outputs import BaseModelOutput

from .runner import BaseRunner
from .sampling import sample_token, standardize

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BatchedSlot:

    bucket: int
    cache: EncoderDecoderCache
    encoder_hidden: torch.Tensor
    mask: torch.Tensor
    position: torch.Tensor
    token_buffers: dict[int, torch.Tensor] = field(default_factory=dict)
    primed: bool = False


def reset_compilation() -> None:

    gc.collect()
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BatchedModelRunner(BaseRunner):

    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int,
        capacity: int,
        compile_mode: str | None = None,
        backend: str = "inductor",
        encoder_bucket: int = 64,
    ) -> None:

        if not model.config.is_encoder_decoder:
            raise ValueError("BatchedModelRunner is for encoder-decoder models")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if encoder_bucket < 1:
            raise ValueError(f"encoder_bucket must be at least 1, got {encoder_bucket}")

        super().__init__(model, capacity, compile_mode, backend)
        self.batch_size: int = batch_size
        self.encoder_bucket: int = encoder_bucket
        self._slots: dict[int, BatchedSlot] = {}

        self._loop = self._draft_loop
        if compile_mode is not None:
            self._loop = torch.compile(self._draft_loop, **self._compile_kwargs)

    def bucket_for(self, input_length: int) -> int:

        return -(-input_length // self.encoder_bucket) * self.encoder_bucket

    @torch.inference_mode()
    def prepare(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BatchedSlot:

        if input_ids.dim() != 2 or input_ids.shape[0] != self.batch_size or attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"Expected input_ids and attention_mask of shape ({self.batch_size}, input_len), "
                f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
            )

        length: int = input_ids.shape[1]
        bucket: int = self.bucket_for(length)
        slot: BatchedSlot = self._slots.get(bucket) or self._new_slot(bucket)

        padded: torch.Tensor = torch.full(
            (self.batch_size, bucket), self.config.pad_token_id, dtype=torch.long, device=slot.mask.device
        )
        padded[:, :length] = input_ids
        slot.mask.zero_()
        slot.mask[:, :length] = attention_mask
        slot.encoder_hidden.copy_(
            self.model.get_encoder()(input_ids=padded, attention_mask=slot.mask).last_hidden_state
        )

        for cache in (slot.cache.self_attention_cache, slot.cache.cross_attention_cache):
            torch._foreach_zero_([layer.cumulative_length for layer in cache.layers])
        slot.cache.is_updated.clear()
        slot.primed = False
        return slot

    @torch.inference_mode()
    def forward(self, slot: BatchedSlot, token_ids: torch.Tensor, position: int) -> torch.Tensor:

        buffer: torch.Tensor = self._load_step_input(slot, token_ids, position)

        if self._uses_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        logits: torch.Tensor = self._step(buffer, slot.position, slot.encoder_hidden, slot.mask, slot.cache)

        self._mark_primed(slot)
        return logits

    @torch.inference_mode()
    def draft_loop(
        self,
        slot: BatchedSlot,
        token_ids: torch.Tensor,
        position: int,
        num_guesses: int,
        temperature: float,
        vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if num_guesses < 1:
            raise ValueError("The loop needs at least one guess")
        if not 1 <= vocab_size <= self.config.vocab_size:
            raise ValueError(f"vocab_size must be in [1, {self.config.vocab_size}], got {vocab_size}")
        buffer: torch.Tensor = self._load_step_input(slot, token_ids, position)

        if self._uses_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        guesses, probs = self._loop(
            buffer, slot.position, slot.encoder_hidden, slot.mask, slot.cache, num_guesses, temperature, vocab_size
        )

        self._mark_primed(slot)
        return guesses.clone(), probs.clone()

    @torch.inference_mode()
    def warmup(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_lengths: Iterable[int],
        repeats: int = 3,
    ) -> None:

        device: torch.device = input_ids.device
        for num_tokens in sorted(set(step_lengths)):
            if 2 * num_tokens > self.capacity:
                raise ValueError(f"A step of {num_tokens} tokens does not fit twice in a capacity of {self.capacity}")
            tokens: torch.Tensor = torch.zeros((self.batch_size, num_tokens), dtype=torch.long, device=device)
            for _ in range(repeats):
                slot: BatchedSlot = self.prepare(input_ids, attention_mask)
                self.forward(slot, tokens, 0)
                self.forward(slot, tokens, num_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logger.info(
            "Warmed up bucket %d for batches of %d and steps of %s tokens",
            self.bucket_for(input_ids.shape[1]),
            self.batch_size,
            sorted(set(step_lengths)),
        )

    @torch.inference_mode()
    def warmup_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_lengths: Iterable[int],
        num_guesses: int,
        temperature: float,
        vocab_size: int,
        repeats: int = 3,
    ) -> None:

        device: torch.device = input_ids.device
        for num_tokens in sorted(set(step_lengths)):
            if 2 * num_tokens + num_guesses - 1 > self.capacity:
                raise ValueError(f"A loop of {num_guesses} guesses does not fit twice in a capacity of {self.capacity}")
            tokens: torch.Tensor = torch.zeros((self.batch_size, num_tokens), dtype=torch.long, device=device)
            for _ in range(repeats):
                slot: BatchedSlot = self.prepare(input_ids, attention_mask)
                self.draft_loop(slot, tokens, 0, num_guesses, temperature, vocab_size)
                self.draft_loop(slot, tokens, num_tokens, num_guesses, temperature, vocab_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _load_step_input(self, slot: BatchedSlot, token_ids: torch.Tensor, position: int) -> torch.Tensor:

        if token_ids.dim() != 2 or token_ids.shape[0] != self.batch_size:
            raise ValueError(
                f"Expected token_ids of shape ({self.batch_size}, num_tokens), got {tuple(token_ids.shape)}"
            )

        num_tokens: int = token_ids.shape[1]
        buffer: torch.Tensor | None = slot.token_buffers.get(num_tokens)
        if buffer is None:
            buffer = torch.zeros((self.batch_size, num_tokens), dtype=torch.long, device=slot.mask.device)
            torch._dynamo.mark_static_address(buffer)
            slot.token_buffers[num_tokens] = buffer
        buffer.copy_(token_ids)
        slot.position.fill_(position)
        return buffer

    @staticmethod
    def _mark_primed(slot: BatchedSlot) -> None:
        if not slot.primed:
            for layer_idx in range(len(slot.cache.cross_attention_cache.layers)):
                slot.cache.is_updated[layer_idx] = True
            slot.primed = True

    def _forward(
        self,
        token_ids: torch.Tensor,
        position: torch.Tensor,
        encoder_hidden: torch.Tensor,
        mask: torch.Tensor,
        cache: EncoderDecoderCache,
    ) -> torch.Tensor:

        for layer in cache.self_attention_cache.layers:
            layer.cumulative_length.copy_(position)
        return self._decode(token_ids, encoder_hidden, mask, cache)

    def _draft_loop(
        self,
        token_ids: torch.Tensor,
        position: torch.Tensor,
        encoder_hidden: torch.Tensor,
        mask: torch.Tensor,
        cache: EncoderDecoderCache,
        num_guesses: int,
        temperature: float,
        vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        for layer in cache.self_attention_cache.layers:
            layer.cumulative_length.copy_(position)

        guesses: list[torch.Tensor] = []
        distributions: list[torch.Tensor] = []
        current: torch.Tensor = token_ids
        for _ in range(num_guesses):
            logits: torch.Tensor = self._decode(current, encoder_hidden, mask, cache)[:, -1, :vocab_size]
            probs: torch.Tensor = standardize(logits, temperature)
            guess: torch.Tensor = probs.argmax(dim=-1, keepdim=True) if temperature == 0 else sample_token(probs)
            guesses.append(guess)
            distributions.append(probs)
            current = guess
        return torch.cat(guesses, dim=1), torch.stack(distributions, dim=1)

    def _decode(
        self, token_ids: torch.Tensor, encoder_hidden: torch.Tensor, mask: torch.Tensor, cache: EncoderDecoderCache
    ) -> torch.Tensor:
        output = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=mask,
            decoder_input_ids=token_ids,
            past_key_values=cache,
            use_cache=True,
        )
        return output.logits

    def _new_slot(self, bucket: int) -> BatchedSlot:

        config: PreTrainedConfig = self.config
        cache: EncoderDecoderCache = EncoderDecoderCache(
            StaticCache(config=config, max_cache_len=self.capacity),
            StaticCache(config=config, max_cache_len=bucket),
        )
        device: torch.device = self.model.device
        for sub_cache in (cache.self_attention_cache, cache.cross_attention_cache):
            sub_cache.early_initialization(self.batch_size, config.num_heads, config.d_kv, self.model.dtype, device)

        with torch.inference_mode():
            encoder_hidden: torch.Tensor = torch.zeros(
                (self.batch_size, bucket, config.d_model), dtype=self.model.dtype, device=device
            )
            mask: torch.Tensor = torch.zeros((self.batch_size, bucket), dtype=torch.long, device=device)
            position: torch.Tensor = torch.zeros((), dtype=torch.long, device=device)
        for tensor in (encoder_hidden, mask, position):
            torch._dynamo.mark_static_address(tensor)

        slot: BatchedSlot = BatchedSlot(bucket, cache, encoder_hidden, mask, position)
        self._slots[bucket] = slot
        return slot
