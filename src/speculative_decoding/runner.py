"""
This module contains the runner that executes the decoding step of a model with static shapes, so it can be compiled.

In eager PyTorch every decoding step launches hundreds of small kernels from Python. At a batch size of 1 that launch
overhead, and not the arithmetic, sets the time of a step, which makes a small draft model almost as slow as a large
target model. Compiling the step with CUDA graphs replays all of those kernels as one launch. A compiled step needs
everything it sees to keep its shapes and its tensor addresses, which is what the runner arranges:

- the KV caches are static and are created once, then reset for every input,
- the encoder output and mask are padded to a bucket length, and are kept in persistent buffers,
- every step length has its own persistent input buffer,
- the cache position is a tensor that is set inside the compiled step, so rolling back costs no extra launches.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from transformers import EncoderDecoderCache, PreTrainedConfig, PreTrainedModel, StaticCache
from transformers.modeling_outputs import BaseModelOutput

from .sampling import sample_token, standardize

logger: logging.Logger = logging.getLogger(__name__)

# Dynamo compiles one graph per combination of bucket, step length and cache state, and gives up past this many.
MIN_RECOMPILE_LIMIT: int = 128


@dataclass
class Slot:
    """
    What a compiled step needs to see unchanged between decoding runs whose input falls in the same length bucket.
    """

    bucket: int
    cache: EncoderDecoderCache
    encoder_hidden: torch.Tensor  # The encoder output, padded to the bucket, shape (1, bucket, d_model).
    mask: torch.Tensor  # The encoder attention mask, shape (1, bucket). 1 for input tokens and 0 for padding.
    position: torch.Tensor  # The number of positions the self-attention cache holds, a scalar.
    token_buffers: dict[int, torch.Tensor] = field(default_factory=dict)  # The step input, by number of tokens.
    primed: bool = False  # Whether the cross-attention cache has been filled for the current input.


class BaseRunner:
    """
    What the runners of encoder-decoder and decoder-only models share: the model, the cache capacity and the compiled
    step. A subclass defines `_forward`, the decoding step that is compiled.
    """

    def __init__(
        self, model: PreTrainedModel, capacity: int, compile_mode: str | None = None, backend: str = "inductor"
    ) -> None:
        """
        Args:
            model: The model to run, in eval mode.
            capacity: The number of positions of the decoded sequence the KV cache holds.
            compile_mode: The `mode` of `torch.compile`, such as "default" or "reduce-overhead", which uses CUDA graphs.
                None runs the same steps without compiling them.
            backend: The `torch.compile` backend. "inductor" is the real one, others are for testing the tracing.
        """

        self.model: PreTrainedModel = model
        self.capacity: int = capacity
        self.compile_mode: str | None = compile_mode

        self._compile_kwargs: dict = {}
        self._step: Callable = self._forward
        if compile_mode is not None:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, MIN_RECOMPILE_LIMIT)
            # The draft and the target run the same code with different constants, such as the attention scale of their
            # head sizes. Dynamo would turn such a float into a symbolic one when it recompiles for the second model,
            # and fail with "Data dependent operator". With the floats specialized each model gets its own graph.
            torch._dynamo.config.specialize_float = True
            options: dict = {"mode": compile_mode} if backend == "inductor" else {}
            self._compile_kwargs = {"fullgraph": True, "dynamic": False, "backend": backend, **options}
            self._step = torch.compile(self._forward, **self._compile_kwargs)

        self._uses_cuda_graphs: bool = compile_mode == "reduce-overhead" and backend == "inductor"

    @property
    def config(self) -> PreTrainedConfig:
        return self.model.config

    @property
    def generation_config(self):
        return self.model.generation_config

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype


class ModelRunner(BaseRunner):
    """
    Runs the decoding steps of an encoder-decoder model on static caches, optionally compiled.

    The runner can be passed wherever a model is accepted by the decoding functions.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        capacity: int,
        compile_mode: str | None = None,
        backend: str = "inductor",
        encoder_bucket: int = 64,
    ) -> None:
        """
        Args:
            model: The model to run. It must be an encoder-decoder model in eval mode.
            capacity: The number of positions of the decoder sequence, including the start token, the KV cache holds.
            compile_mode: The `mode` of `torch.compile`, such as "default" or "reduce-overhead", which uses CUDA graphs.
                None runs the same steps without compiling them.
            backend: The `torch.compile` backend. "inductor" is the real one, others are for testing the tracing.
            encoder_bucket: The inputs are padded to a multiple of this length, and each length gets its own caches and
                its own compiled graphs.
        """

        if not model.config.is_encoder_decoder:
            raise ValueError("ModelRunner is for encoder-decoder models, use DecoderOnlyRunner for decoder-only ones")

        super().__init__(model, capacity, compile_mode, backend)
        self.encoder_bucket: int = encoder_bucket
        self._slots: dict[int, Slot] = {}

        self._loop: Callable = self._draft_loop
        if compile_mode is not None:
            self._loop = torch.compile(self._draft_loop, **self._compile_kwargs)

    def bucket_for(self, input_length: int) -> int:
        """
        The padded input length the caches and compiled graphs for an input of this length are made for.

        Args:
            input_length: The length of the input, excluding the start token.

        Returns:
            The length of the bucket, which is a multiple of `encoder_bucket`.
        """

        return -(-input_length // self.encoder_bucket) * self.encoder_bucket

    @torch.inference_mode()
    def prepare(self, input_ids: torch.Tensor) -> Slot:
        """
        Encode an input and reset the caches of its bucket, ready for decoding to start from the start token.

        Args:
            input_ids: The input token ids, shape (1, input_len).

        Returns:
            The slot to pass to `forward`. It is reused by every input of the same bucket, so only one decoding run
            can use it at a time.
        """

        length: int = input_ids.shape[1]
        bucket: int = self.bucket_for(length)
        slot: Slot = self._slots.get(bucket) or self._new_slot(bucket)

        padded: torch.Tensor = torch.full(
            (1, bucket), self.config.pad_token_id, dtype=torch.long, device=slot.mask.device
        )
        padded[:, :length] = input_ids
        slot.mask.zero_()
        slot.mask[:, :length] = 1
        slot.encoder_hidden.copy_(
            self.model.get_encoder()(input_ids=padded, attention_mask=slot.mask).last_hidden_state
        )

        for cache in (slot.cache.self_attention_cache, slot.cache.cross_attention_cache):
            torch._foreach_zero_([layer.cumulative_length for layer in cache.layers])
        slot.cache.is_updated.clear()
        slot.primed = False
        return slot

    @torch.inference_mode()
    def forward(self, slot: Slot, token_ids: torch.Tensor, position: int) -> torch.Tensor:
        """
        Run the decoder on new positions.

        Args:
            slot: The slot returned by `prepare`.
            token_ids: The tokens to process, shape (1, num_tokens). They are the positions from `position` on.
            position: The number of positions that are already cached. Anything cached past it is discarded.

        Returns:
            The logits of the new positions, shape (num_tokens, vocab_size). A compiled step may overwrite them on its
            next call, so use them before calling `forward` again.
        """

        buffer: torch.Tensor = self._load_step_input(slot, token_ids, position)

        if self._uses_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        logits: torch.Tensor = self._step(buffer, slot.position, slot.encoder_hidden, slot.mask, slot.cache)

        self._mark_primed(slot)
        return logits

    @torch.inference_mode()
    def draft_loop(
        self, slot: Slot, token_ids: torch.Tensor, position: int, num_guesses: int, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draft `num_guesses` tokens autoregressively in a single compiled call, so that with CUDA graphs the whole loop,
        including the sampling, is one launch instead of one launch and one sampling step per guess.

        Args:
            slot: The slot returned by `prepare`.
            token_ids: The tokens to process first, shape (1, num_tokens). They are the positions from `position` on.
            position: The number of positions that are already cached. Anything cached past it is discarded.
            num_guesses: How many tokens to sample. The last one is not fed back to the model.
            temperature: The sampling temperature. 0 means argmax.

        Returns:
            The guesses, shape (num_guesses,), and the distributions they were sampled from, shape
            (num_guesses, vocab_size). Both are copies, so a later compiled call can't overwrite them.
        """

        if num_guesses < 1:
            raise ValueError("The loop needs at least one guess")
        buffer: torch.Tensor = self._load_step_input(slot, token_ids, position)

        if self._uses_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        guesses, probs = self._loop(
            buffer, slot.position, slot.encoder_hidden, slot.mask, slot.cache, num_guesses, temperature
        )

        self._mark_primed(slot)
        # The target's compiled step begins a new CUDA graph step, which may reuse the memory of these outputs.
        return guesses.clone(), probs.clone()

    @torch.inference_mode()
    def warmup(self, input_ids: torch.Tensor, step_lengths: Iterable[int], repeats: int = 3) -> None:
        """
        Compile and record the graphs for the bucket of an input, so that no compiling happens while timing.

        Every step length is run in both cache states: the first call of an input, which fills the cross-attention
        cache, and the later ones. Recording CUDA graphs takes a few calls, hence the repeats.

        Args:
            input_ids: An input of the bucket to warm up, shape (1, input_len).
            step_lengths: The numbers of tokens the decoding steps will be fed.
            repeats: How many times to run each combination.
        """

        device: torch.device = input_ids.device
        for num_tokens in sorted(set(step_lengths)):
            if 2 * num_tokens > self.capacity:
                raise ValueError(f"A step of {num_tokens} tokens does not fit twice in a capacity of {self.capacity}")
            tokens: torch.Tensor = torch.zeros((1, num_tokens), dtype=torch.long, device=device)
            for _ in range(repeats):
                slot: Slot = self.prepare(input_ids)
                self.forward(slot, tokens, 0)
                self.forward(slot, tokens, num_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logger.info(
            "Warmed up bucket %d for steps of %s tokens", self.bucket_for(input_ids.shape[1]), sorted(set(step_lengths))
        )

    @staticmethod
    def _load_step_input(slot: Slot, token_ids: torch.Tensor, position: int) -> torch.Tensor:
        """
        Copy the step input and the cache position into the persistent buffers a compiled step reads.
        """

        num_tokens: int = token_ids.shape[1]
        buffer: torch.Tensor | None = slot.token_buffers.get(num_tokens)
        if buffer is None:
            buffer = torch.zeros((1, num_tokens), dtype=torch.long, device=slot.mask.device)
            torch._dynamo.mark_static_address(buffer)
            slot.token_buffers[num_tokens] = buffer
        buffer.copy_(token_ids)
        slot.position.fill_(position)
        return buffer

    @staticmethod
    def _mark_primed(slot: Slot) -> None:
        if not slot.primed:
            # The first call fills the cross-attention cache, the later ones reuse it. Setting it here, rather than
            # relying on the compiled code to do it, makes the state certain.
            for layer_idx in range(len(slot.cache.cross_attention_cache.layers)):
                slot.cache.is_updated[layer_idx] = True
            slot.primed = True

    @torch.inference_mode()
    def warmup_loop(
        self,
        input_ids: torch.Tensor,
        step_lengths: Iterable[int],
        num_guesses: int,
        temperature: float,
        repeats: int = 3,
    ) -> None:
        """
        Compile and record the graphs of `draft_loop` for the bucket of an input, like `warmup` does for single steps.

        Args:
            input_ids: An input of the bucket to warm up, shape (1, input_len).
            step_lengths: The numbers of tokens the first step of a loop can be fed.
            num_guesses: The number of guesses the loop samples.
            temperature: The sampling temperature the loop will run with.
            repeats: How many times to run each combination.
        """

        device: torch.device = input_ids.device
        for num_tokens in sorted(set(step_lengths)):
            if 2 * num_tokens + num_guesses - 1 > self.capacity:
                raise ValueError(f"A loop of {num_guesses} guesses does not fit twice in a capacity of {self.capacity}")
            tokens: torch.Tensor = torch.zeros((1, num_tokens), dtype=torch.long, device=device)
            for _ in range(repeats):
                slot: Slot = self.prepare(input_ids)
                self.draft_loop(slot, tokens, 0, num_guesses, temperature)
                self.draft_loop(slot, tokens, num_tokens, num_guesses, temperature)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _forward(
        self,
        token_ids: torch.Tensor,
        position: torch.Tensor,
        encoder_hidden: torch.Tensor,
        mask: torch.Tensor,
        cache: EncoderDecoderCache,
    ) -> torch.Tensor:
        """
        The decoding step that is compiled. Everything it touches is passed in, so that it keeps the same identities.
        """

        # Roll the self-attention cache back or forward to `position`. Entries past it are masked out of attention.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The drafting loop that is compiled. The cache position advances by itself as the steps write to the cache, so
        it is only set once, and the sampled token is fed back to the model without leaving the device.
        """

        for layer in cache.self_attention_cache.layers:
            layer.cumulative_length.copy_(position)

        guesses: list[torch.Tensor] = []
        distributions: list[torch.Tensor] = []
        current: torch.Tensor = token_ids
        for _ in range(num_guesses):
            probs: torch.Tensor = standardize(self._decode(current, encoder_hidden, mask, cache)[-1], temperature)
            guess: torch.Tensor = probs.argmax(dim=-1, keepdim=True) if temperature == 0 else sample_token(probs)
            guesses.append(guess)
            distributions.append(probs)
            current = guess.view(1, 1)
        return torch.cat(guesses), torch.stack(distributions)

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
        return output.logits[0]

    def _new_slot(self, bucket: int) -> Slot:
        """
        Create a new slot for a bucket, with static caches and buffers.

        Args:
            bucket: The length of the input, padded to a multiple of `encoder_bucket`.

        Returns:
            A new slot for the bucket, with static caches and buffers.
        """

        config: PreTrainedConfig = self.config
        cache: EncoderDecoderCache = EncoderDecoderCache(
            StaticCache(config=config, max_cache_len=self.capacity),
            StaticCache(config=config, max_cache_len=bucket),
        )
        # Allocate the cache tensors now. When they are allocated inside a compiled call they are not marked as static
        # addresses, and CUDA graphs can't be used with them.
        device: torch.device = self.model.device
        for sub_cache in (cache.self_attention_cache, cache.cross_attention_cache):
            sub_cache.early_initialization(1, config.num_heads, config.d_kv, self.model.dtype, device)

        with torch.inference_mode():
            encoder_hidden: torch.Tensor = torch.zeros(
                (1, bucket, config.d_model), dtype=self.model.dtype, device=device
            )
            mask: torch.Tensor = torch.zeros((1, bucket), dtype=torch.long, device=device)
            position: torch.Tensor = torch.zeros((), dtype=torch.long, device=device)
        for tensor in (encoder_hidden, mask, position):
            torch._dynamo.mark_static_address(tensor)

        slot: Slot = Slot(bucket, cache, encoder_hidden, mask, position)
        self._slots[bucket] = slot
        return slot


@dataclass
class DecoderOnlySlot:
    """
    What a compiled step of a decoder-only model needs to see unchanged between decoding runs.
    """

    cache: StaticCache
    position: torch.Tensor  # The number of positions the cache holds, a scalar.
    token_buffers: dict[int, torch.Tensor] = field(default_factory=dict)  # The step input, by number of tokens.


class DecoderOnlyRunner(BaseRunner):
    """
    Runs the decoding steps of a decoder-only model on a static cache, optionally compiled.

    There is no encoder and the cache does not depend on the length of the prompt, so a single slot serves every input.
    The prompt is processed by `prefill`, which runs eagerly on the same cache: it is bound by arithmetic and not by
    launch overhead, and it would need a compiled graph for every prompt length.

    The runner can be passed wherever a model is accepted by the decoding functions.
    """

    def __init__(
        self, model: PreTrainedModel, capacity: int, compile_mode: str | None = None, backend: str = "inductor"
    ) -> None:
        """
        Args:
            model: The model to run. It must be a decoder-only model in eval mode.
            capacity: The number of positions, prompt included, the KV cache holds.
            compile_mode: The `mode` of `torch.compile`, such as "default" or "reduce-overhead", which uses CUDA graphs.
                None runs the same steps without compiling them.
            backend: The `torch.compile` backend. "inductor" is the real one, others are for testing the tracing.
        """

        if model.config.is_encoder_decoder:
            raise ValueError("DecoderOnlyRunner is for decoder-only models, use ModelRunner for encoder-decoder ones")

        super().__init__(model, capacity, compile_mode, backend)
        self._slot: DecoderOnlySlot | None = None

    @torch.inference_mode()
    def prepare(self, input_ids: torch.Tensor) -> DecoderOnlySlot:  # pylint: disable=unused-argument
        """
        Reset the cache, ready for a new prompt.

        Args:
            input_ids: The prompt token ids, shape (1, input_len). Unused, the cache does not depend on the prompt.

        Returns:
            The slot to pass to `prefill` and `forward`. It is reused by every input, so only one decoding run can use
            it at a time.
        """

        slot: DecoderOnlySlot = self._slot or self._new_slot()
        torch._foreach_zero_([layer.cumulative_length for layer in slot.cache.layers])
        return slot

    @torch.inference_mode()
    def prefill(self, slot: DecoderOnlySlot, token_ids: torch.Tensor) -> None:
        """
        Process the first positions of the prompt, eagerly, filling the cache. The logits are not needed.

        Args:
            slot: The slot returned by `prepare`.
            token_ids: The tokens to process, shape (1, num_tokens). They are the positions from 0 on.
        """

        torch._foreach_zero_([layer.cumulative_length for layer in slot.cache.layers])
        self.model(input_ids=token_ids, past_key_values=slot.cache, use_cache=True, logits_to_keep=1)

    @torch.inference_mode()
    def forward(self, slot: DecoderOnlySlot, token_ids: torch.Tensor, position: int) -> torch.Tensor:
        """
        Run the model on new positions.

        Args:
            slot: The slot returned by `prepare`.
            token_ids: The tokens to process, shape (1, num_tokens). They are the positions from `position` on.
            position: The number of positions that are already cached. Anything cached past it is discarded.

        Returns:
            The logits of the new positions, shape (num_tokens, vocab_size). A compiled step may overwrite them on its
            next call, so use them before calling `forward` again.
        """

        num_tokens: int = token_ids.shape[1]
        buffer: torch.Tensor | None = slot.token_buffers.get(num_tokens)
        if buffer is None:
            buffer = torch.zeros((1, num_tokens), dtype=torch.long, device=slot.position.device)
            torch._dynamo.mark_static_address(buffer)
            slot.token_buffers[num_tokens] = buffer
        buffer.copy_(token_ids)
        slot.position.fill_(position)

        if self._uses_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        return self._step(buffer, slot.position, slot.cache)

    @torch.inference_mode()
    def warmup(self, input_ids: torch.Tensor, step_lengths: Iterable[int], repeats: int = 3) -> None:
        """
        Compile and record the graphs of the decoding steps, so that no compiling happens while timing.

        Recording CUDA graphs takes a few calls, hence the repeats.

        Args:
            input_ids: An input to warm up with, shape (1, input_len). Only its device matters.
            step_lengths: The numbers of tokens the decoding steps will be fed.
            repeats: How many times to run each step length.
        """

        device: torch.device = input_ids.device
        for num_tokens in sorted(set(step_lengths)):
            if 2 * num_tokens > self.capacity:
                raise ValueError(f"A step of {num_tokens} tokens does not fit twice in a capacity of {self.capacity}")
            tokens: torch.Tensor = torch.zeros((1, num_tokens), dtype=torch.long, device=device)
            for _ in range(repeats):
                slot: DecoderOnlySlot = self.prepare(input_ids)
                self.forward(slot, tokens, 0)
                self.forward(slot, tokens, num_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logger.info("Warmed up the decoder-only steps of %s tokens", sorted(set(step_lengths)))

    def _forward(self, token_ids: torch.Tensor, position: torch.Tensor, cache: StaticCache) -> torch.Tensor:
        """
        The decoding step that is compiled. Everything it touches is passed in, so that it keeps the same identities.
        """

        # Roll the cache back or forward to `position`. Entries past it are masked out of attention.
        for layer in cache.layers:
            layer.cumulative_length.copy_(position)

        output = self.model(input_ids=token_ids, past_key_values=cache, use_cache=True)
        return output.logits[0]

    def _new_slot(self) -> DecoderOnlySlot:
        """
        Create the slot, with a static cache and the position tensor.
        """

        config: PreTrainedConfig = self.config.get_text_config()
        cache: StaticCache = StaticCache(config=config, max_cache_len=self.capacity)
        # Allocate the cache tensors now. When they are allocated inside a compiled call they are not marked as static
        # addresses, and CUDA graphs can't be used with them.
        device: torch.device = self.model.device
        head_dim: int = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        num_kv_heads: int = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        cache.early_initialization(1, num_kv_heads, head_dim, self.model.dtype, device)

        with torch.inference_mode():
            position: torch.Tensor = torch.zeros((), dtype=torch.long, device=device)
        torch._dynamo.mark_static_address(position)

        self._slot = DecoderOnlySlot(cache, position)
        return self._slot
