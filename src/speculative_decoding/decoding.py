"""
This module contains the decoding algorithms.
"""

from dataclasses import dataclass
from typing import Self

import torch
from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedConfig, PreTrainedModel, StaticCache

from .sampling import speculative_sample, standardize


@dataclass
class DecodingStats:
    """
    Statistics collected during speculative decoding.
    """

    iterations: int = 0  # The number of target model calls.
    drafted: int = 0  # The number of tokens proposed by the draft model.
    accepted: int = 0  # The number of draft tokens accepted by the target model.
    rejected: int = 0  # The number of draft tokens rejected by the target model.

    @property
    def acceptance_rate(self) -> float:
        """
        The empirical estimate of alpha. Draft tokens after the first rejection of an iteration are never evaluated,
        so they are left out, otherwise the estimate would be biased low.
        """
        evaluated: int = self.accepted + self.rejected
        return self.accepted / evaluated if evaluated else 0.0

    def __iadd__(self, other: "DecodingStats") -> Self:
        self.iterations += other.iterations
        self.drafted += other.drafted
        self.accepted += other.accepted
        self.rejected += other.rejected
        return self


def static_capacity(model: PreTrainedModel, input_ids: torch.Tensor, extra_positions: int) -> int:
    """
    The capacity of a static KV cache that can hold the starting sequence plus `extra_positions` more positions.
    """

    start: int = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    return start + extra_positions


class DecoderState:
    """
    The decoding state of one model: the token sequence being decoded, and the KV cache of its prefix.

    Encoder-decoder models encode the input once and decode from the start token.
    Decoder-only models treat the input as the beginning of the sequence being decoded.

    The cache may hold fewer positions than the sequence. `logits` feeds the model only the positions that aren't
    cached yet, so appending a token costs one position and not the whole sequence. `truncate` rolls the sequence, and
    the cache, back, which is what speculative decoding needs after a rejection.

    The sequence lives in a buffer on the device, and only its length is kept on the CPU. A token sampled on the device
    can then be appended and fed back to the model without the CPU waiting for the device to produce its value.

    By default the KV cache grows as positions are added. With `capacity` it is a static cache, preallocated for that
    many positions of the sequence, whose tensors are updated in place and never change shape. That is what lets the
    model step be compiled, since a growing cache changes the shapes of every call. Rolling a static cache back only
    resets its position counters, because the stale entries past the counters are masked out of attention.
    """

    def __init__(
        self, model: PreTrainedModel, input_ids: torch.Tensor, use_cache: bool = True, capacity: int | None = None
    ) -> None:
        """
        Args:
            model: The model to decode with. It must be in eval mode.
            input_ids: The input (prompt) token ids, shape (1, input_len).
            use_cache: Whether to keep a KV cache. Without it every call re-processes the whole sequence.
            capacity: The number of sequence positions a static KV cache can hold, or None for a growing cache. The
                sequence can never be longer than this, see `static_capacity`.
        """

        if capacity is not None and not use_cache:
            raise ValueError("A static cache needs use_cache=True")

        self.model: PreTrainedModel = model
        self.use_cache: bool = use_cache
        self.capacity: int | None = capacity
        self.device: torch.device = input_ids.device
        self.cached: int = 0  # The number of leading sequence positions the model has processed.

        self.encoder_outputs = None
        if model.config.is_encoder_decoder:
            self.encoder_outputs = model.get_encoder()(input_ids=input_ids)
            initial: torch.Tensor = torch.tensor([model.config.decoder_start_token_id], device=self.device)
        else:
            initial = input_ids[0]

        self.cache: Cache | None = None
        if use_cache:
            self.cache = self._new_cache(model, capacity, input_ids.shape[1])

        self.length: int = len(initial)
        self.buffer: torch.Tensor = torch.empty((1, max(2 * self.length, 64)), dtype=torch.long, device=self.device)
        self.buffer[0, : self.length] = initial
        self.start: int = self.length

    @staticmethod
    def _new_cache(model: PreTrainedModel, capacity: int | None, encoder_length: int) -> Cache:
        """
        Create a new KV cache for the model, either static or dynamic.

        Args:
            model: The model to decode with.
            capacity: The number of sequence positions a static KV cache can hold, or None for a growing cache.
            encoder_length: The length of the encoder input.

        Returns:
            A new cache instance.
        """

        config: PreTrainedConfig = model.config
        if capacity is None:
            if config.is_encoder_decoder:
                # The self-attention and cross-attention keys and values must be cached separately.
                return EncoderDecoderCache(DynamicCache(config=config), DynamicCache(config=config))
            return DynamicCache(config=config)

        if config.is_encoder_decoder:
            # The cross-attention cache holds the encoded input, so it is as long as the input.
            return EncoderDecoderCache(
                StaticCache(config=config, max_cache_len=capacity),
                StaticCache(config=config, max_cache_len=encoder_length),
            )
        return StaticCache(config=config, max_cache_len=capacity)

    def _rewind(self, length: int) -> None:
        """
        Roll a static cache back to `length` positions. Entries past it stay in memory but are masked out.

        Args:
            length: The new length of the sequence. Must be less than or equal to the current
        """

        cache: Cache = self.cache.self_attention_cache if isinstance(self.cache, EncoderDecoderCache) else self.cache
        for layer in cache.layers:
            layer.cumulative_length.fill_(length)

    def __len__(self) -> int:
        return self.length

    @property
    def tokens(self) -> list[int]:
        """The sequence as a list. Copying it to the CPU waits for the device, so keep it out of the decoding loop."""
        return self.buffer[0, : self.length].tolist()

    @tokens.setter
    def tokens(self, tokens: list[int]) -> None:
        self.length = 0
        self.extend(tokens)

    def extend(self, tokens: list[int] | torch.Tensor) -> None:
        """
        Append tokens to the sequence.

        Args:
            tokens: The tokens, as a list or as a 1-D tensor. A tensor on the device is copied without waiting for it.
        """

        count: int = len(tokens)
        if self.length + count > self.buffer.shape[1]:
            grown: torch.Tensor = torch.empty((1, 2 * (self.length + count)), dtype=torch.long, device=self.device)
            grown[0, : self.length] = self.buffer[0, : self.length]
            self.buffer = grown

        if isinstance(tokens, torch.Tensor):
            self.buffer[0, self.length : self.length + count] = tokens
        elif count == 1:
            self.buffer[0, self.length] = tokens[0]  # A scalar is passed to the kernel, with no host-to-device copy.
        else:
            self.buffer[0, self.length : self.length + count] = torch.tensor(tokens)
        self.length += count

    def truncate(self, length: int) -> None:
        """
        Roll the sequence, and the cache, back to the first `length` positions.

        Args:
            length: The new length of the sequence. Must be less than or equal to the current length.
        """

        self.length = length
        if self.cached > length:
            if self.cache is not None:
                if self.capacity is None:
                    # Cache.crop takes a negative number of tokens to remove.
                    self.cache.crop(length - self.cached)
                else:
                    self._rewind(length)
            self.cached = length

    def logits(self) -> torch.Tensor:
        """
        Run the model on the positions that aren't cached yet.

        Returns:
            The logits, shape (uncached_len, vocab_size). Row i predicts the token following uncached position i.
        """

        if self.capacity is not None and self.length > self.capacity:
            raise ValueError(f"The sequence has {self.length} positions, more than the static cache capacity")

        uncached: int = self.length - self.cached
        ids: torch.Tensor = (
            self.buffer[:, self.cached : self.length] if self.use_cache else self.buffer[:, : self.length]
        )

        if self.encoder_outputs is not None:
            output = self.model(
                encoder_outputs=self.encoder_outputs,
                decoder_input_ids=ids,
                past_key_values=self.cache,
                use_cache=self.use_cache,
            )
        else:
            output = self.model(input_ids=ids, past_key_values=self.cache, use_cache=self.use_cache)

        self.cached = self.length
        return output.logits[0, -uncached:]


@torch.inference_mode()
def autoregressive_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    temperature: float,
    max_new_tokens: int,
    use_cache: bool = True,
    static_cache: bool = False,
) -> list[int]:
    """
    Standard autoregressive decoding with a single model, one token per model call. This is the baseline.

    Supports both encoder-decoder models (e.g. T5) and decoder-only models (e.g. GPT).

    Args:
        model: The model to decode from. It must be in eval mode.
        input_ids: The input (prompt) token ids, shape (1, input_len).
        temperature: The sampling temperature. 0 means argmax.
        max_new_tokens: The maximum number of tokens to generate.
        use_cache: Whether to use the KV cache. Without it every call re-processes the whole sequence.
        static_cache: Whether the KV cache is preallocated for the whole generation, instead of growing.

    Returns:
        The generated token ids, excluding the input and the decoder start token, including the EOS token if reached.
    """

    eos_token_id: int | None = model.config.eos_token_id
    capacity: int | None = static_capacity(model, input_ids, max_new_tokens) if static_cache else None
    state: DecoderState = DecoderState(model, input_ids, use_cache, capacity)

    for _ in range(max_new_tokens):
        probs: torch.Tensor = standardize(state.logits()[-1], temperature)
        token: torch.Tensor = torch.multinomial(probs, 1)
        state.extend(token)

        # Stopping at EOS needs the token on the CPU, so every step waits for the device, as Hugging Face does too.
        if int(token.item()) == eos_token_id:
            break

    return state.buffer[0, state.start : state.length].tolist()


@torch.inference_mode()
def speculative_decode(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    input_ids: torch.Tensor,
    temperature: float,
    gamma: int,
    max_new_tokens: int,
    use_cache: bool = True,
    static_cache: bool = False,
) -> tuple[list[int], DecodingStats]:
    """
    Speculative decoding (Algorithm 1 of the paper). The output follows the same distribution as `target` alone.

    Each iteration the draft model proposes `gamma` tokens, then the target model scores all of them in a single
    forward pass. Draft tokens are kept until the first rejection, which is replaced by a token from the adjusted
    distribution. If all are accepted, one extra token is sampled from the target. So each iteration produces between
    1 and gamma + 1 tokens.

    Both models must share a vocabulary, and may each be encoder-decoder or decoder-only.

    Args:
        target: The large model whose distribution we want to sample from. It must be in eval mode.
        draft: The small, fast approximation model. It must be in eval mode.
        input_ids: The input (prompt) token ids, shape (1, input_len).
        temperature: The sampling temperature, applied to both models. 0 means argmax.
        gamma: The number of tokens the draft model proposes per iteration.
        max_new_tokens: The maximum number of tokens to generate.
        use_cache: Whether to use the KV cache. Without it every call re-processes the whole sequence.
        static_cache: Whether the KV caches are preallocated for the whole generation, instead of growing.

    Returns:
        The generated token ids (same convention as `autoregressive_decode`) and the decoding statistics.
    """

    eos_token_id: int | None = target.config.eos_token_id
    # An iteration never feeds the models more than max_new_tokens positions past the start. The extra gamma + 1 is slack.
    extra: int = max_new_tokens + gamma + 1
    target_state: DecoderState = DecoderState(
        target, input_ids, use_cache, static_capacity(target, input_ids, extra) if static_cache else None
    )
    draft_state: DecoderState = DecoderState(
        draft, input_ids, use_cache, static_capacity(draft, input_ids, extra) if static_cache else None
    )
    stats: DecodingStats = DecodingStats()
    generated: list[int] = []

    while len(generated) < max_new_tokens:
        # An iteration emits up to gamma + 1 tokens, so never draft more than what is still needed.
        num_draft: int = min(gamma, max_new_tokens - len(generated) - 1)
        verified: int = len(target_state)  # The length of the sequence both models agree on.

        # Sample num_draft guesses from the draft model autoregressively. The guesses stay on the device.
        guesses: list[torch.Tensor] = []
        draft_probs: torch.Tensor = torch.empty(
            (num_draft, draft.config.vocab_size), dtype=draft.dtype, device=draft_state.device
        )
        for i in range(num_draft):
            draft_probs[i] = standardize(draft_state.logits()[-1], temperature)
            guesses.append(torch.multinomial(draft_probs[i], 1))
            draft_state.extend(guesses[-1])
        guess_ids: torch.Tensor = torch.cat(guesses) if guesses else draft_probs.new_empty(0, dtype=torch.long)

        # Run the target model once on everything it hasn't processed. The last num_draft + 1 rows are p_1 ... p_{n+1}.
        target_state.extend(guess_ids)
        target_probs: torch.Tensor = standardize(target_state.logits()[-(num_draft + 1) :], temperature)
        stats.iterations += 1
        stats.drafted += num_draft

        # Keep guesses until the first rejection, which is replaced by a token from the adjusted distribution. If all
        # guesses were accepted we get an extra token from the target for free. This is the one place that waits for
        # the device: the CPU needs the number of accepted guesses to know how far to roll back.
        num_accepted_tensor, token = speculative_sample(target_probs, draft_probs, guess_ids)
        *drafted_tokens, num_accepted, extra_token = torch.cat(
            [guess_ids, num_accepted_tensor.view(1), token.view(1)]
        ).tolist()
        new_tokens: list[int] = drafted_tokens[:num_accepted] + [extra_token]
        stats.accepted += num_accepted
        stats.rejected += num_accepted < num_draft

        # Stop at the first EOS and at the token limit.
        if eos_token_id is not None and eos_token_id in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
        new_tokens = new_tokens[: max_new_tokens - len(generated)]
        generated.extend(new_tokens)
        if len(generated) >= max_new_tokens or (eos_token_id is not None and generated[-1] == eos_token_id):
            break

        # Roll both models back to the accepted guesses, then append the token that replaced the first rejected one
        # (or the extra token). The positions of the accepted guesses stay cached.
        for state in (target_state, draft_state):
            state.truncate(verified + num_accepted)
            state.extend([extra_token])

    return generated, stats
