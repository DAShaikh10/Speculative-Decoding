"""
This module contains the decoding algorithms.
"""

from dataclasses import dataclass

import torch
from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel

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

    def __iadd__(self, other: "DecodingStats") -> "DecodingStats":
        self.iterations += other.iterations
        self.drafted += other.drafted
        self.accepted += other.accepted
        self.rejected += other.rejected
        return self


class DecoderState:
    """
    The decoding state of one model: the token sequence being decoded, and the KV cache of its prefix.

    Encoder-decoder models encode the input once and decode from the start token.
    Decoder-only models treat the input as the beginning of the sequence being decoded.

    The cache may hold fewer positions than the sequence. `logits` feeds the model only the positions that aren't
    cached yet, so appending a token costs one position and not the whole sequence. `truncate` rolls the sequence, and
    the cache, back, which is what speculative decoding needs after a rejection.
    """

    def __init__(self, model: PreTrainedModel, input_ids: torch.Tensor, use_cache: bool = True) -> None:
        self.model: PreTrainedModel = model
        self.use_cache: bool = use_cache
        self.device: torch.device = input_ids.device
        self.cached: int = 0  # The number of leading sequence positions the model has processed.

        self.cache: Cache | None = None
        if use_cache:
            if model.config.is_encoder_decoder:
                # The self-attention and cross-attention keys and values must be cached separately.
                self.cache = EncoderDecoderCache(DynamicCache(config=model.config), DynamicCache(config=model.config))
            else:
                self.cache = DynamicCache(config=model.config)

        self.encoder_outputs = None
        if model.config.is_encoder_decoder:
            self.encoder_outputs = model.get_encoder()(input_ids=input_ids)
            self.tokens: list[int] = [model.config.decoder_start_token_id]
        else:
            self.tokens = input_ids[0].tolist()
        self.start: int = len(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def extend(self, tokens: list[int]) -> None:
        self.tokens.extend(tokens)

    def truncate(self, length: int) -> None:
        """
        Roll the sequence, and the cache, back to the first `length` positions.

        Args:
            length: The new length of the sequence. Must be less than or equal to the current length.
        """

        del self.tokens[length:]
        if self.cached > length:
            if self.cache is not None:
                # Cache.crop takes a negative number of tokens to remove.
                self.cache.crop(length - self.cached)
            self.cached = length

    def logits(self) -> torch.Tensor:
        """
        Run the model on the positions that aren't cached yet.

        Returns:
            The logits, shape (uncached_len, vocab_size). Row i predicts the token following uncached position i.
        """

        uncached: int = len(self.tokens) - self.cached
        if self.use_cache:
            chunk: list[int] = self.tokens[self.cached :]
        else:
            chunk = self.tokens

        ids: torch.Tensor = torch.tensor([chunk], device=self.device)
        if self.encoder_outputs is not None:
            output = self.model(
                encoder_outputs=self.encoder_outputs,
                decoder_input_ids=ids,
                past_key_values=self.cache,
                use_cache=self.use_cache,
            )
        else:
            output = self.model(input_ids=ids, past_key_values=self.cache, use_cache=self.use_cache)

        self.cached = len(self.tokens)
        return output.logits[0, -uncached:]


@torch.inference_mode()
def autoregressive_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    temperature: float,
    max_new_tokens: int,
    use_cache: bool = True,
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

    Returns:
        The generated token ids, excluding the input and the decoder start token, including the EOS token if reached.
    """

    eos_token_id: int | None = model.config.eos_token_id
    state: DecoderState = DecoderState(model, input_ids, use_cache)

    for _ in range(max_new_tokens):
        probs: torch.Tensor = standardize(state.logits()[-1], temperature)
        token: int = int(torch.multinomial(probs, 1).item())
        state.extend([token])

        if token == eos_token_id:
            break

    return state.tokens[state.start :]


@torch.inference_mode()
def speculative_decode(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    input_ids: torch.Tensor,
    temperature: float,
    gamma: int,
    max_new_tokens: int,
    use_cache: bool = True,
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

    Returns:
        The generated token ids (same convention as `autoregressive_decode`) and the decoding statistics.
    """

    eos_token_id: int | None = target.config.eos_token_id
    target_state: DecoderState = DecoderState(target, input_ids, use_cache)
    draft_state: DecoderState = DecoderState(draft, input_ids, use_cache)
    stats: DecodingStats = DecodingStats()
    generated: list[int] = []

    while len(generated) < max_new_tokens:
        # An iteration emits up to gamma + 1 tokens, so never draft more than what is still needed.
        num_draft: int = min(gamma, max_new_tokens - len(generated) - 1)
        verified: int = len(target_state)  # The length of the sequence both models agree on.

        # Sample num_draft guesses from the draft model autoregressively.
        guesses: list[int] = []
        draft_probs: list[torch.Tensor] = []
        for _ in range(num_draft):
            q: torch.Tensor = standardize(draft_state.logits()[-1], temperature)
            guess: int = int(torch.multinomial(q, 1).item())
            draft_probs.append(q)
            guesses.append(guess)
            draft_state.extend([guess])

        # Run the target model once on everything it hasn't processed. The last num_draft + 1 rows are p_1 ... p_{n+1}.
        target_state.extend(guesses)
        target_probs: torch.Tensor = standardize(target_state.logits()[-(num_draft + 1) :], temperature)
        stats.iterations += 1
        stats.drafted += num_draft

        # Keep guesses until the first rejection, which is replaced by a token from the adjusted distribution.
        new_tokens: list[int] = []
        num_accepted: int = 0
        for i, guess in enumerate(guesses):
            token, accepted = speculative_sample(target_probs[i], draft_probs[i], guess)
            new_tokens.append(token)
            if not accepted:
                stats.rejected += 1
                break
            num_accepted += 1
        else:
            # All guesses were accepted, so we get an extra token from the target for free.
            new_tokens.append(int(torch.multinomial(target_probs[num_draft], 1).item()))
        stats.accepted += num_accepted

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
            state.extend(new_tokens[num_accepted:])

    return generated, stats
