"""
This module contains batched decoding: the autoregressive baseline and speculative decoding (Algorithm 1 of the paper),
for a batch of sequences decoded together. It leaves the batch size 1 code of `decoding.py` untouched.

The paper only evaluates a batch size of 1, so the batched algorithm is not in the paper. This is the simplest one that
keeps the output distribution of every sequence exact, and it runs in lockstep:

- All sequences of the batch have the same length at all times, so the KV caches stay rectangular and are rolled back
  with a single `crop`.
- In every iteration the draft model guesses gamma tokens for every sequence, and the target model verifies all of them
  in one pass. Sequence b accepts n_b guesses, as in Algorithm 1 (its guesses up to the first rejected one).
- The batch advances by n = min(n_b) over the sequences that are not finished, plus one token for each sequence. A
  sequence with n_b == n takes the token resampled from the adjusted distribution norm(max(0, p - q)), as in Algorithm
  1. A sequence with n_b > n takes its guess number n, which it has accepted. That is the accepting branch of the same
  speculative sampling step, so the token still has the distribution p. The guesses past n are discarded.

Why the output stays exact: for a sequence, the block of tokens of an iteration is what Algorithm 1 would produce,
cut off after a position that depends only on the other sequences (whose randomness is independent of it). Each token
of the block is drawn from p given the tokens before it, and the next iteration draws fresh tokens given the whole
prefix, so cutting the block short changes nothing about the distribution. The cost is efficiency: the sequences wait
for the one that rejected first, so a batched iteration yields fewer tokens than Equation (1) says for a single
sequence. If every sequence accepted each guess with the same probability alpha independently, the batch of B
sequences would advance by Equation (1) at alpha^B, because P(min n_b >= k) = alpha^(k B). `BatchedDecodingStats`
measures the loss.

Sequences that reach the end-of-sequence token are finished, but they stay in the batch (their later tokens are
ignored), and no longer limit n. Both decoders here do that, so they are compared fairly. Only encoder-decoder models
are supported, with a dynamic KV cache.
"""

from dataclasses import dataclass
from typing import Self

import torch
from transformers import DynamicCache, EncoderDecoderCache, PreTrainedModel

from .sampling import standardize


@dataclass
class BatchedDecodingStats:
    """
    The counters of a batched speculative decoding run. They add up across batches with `+=`.
    """

    iterations: int = 0  # The number of target model calls, one per iteration of the whole batch.
    row_iterations: int = 0  # The number of (sequence, iteration) pairs where the sequence wasn't finished yet.
    drafted: int = 0  # The guesses proposed to the sequences that weren't finished.
    accepted: int = 0  # The guesses accepted, counting each sequence up to its first rejection.
    rejected: int = 0  # The number of times a sequence reached a rejection, at most once per iteration.
    positions: int = 0  # The sum over the iterations of the tokens every sequence advanced by (n + 1).

    def __iadd__(self, other: Self) -> Self:
        self.iterations += other.iterations
        self.row_iterations += other.row_iterations
        self.drafted += other.drafted
        self.accepted += other.accepted
        self.rejected += other.rejected
        self.positions += other.positions
        return self

    @property
    def acceptance_rate(self) -> float:
        """
        The empirical alpha: accepted guesses over the guesses whose acceptance was decided, which are the accepted ones
        and the one rejected guess of a sequence that reached a rejection. It is computed per sequence, before the
        lockstep minimum, so it doesn't depend on the batch size the way the tokens per iteration do.
        """

        decided: int = self.accepted + self.rejected
        return self.accepted / decided if decided else 0.0

    @property
    def tokens_per_iteration(self) -> float:
        """
        The tokens every unfinished sequence gains per target model call. This is the batched counterpart of Equation
        (1), and is at most that, because the batch advances by the minimum over the sequences.
        """

        return self.positions / self.iterations if self.iterations else 0.0


def eos_token_ids(model: PreTrainedModel) -> set[int]:
    """
    The ids of the tokens that end a sequence, from the generation config of the model.
    """

    eos = model.config.eos_token_id
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)


class BatchedDecoderState:
    """
    The state of a batch of sequences being decoded by one encoder-decoder model: the tokens, and the KV cache with a
    count of how many positions it holds. All the sequences have the same length.

    The decoder always starts from the decoder start token. Padded inputs are handled through the attention mask.
    """

    def __init__(self, model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        """
        Args:
            model: An encoder-decoder model in eval mode.
            input_ids: The encoder inputs, shape (batch, input_len), padded on the right.
            attention_mask: 1 for the input tokens and 0 for the padding, shape (batch, input_len).

        Raises:
            ValueError: If the model isn't an encoder-decoder model, or the shapes disagree.
        """

        if not model.config.is_encoder_decoder:
            raise ValueError("Batched decoding is only implemented for encoder-decoder models")
        if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both have the shape (batch, input_len)")

        self.model: PreTrainedModel = model
        self.attention_mask: torch.Tensor = attention_mask
        self.encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        self.cache: EncoderDecoderCache = EncoderDecoderCache(
            DynamicCache(config=model.config), DynamicCache(config=model.config)
        )

        start: int = model.config.decoder_start_token_id
        self.buffer: torch.Tensor = torch.full(
            (input_ids.shape[0], 64), start, dtype=torch.long, device=input_ids.device
        )
        self.length: int = 1  # The start token.
        self.cached: int = 0  # The positions the KV cache holds.

    def __len__(self) -> int:
        return self.length

    def extend(self, tokens: torch.Tensor) -> None:
        """
        Append tokens to every sequence.

        Args:
            tokens: The new tokens, shape (batch, count). The count may be 0.
        """

        count: int = tokens.shape[1]
        if self.length + count > self.buffer.shape[1]:
            grown: torch.Tensor = self.buffer.new_zeros((self.buffer.shape[0], 2 * (self.length + count)))
            grown[:, : self.length] = self.buffer[:, : self.length]
            self.buffer = grown
        self.buffer[:, self.length : self.length + count] = tokens
        self.length += count

    def logits(self) -> torch.Tensor:
        """
        Run the decoder on the positions that aren't in the KV cache yet.

        Returns:
            The logits of those positions, shape (batch, uncached, vocab_size). The last row predicts the token that
            follows the whole sequence.
        """

        if self.cached >= self.length:
            raise ValueError("There are no new positions to process")

        output = self.model(
            encoder_outputs=self.encoder_outputs,
            attention_mask=self.attention_mask,
            decoder_input_ids=self.buffer[:, self.cached : self.length],
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cached = self.length
        return output.logits

    def truncate(self, length: int) -> None:
        """
        Roll every sequence back to `length` positions.
        """

        if self.cached > length:
            self.cache.crop(length - self.cached)  # Cache.crop takes a negative number of tokens to remove.
            self.cached = length
        self.length = length


def batched_speculative_sample(
    p: torch.Tensor, q: torch.Tensor, guesses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Speculative sampling for all the guesses of a batch at once (Algorithm 1 of the paper, for every sequence).

    Per sequence, this is `speculative_sample` of `sampling.py`: the number of accepted guesses n_b, and the token that
    follows them, sampled from the adjusted distribution norm(max(0, p - q)) at position n_b, or from the last row of
    `p` if every guess was accepted. Nothing leaves the device.

    Args:
        p: The target distributions, shape (batch, num_guesses + 1, vocab_size).
        q: The draft distributions, shape (batch, num_guesses, vocab_size).
        guesses: The tokens sampled from `q`, shape (batch, num_guesses).

    Returns:
        The number of accepted guesses of every sequence and the token that follows them, both of shape (batch,).

    Raises:
        ValueError: If the shapes disagree.
    """

    batch_size, num_guesses = guesses.shape
    vocab_size: int = p.shape[-1]
    if p.shape != (batch_size, num_guesses + 1, vocab_size) or q.shape != (batch_size, num_guesses, vocab_size):
        raise ValueError(f"Inconsistent shapes: p {tuple(p.shape)}, q {tuple(q.shape)}, guesses {tuple(guesses.shape)}")

    # The comparisons are done in float32 even for bfloat16 models, like `speculative_sample` does.
    p = p.float()
    q = q.float()

    # Accept guess i with probability min(1, p(x) / q(x)). The comparison is strict, so that a draw of exactly 0 can't
    # accept a token the target gives probability 0.
    index: torch.Tensor = guesses.unsqueeze(-1)
    p_guess: torch.Tensor = p[:, :num_guesses].gather(2, index).squeeze(-1)
    q_guess: torch.Tensor = q.gather(2, index).squeeze(-1)
    r: torch.Tensor = torch.rand((batch_size, num_guesses), device=p.device)
    accepted: torch.Tensor = r * q_guess < p_guess
    num_accepted: torch.Tensor = accepted.long().cumprod(dim=1).sum(dim=1)  # The leading run of acceptances.

    # The distribution at the position of the first rejection of each sequence. The zero row pads `q` so that when every
    # guess is accepted the adjusted distribution is just the last row of `p`.
    position: torch.Tensor = num_accepted.view(batch_size, 1, 1).expand(batch_size, 1, vocab_size)
    p_next: torch.Tensor = p.gather(1, position).squeeze(1)
    q_next: torch.Tensor = (
        torch.cat([q, q.new_zeros((batch_size, 1, vocab_size))], dim=1).gather(1, position).squeeze(1)
    )
    residual: torch.Tensor = torch.clamp(p_next - q_next, min=0)
    # The residual is only empty through floating point error when p == q; p is then the right distribution.
    distribution: torch.Tensor = torch.where(residual.sum(dim=-1, keepdim=True) > 0, residual, p_next)
    token: torch.Tensor = torch.multinomial(distribution / distribution.sum(dim=-1, keepdim=True), 1).squeeze(1)

    return num_accepted, token


def _append_tokens(outputs: list[list[int]], finished: list[bool], new_tokens: list[list[int]], eos: set[int]) -> None:
    """
    Add the new tokens of every sequence that isn't finished, up to and including its end-of-sequence token.
    """

    for row, tokens in enumerate(new_tokens):
        if finished[row]:
            continue
        for token in tokens:
            outputs[row].append(token)
            if token in eos:
                finished[row] = True
                break


@torch.inference_mode()
def batched_autoregressive_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    max_new_tokens: int,
) -> list[list[int]]:
    """
    Decode a batch of inputs with the target model alone, one token per call: the baseline.

    The batch runs until every sequence is finished or `max_new_tokens` is reached. Finished sequences stay in the
    batch.

    Args:
        model: The encoder-decoder model.
        input_ids: The encoder inputs, shape (batch, input_len), padded on the right.
        attention_mask: 1 for the input tokens and 0 for the padding, shape (batch, input_len).
        temperature: The sampling temperature. 0 means greedy decoding.
        max_new_tokens: The generation limit.

    Returns:
        The generated tokens of every sequence, without the start token and including the end-of-sequence token.
    """

    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")

    eos: set[int] = eos_token_ids(model)
    state: BatchedDecoderState = BatchedDecoderState(model, input_ids, attention_mask)
    outputs: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
    finished: list[bool] = [False] * input_ids.shape[0]

    for _ in range(max_new_tokens):
        probabilities: torch.Tensor = standardize(state.logits()[:, -1, :], temperature)
        token: torch.Tensor = torch.multinomial(probabilities, 1)
        state.extend(token)

        _append_tokens(outputs, finished, token.tolist(), eos)
        if all(finished):
            break

    return outputs


@torch.inference_mode()
def batched_speculative_decode(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    gamma: int,
    max_new_tokens: int,
) -> tuple[list[list[int]], BatchedDecodingStats]:
    """
    Decode a batch of inputs with batched speculative decoding in lockstep (see the module docstring).

    Each iteration the draft model guesses `gamma` tokens for every sequence, the target model scores all of them in
    one call, and the batch advances by the smallest number of accepted guesses over the unfinished sequences, plus one
    token. The output of every sequence has the distribution of the target model alone.

    Args:
        target: The target model Mp, an encoder-decoder model.
        draft: The draft model Mq, an encoder-decoder model with the same vocabulary.
        input_ids: The encoder inputs, shape (batch, input_len), padded on the right.
        attention_mask: 1 for the input tokens and 0 for the padding, shape (batch, input_len).
        temperature: The sampling temperature for both models. 0 means greedy decoding.
        gamma: The number of guesses per iteration.
        max_new_tokens: The generation limit.

    Returns:
        The generated tokens of every sequence, without the start token and including the end-of-sequence token, and
        the counters of the run.
    """

    if gamma < 1:
        raise ValueError(f"gamma must be at least 1, got {gamma}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")

    batch_size: int = input_ids.shape[0]
    device: torch.device = input_ids.device
    eos: set[int] = eos_token_ids(target)
    target_state: BatchedDecoderState = BatchedDecoderState(target, input_ids, attention_mask)
    draft_state: BatchedDecoderState = BatchedDecoderState(draft, input_ids, attention_mask)
    vocab_size: int = draft.config.vocab_size

    outputs: list[list[int]] = [[] for _ in range(batch_size)]
    finished: list[bool] = [False] * batch_size
    stats: BatchedDecodingStats = BatchedDecodingStats()
    generated: int = 0  # The tokens every sequence has gained, which is the same for all of them.

    while generated < max_new_tokens and not all(finished):
        verified: int = len(target_state)  # The positions both states agree on at the start of the iteration.
        # Leave room for the token that follows the guesses, so that the generation limit is never overshot.
        num_draft: int = min(gamma, max_new_tokens - generated - 1)

        # Sample the guesses from the draft model autoregressively, for the whole batch.
        guesses: list[torch.Tensor] = []
        draft_probabilities: torch.Tensor = torch.empty(
            (batch_size, num_draft, vocab_size), dtype=draft.dtype, device=device
        )
        for i in range(num_draft):
            probabilities: torch.Tensor = standardize(draft_state.logits()[:, -1, :], temperature)
            guess: torch.Tensor = torch.multinomial(probabilities, 1)
            draft_state.extend(guess)
            guesses.append(guess)
            draft_probabilities[:, i] = probabilities
        guess_ids: torch.Tensor = (
            torch.cat(guesses, dim=1) if guesses else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        )

        # Run the target model once on the last token and all the guesses.
        target_state.extend(guess_ids)
        target_probabilities: torch.Tensor = standardize(target_state.logits()[:, -(num_draft + 1) :, :], temperature)

        num_accepted, resampled = batched_speculative_sample(target_probabilities, draft_probabilities, guess_ids)

        # One transfer to the host per iteration: the guesses, n_b and the resampled token of every sequence.
        rows: list[list[int]] = torch.cat(
            [guess_ids, num_accepted.view(batch_size, 1), resampled.view(batch_size, 1)], dim=1
        ).tolist()
        accepted_counts: list[int] = [row[num_draft] for row in rows]
        active: list[int] = [row for row in range(batch_size) if not finished[row]]

        # The batch advances by the fewest accepted guesses among the sequences that aren't finished.
        advance: int = min(accepted_counts[row] for row in active)
        new_tokens: list[list[int]] = []
        for row, values in enumerate(rows):
            if accepted_counts[row] > advance:
                last: int = values[advance]  # Its guess number `advance` was accepted, so it is kept.
            else:
                last = values[num_draft + 1]  # Rejected here, so the token comes from the adjusted distribution.
            new_tokens.append(values[:advance] + [last])

        stats.iterations += 1
        stats.row_iterations += len(active)
        stats.drafted += num_draft * len(active)
        stats.accepted += sum(accepted_counts[row] for row in active)
        stats.rejected += sum(accepted_counts[row] < num_draft for row in active)
        stats.positions += advance + 1

        _append_tokens(outputs, finished, new_tokens, eos)
        generated += advance + 1
        if generated >= max_new_tokens or all(finished):
            break

        # Roll both models back to the accepted guesses, then add the token that follows them.
        last_tokens: torch.Tensor = torch.tensor(
            [[tokens[-1]] for tokens in new_tokens], dtype=torch.long, device=device
        )
        for state in (target_state, draft_state):
            state.truncate(verified + advance)
            state.extend(last_tokens)

    return outputs, stats
