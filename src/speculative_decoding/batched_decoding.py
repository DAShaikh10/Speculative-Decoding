"""
This module contains batched decoding: the autoregressive baseline and speculative decoding, for a batch of sequences decoded together.
"""

from dataclasses import dataclass, field
from typing import Self

import torch
from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel, StaticCache

from .batched_runner import BatchedModelRunner
from .decoding import eos_token_ids, static_capacity
from .sampling import standardize

__all__ = [
    "BatchedDecoderState",
    "BatchedDecodingStats",
    "batched_autoregressive_decode",
    "batched_speculative_decode",
    "batched_speculative_sample",
    "eos_token_ids",
]


@dataclass
class BatchedDecodingStats:

    iterations: int = 0
    row_iterations: int = 0
    drafted: int = 0
    accepted: int = 0
    rejected: int = 0
    positions: int = 0
    beta_sum: float = 0.0
    beta_count: int = 0
    example_counts: list[list[int]] = field(default_factory=list)

    def __iadd__(self, other: Self) -> Self:
        self.iterations += other.iterations
        self.row_iterations += other.row_iterations
        self.drafted += other.drafted
        self.accepted += other.accepted
        self.rejected += other.rejected
        self.positions += other.positions
        self.beta_sum += other.beta_sum
        self.beta_count += other.beta_count
        self.example_counts.extend(other.example_counts)
        return self

    @property
    def acceptance_rate(self) -> float:

        decided: int = self.accepted + self.rejected
        return self.accepted / decided if decided else 0.0

    @property
    def analytic_alpha(self) -> float:

        return self.beta_sum / self.beta_count if self.beta_count else 0.0

    @property
    def tokens_per_iteration(self) -> float:

        return self.positions / self.iterations if self.iterations else 0.0


class BatchedDecoderState:

    def __init__(
        self,
        model: PreTrainedModel | BatchedModelRunner,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        capacity: int | None = None,
        vocab_size: int | None = None,
    ) -> None:

        self.runner: BatchedModelRunner | None = model if isinstance(model, BatchedModelRunner) else None
        if self.runner is not None:
            if capacity is not None:
                raise ValueError("A runner has its own capacity")
            model, capacity = self.runner.model, self.runner.capacity

        if not model.config.is_encoder_decoder:
            raise ValueError("Batched decoding is only implemented for encoder-decoder models")
        if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both have the shape (batch, input_len)")

        self.model: PreTrainedModel = model
        self.capacity: int | None = capacity
        self.vocab_size: int = vocab_size or model.config.vocab_size
        self.encoder_outputs = None
        self.slot = None
        self.cache: Cache
        if self.runner is not None:
            self.slot = self.runner.prepare(input_ids, attention_mask)
            self.cache = self.slot.cache
            self.attention_mask: torch.Tensor = self.slot.mask
        else:
            self.attention_mask = attention_mask
            self.encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            self.cache = self._new_cache(model, capacity, input_ids.shape[1])

        start: int = model.config.decoder_start_token_id
        self.buffer: torch.Tensor = torch.full(
            (input_ids.shape[0], 64), start, dtype=torch.long, device=input_ids.device
        )
        self.length: int = 1
        self.cached: int = 0

    @staticmethod
    def _new_cache(model: PreTrainedModel, capacity: int | None, encoder_length: int) -> EncoderDecoderCache:

        if capacity is None:
            return EncoderDecoderCache(DynamicCache(config=model.config), DynamicCache(config=model.config))

        return EncoderDecoderCache(
            StaticCache(config=model.config, max_cache_len=capacity),
            StaticCache(config=model.config, max_cache_len=encoder_length),
        )

    def _rewind(self, length: int) -> None:

        for layer in self.cache.self_attention_cache.layers:
            layer.cumulative_length.fill_(length)

    def __len__(self) -> int:
        return self.length

    def extend(self, tokens: torch.Tensor) -> None:

        count: int = tokens.shape[1]
        if self.length + count > self.buffer.shape[1]:
            grown: torch.Tensor = self.buffer.new_zeros((self.buffer.shape[0], 2 * (self.length + count)))
            grown[:, : self.length] = self.buffer[:, : self.length]
            self.buffer = grown
        self.buffer[:, self.length : self.length + count] = tokens
        self.length += count

    def logits(self) -> torch.Tensor:

        if self.cached >= self.length:
            raise ValueError("There are no new positions to process")
        if self.capacity is not None and self.length > self.capacity:
            raise ValueError(f"The sequence has {self.length} positions, more than the static cache capacity")

        if self.runner is not None:
            logits: torch.Tensor = self.runner.forward(
                self.slot, self.buffer[:, self.cached : self.length], self.cached
            )
            self.cached = self.length
            return logits[..., : self.vocab_size]

        output = self.model(
            encoder_outputs=self.encoder_outputs,
            attention_mask=self.attention_mask,
            decoder_input_ids=self.buffer[:, self.cached : self.length],
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cached = self.length
        return output.logits[..., : self.vocab_size]

    def sample_guesses(self, num_guesses: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:

        if self.runner is None:
            raise ValueError("The drafting loop needs a runner")
        guesses, probs = self.runner.draft_loop(
            self.slot,
            self.buffer[:, self.cached : self.length],
            self.cached,
            num_guesses,
            temperature,
            self.vocab_size,
        )
        self.cached = self.length + num_guesses - 1
        self.extend(guesses)
        return guesses, probs

    def truncate(self, length: int) -> None:

        if self.cached > length:
            if self.runner is None:
                if self.capacity is None:
                    self.cache.crop(length - self.cached)
                else:
                    self._rewind(length)
            self.cached = length
        self.length = length


def batched_speculative_sample(
    p: torch.Tensor, q: torch.Tensor, guesses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_size, num_guesses = guesses.shape
    vocab_size: int = p.shape[-1]
    if p.shape != (batch_size, num_guesses + 1, vocab_size) or q.shape != (batch_size, num_guesses, vocab_size):
        raise ValueError(f"Inconsistent shapes: p {tuple(p.shape)}, q {tuple(q.shape)}, guesses {tuple(guesses.shape)}")

    p = p.float()
    q = q.float()

    index: torch.Tensor = guesses.unsqueeze(-1)
    p_guess: torch.Tensor = p[:, :num_guesses].gather(2, index).squeeze(-1)
    q_guess: torch.Tensor = q.gather(2, index).squeeze(-1)
    r: torch.Tensor = torch.rand((batch_size, num_guesses), device=p.device)
    accepted: torch.Tensor = r * q_guess < p_guess
    num_accepted: torch.Tensor = accepted.long().cumprod(dim=1).sum(dim=1)

    position: torch.Tensor = num_accepted.view(batch_size, 1, 1).expand(batch_size, 1, vocab_size)
    p_next: torch.Tensor = p.gather(1, position).squeeze(1)
    q_next: torch.Tensor = (
        torch.cat([q, q.new_zeros((batch_size, 1, vocab_size))], dim=1).gather(1, position).squeeze(1)
    )
    residual: torch.Tensor = torch.clamp(p_next - q_next, min=0)
    distribution: torch.Tensor = torch.where(residual.sum(dim=-1, keepdim=True) > 0, residual, p_next)
    token: torch.Tensor = torch.multinomial(distribution / distribution.sum(dim=-1, keepdim=True), 1).squeeze(1)

    return num_accepted, token


def _append_tokens(
    outputs: list[list[int]],
    finished: list[bool],
    new_tokens: list[list[int]],
    eos: set[int],
    limit: int | None = None,
) -> None:

    for row, tokens in enumerate(new_tokens):
        if finished[row]:
            continue
        for token in tokens:
            if limit is not None and len(outputs[row]) >= limit:
                break
            outputs[row].append(token)
            if token in eos:
                finished[row] = True
                break


@torch.inference_mode()
def batched_autoregressive_decode(
    model: PreTrainedModel | BatchedModelRunner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    max_new_tokens: int,
    static_cache: bool = False,
    vocab_size: int | None = None,
) -> list[list[int]]:

    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")

    eos: set[int] = eos_token_ids(model)
    use_static: bool = static_cache and not isinstance(model, BatchedModelRunner)
    capacity: int | None = static_capacity(model, input_ids, max_new_tokens) if use_static else None
    state: BatchedDecoderState = BatchedDecoderState(model, input_ids, attention_mask, capacity, vocab_size)
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
    target: PreTrainedModel | BatchedModelRunner,
    draft: PreTrainedModel | BatchedModelRunner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    gamma: int,
    max_new_tokens: int,
    static_cache: bool = False,
    vocab_size: int | None = None,
    fuse_draft: bool = False,
) -> tuple[list[list[int]], BatchedDecodingStats]:

    if gamma < 1:
        raise ValueError(f"gamma must be at least 1, got {gamma}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
    if isinstance(target, BatchedModelRunner) != isinstance(draft, BatchedModelRunner):
        raise ValueError("Either both models or neither must be runners")
    fixed_steps: bool = isinstance(target, BatchedModelRunner)
    if fuse_draft and not fixed_steps:
        raise ValueError("Fusing the drafting loop needs runners")

    batch_size: int = input_ids.shape[0]
    device: torch.device = input_ids.device
    eos: set[int] = eos_token_ids(target)
    vocab_size = vocab_size or min(target.config.vocab_size, draft.config.vocab_size)
    extra: int = max_new_tokens + gamma + 1
    target_state: BatchedDecoderState = BatchedDecoderState(
        target,
        input_ids,
        attention_mask,
        static_capacity(target, input_ids, extra) if static_cache and not fixed_steps else None,
        vocab_size,
    )
    draft_state: BatchedDecoderState = BatchedDecoderState(
        draft,
        input_ids,
        attention_mask,
        static_capacity(draft, input_ids, extra) if static_cache and not fixed_steps else None,
        vocab_size,
    )

    outputs: list[list[int]] = [[] for _ in range(batch_size)]
    finished: list[bool] = [False] * batch_size
    stats: BatchedDecodingStats = BatchedDecodingStats()
    generated: int = 0
    beta_sum: torch.Tensor = torch.zeros((), dtype=torch.float32, device=device)
    beta_count: torch.Tensor = torch.zeros((), dtype=torch.long, device=device)
    example_counts: list[list[int]] = [[0, 0] for _ in range(batch_size)]

    while generated < max_new_tokens and not all(finished):
        verified: int = len(target_state)
        num_draft: int = gamma if fixed_steps else min(gamma, max_new_tokens - generated - 1)

        if fuse_draft:
            guess_ids, draft_probabilities = draft_state.sample_guesses(num_draft, temperature)
        else:
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

        target_state.extend(guess_ids)
        target_probabilities: torch.Tensor = standardize(target_state.logits()[:, -(num_draft + 1) :, :], temperature)

        num_accepted, resampled = batched_speculative_sample(target_probabilities, draft_probabilities, guess_ids)

        rows: list[list[int]] = torch.cat(
            [guess_ids, num_accepted.view(batch_size, 1), resampled.view(batch_size, 1)], dim=1
        ).tolist()
        accepted_counts: list[int] = [row[num_draft] for row in rows]
        active: list[int] = [row for row in range(batch_size) if not finished[row]]

        advance: int = min(accepted_counts[row] for row in active)

        unfinished: torch.Tensor = torch.tensor([not done for done in finished], device=device)
        counted: torch.Tensor = (
            torch.arange(num_draft, device=device).unsqueeze(0) <= num_accepted.clamp(max=advance).unsqueeze(1)
        ) & unfinished.unsqueeze(1)
        beta: torch.Tensor = torch.minimum(target_probabilities[:, :num_draft], draft_probabilities).sum(
            dim=-1, dtype=torch.float32
        )
        beta_sum += (beta * counted).sum()
        beta_count += counted.sum()

        new_tokens: list[list[int]] = []
        for row, values in enumerate(rows):
            if accepted_counts[row] > advance:
                last: int = values[advance]
            else:
                last = values[num_draft + 1]
            new_tokens.append(values[:advance] + [last])

        stats.iterations += 1
        stats.row_iterations += len(active)
        stats.drafted += num_draft * len(active)
        for row in active:
            if accepted_counts[row] > advance:
                example_counts[row][0] += advance + 1
            else:
                example_counts[row][0] += advance
                example_counts[row][1] += advance < num_draft
        stats.positions += advance + 1

        _append_tokens(outputs, finished, new_tokens, eos, max_new_tokens)
        generated += advance + 1
        if generated >= max_new_tokens or all(finished):
            break

        last_tokens: torch.Tensor = torch.tensor(
            [[tokens[-1]] for tokens in new_tokens], dtype=torch.long, device=device
        )
        for state in (target_state, draft_state):
            state.truncate(verified + advance)
            state.extend(last_tokens)

    stats.beta_sum, stats.beta_count = float(beta_sum.item()), int(beta_count.item())
    stats.example_counts = example_counts
    stats.accepted = sum(counts[0] for counts in example_counts)
    stats.rejected = sum(counts[1] for counts in example_counts)
    return outputs, stats
