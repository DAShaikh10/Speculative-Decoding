"""
Tests for the decoding algorithms. At temperature 0 our output must equal Hugging Face's greedy decoding.
"""

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

from speculative_decoding.decoding import DecoderState, autoregressive_decode, speculative_decode

MAX_NEW_TOKENS: int = 20


def test_autoregressive_decode_matches_huggingface_greedy_encoder_decoder() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()
    input_ids: torch.Tensor = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids

    ours: list[int] = autoregressive_decode(model, input_ids, temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)

    reference: torch.Tensor = model.generate(input_ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    # Hugging Face includes the decoder start token at position 0.
    assert ours == reference[0, 1:].tolist()


def test_autoregressive_decode_matches_huggingface_greedy_decoder_only() -> None:
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").eval()
    input_ids: torch.Tensor = tokenizer("The house is", return_tensors="pt").input_ids

    ours: list[int] = autoregressive_decode(model, input_ids, temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)

    reference: torch.Tensor = model.generate(input_ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    assert ours == reference[0, input_ids.shape[1] :].tolist()


def test_autoregressive_decode_respects_max_new_tokens() -> None:
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").eval()
    input_ids: torch.Tensor = torch.tensor([[1, 2, 3]])
    assert len(autoregressive_decode(model, input_ids, temperature=1.0, max_new_tokens=5)) <= 5


def random_copy(model_name: str, seed: int):
    """A randomly initialised model with the same architecture and vocabulary, which disagrees with the original."""
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_name)
    auto = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    return auto.from_config(config).eval()


@pytest.mark.parametrize("model_name", ["google-t5/t5-small", "sshleifer/tiny-gpt2"])
@pytest.mark.parametrize("gamma", [1, 4])
def test_speculative_decode_zero_temperature_equals_baseline(model_name: str, gamma: int) -> None:
    """At temperature 0 the output must be token-identical to the baseline, even with a bad draft model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    auto = AutoModelForSeq2SeqLM if "t5" in model_name else AutoModelForCausalLM
    target = auto.from_pretrained(model_name).eval()
    draft = random_copy(model_name, seed=0)
    input_ids: torch.Tensor = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids

    # The reference is the baseline without the KV cache, so that a cache bug cannot cancel out.
    expected: list[int] = autoregressive_decode(
        target, input_ids, temperature=0.0, max_new_tokens=MAX_NEW_TOKENS, use_cache=False
    )
    actual, stats = speculative_decode(
        target, draft, input_ids, temperature=0.0, gamma=gamma, max_new_tokens=MAX_NEW_TOKENS
    )

    assert actual == expected
    assert stats.accepted < stats.drafted  # The random draft must have been rejected sometimes.


def test_speculative_decode_identical_draft_accepts_everything() -> None:
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").eval()
    input_ids: torch.Tensor = torch.tensor([[1, 2, 3]])
    tokens, stats = speculative_decode(model, model, input_ids, temperature=1.0, gamma=4, max_new_tokens=16)
    assert stats.accepted == stats.drafted
    assert len(tokens) <= 16


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_speculative_decode_respects_max_new_tokens(temperature: float) -> None:
    target = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").eval()
    draft = random_copy("sshleifer/tiny-gpt2", seed=1)
    input_ids: torch.Tensor = torch.tensor([[1, 2, 3]])
    for max_new_tokens in [1, 2, 7]:
        tokens, _ = speculative_decode(target, draft, input_ids, temperature, gamma=4, max_new_tokens=max_new_tokens)
        assert len(tokens) <= max_new_tokens


def tiny_gpt(seed: int) -> GPT2LMHeadModel:
    """A tiny randomly initialised model over a 6 token vocabulary, with a non-uniform output distribution."""
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=6, n_positions=16, n_embd=16, n_layer=1, n_head=2, initializer_range=0.8, bos_token_id=None, eos_token_id=None
    )
    return GPT2LMHeadModel(config).eval()


@torch.inference_mode()
def exact_two_token_distribution(model: GPT2LMHeadModel, prompt: torch.Tensor) -> torch.Tensor:
    """The exact joint distribution p(x1, x2 | prompt), shape (vocab, vocab)."""
    first: torch.Tensor = torch.softmax(model(input_ids=prompt).logits[0, -1], dim=-1)
    rows: list[torch.Tensor] = []
    for x1 in range(first.shape[0]):
        extended: torch.Tensor = torch.cat([prompt, torch.tensor([[x1]])], dim=1)
        rows.append(first[x1] * torch.softmax(model(input_ids=extended).logits[0, -1], dim=-1))
    return torch.stack(rows)


def test_speculative_decode_samples_from_target_distribution() -> None:
    """At temperature 1 the sequences must follow the target's distribution, not the draft's (Appendix A.1)."""
    target: GPT2LMHeadModel = tiny_gpt(seed=0)
    draft: GPT2LMHeadModel = tiny_gpt(seed=1)
    prompt: torch.Tensor = torch.tensor([[1, 2]])
    num_samples: int = 8000

    expected: torch.Tensor = exact_two_token_distribution(target, prompt)
    draft_distribution: torch.Tensor = exact_two_token_distribution(draft, prompt)
    # The draft must differ from the target, otherwise this test proves nothing.
    assert 0.5 * (expected - draft_distribution).abs().sum() > 0.1

    torch.manual_seed(0)
    counts: torch.Tensor = torch.zeros_like(expected)
    for _ in range(num_samples):
        tokens, _ = speculative_decode(target, draft, prompt, temperature=1.0, gamma=3, max_new_tokens=2)
        counts[tokens[0], tokens[1]] += 1

    total_variation: float = (0.5 * (counts / num_samples - expected).abs().sum()).item()
    assert total_variation < 0.05


@pytest.mark.parametrize("model_name", ["google-t5/t5-small", "sshleifer/tiny-gpt2"])
def test_decoder_state_logits_match_uncached_after_rollback(model_name: str) -> None:
    """After rolling the KV cache back and appending tokens, the logits must equal those of a fresh uncached run."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    auto = AutoModelForSeq2SeqLM if "t5" in model_name else AutoModelForCausalLM
    model = auto.from_pretrained(model_name).eval()
    input_ids: torch.Tensor = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids

    with torch.inference_mode():
        cached: DecoderState = DecoderState(model, input_ids, use_cache=True)
        cached.logits()
        cached.extend([5, 6, 7, 8])
        cached.logits()
        cached.truncate(len(cached) - 3)  # Reject the last three tokens.
        cached.extend([9, 10])
        actual: torch.Tensor = cached.logits()

        reference: DecoderState = DecoderState(model, input_ids, use_cache=False)
        reference.tokens = list(cached.tokens)
        expected: torch.Tensor = reference.logits()[-actual.shape[0] :]

    assert torch.allclose(actual, expected, atol=1e-3)
    assert cached.cache is not None and cached.cache.get_seq_length() == len(cached)


@pytest.mark.parametrize("model_name", ["google-t5/t5-small", "sshleifer/tiny-gpt2"])
def test_speculative_decode_cache_matches_no_cache_at_zero_temperature(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    auto = AutoModelForSeq2SeqLM if "t5" in model_name else AutoModelForCausalLM
    target = auto.from_pretrained(model_name).eval()
    draft = random_copy(model_name, seed=2)
    input_ids: torch.Tensor = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids

    cached, cached_stats = speculative_decode(target, draft, input_ids, 0.0, gamma=4, max_new_tokens=MAX_NEW_TOKENS)
    uncached, uncached_stats = speculative_decode(
        target, draft, input_ids, 0.0, gamma=4, max_new_tokens=MAX_NEW_TOKENS, use_cache=False
    )

    assert cached == uncached
    assert cached_stats == uncached_stats
