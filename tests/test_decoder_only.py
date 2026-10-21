"""
Tests for the decoder-only path: a Qwen2 style model with a prefilled prompt, a padded output layer, several EOS tokens
and the decoder-only runner. The models are tiny and randomly initialised, so nothing is downloaded but a tokenizer.
"""

import pytest
import torch
from torch._dynamo.utils import counters
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from speculative_decoding.decoding import DecoderState, autoregressive_decode, eos_token_ids, speculative_decode
from speculative_decoding.evaluation import encode
from speculative_decoding.runner import DecoderOnlyRunner, ModelRunner

MAX_NEW_TOKENS: int = 12
GAMMA: int = 3
VOCAB: int = 96
PROMPT: list[list[int]] = [[5, 17, 3, 42, 8, 11, 60, 2]]


def tiny_qwen(seed: int, vocab_size: int = VOCAB, layers: int = 2) -> Qwen2ForCausalLM:
    """A randomly initialised Qwen2 model. Different seeds give models that disagree."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=VOCAB - 1,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()


@pytest.fixture(scope="module")
def target() -> Qwen2ForCausalLM:
    return tiny_qwen(seed=0, layers=3)


@pytest.fixture(scope="module")
def draft() -> Qwen2ForCausalLM:
    return tiny_qwen(seed=1)


def prompt(length: int = 8) -> torch.Tensor:
    return torch.tensor([(PROMPT[0] * 4)[:length]])


@pytest.mark.parametrize("length", [1, 2, 8])
def test_autoregressive_decode_matches_huggingface_greedy(target, length: int) -> None:
    input_ids = prompt(length)
    ours = autoregressive_decode(target, input_ids, temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    reference = target.generate(input_ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=0)
    expected = reference[0, input_ids.shape[1] :].tolist()
    assert ours == expected[: len(ours)] and len(ours) in (len(expected), MAX_NEW_TOKENS)


@pytest.mark.parametrize("length", [1, 2, 8])
def test_prefill_leaves_the_last_prompt_token_uncached(target, length: int) -> None:
    state = DecoderState(target, prompt(length))
    assert state.cached == length - 1
    assert state.logits().shape == (1, VOCAB)  # Only the last token is left to process.


@pytest.mark.parametrize("static_cache", [False, True])
@pytest.mark.parametrize("length", [1, 8])
def test_speculative_decode_equals_baseline_at_zero_temperature(target, draft, static_cache: bool, length: int) -> None:
    input_ids = prompt(length)
    baseline = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS, static_cache=static_cache)
    tokens, stats = speculative_decode(target, draft, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS, static_cache=static_cache)
    assert tokens == baseline
    assert stats.iterations > 0


@pytest.mark.parametrize("compile_mode, backend", [(None, "inductor"), ("default", "eager")])
@pytest.mark.parametrize("length", [1, 8])
def test_decoder_only_runners_equal_baseline(target, draft, compile_mode, backend, length: int) -> None:
    input_ids = prompt(length)
    baseline = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS)

    capacity = length + MAX_NEW_TOKENS + GAMMA + 1
    target_runner = DecoderOnlyRunner(target, capacity, compile_mode, backend)
    draft_runner = DecoderOnlyRunner(draft, capacity, compile_mode, backend)
    target_runner.warmup(input_ids, [1, GAMMA + 1], repeats=1)
    draft_runner.warmup(input_ids, [1, 2], repeats=1)

    assert autoregressive_decode(target_runner, input_ids, 0.0, MAX_NEW_TOKENS) == baseline
    tokens, _ = speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)
    assert tokens == baseline


def test_models_with_different_head_sizes_compile_one_after_the_other() -> None:
    """The draft and the target differ in constants such as the attention scale, which dynamo must not make symbolic."""

    def model(hidden_size: int, heads: int, eps: float) -> Qwen2ForCausalLM:
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=VOCAB,
            hidden_size=hidden_size,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=heads,
            num_key_value_heads=2,
            rms_norm_eps=eps,
            eos_token_id=VOCAB - 1,
        )
        return Qwen2ForCausalLM(config).eval()

    small, large = model(64, 4, 1e-6), model(128, 2, 1e-5)  # Head sizes 16 and 64.
    capacity = 8 + MAX_NEW_TOKENS + GAMMA + 1
    small_runner = DecoderOnlyRunner(small, capacity, "default", "eager")
    large_runner = DecoderOnlyRunner(large, capacity, "default", "eager")
    small_runner.warmup(prompt(), [1, 2], repeats=1)
    large_runner.warmup(prompt(), [1, GAMMA + 1], repeats=1)

    expected = autoregressive_decode(large, prompt(), 0.0, MAX_NEW_TOKENS)
    tokens, _ = speculative_decode(large_runner, small_runner, prompt(), 0.0, GAMMA, MAX_NEW_TOKENS)
    assert tokens == expected


def test_one_decoder_only_runner_serves_inputs_of_different_lengths(target, draft) -> None:
    capacity = 8 + MAX_NEW_TOKENS + GAMMA + 1
    target_runner = DecoderOnlyRunner(target, capacity)
    draft_runner = DecoderOnlyRunner(draft, capacity)
    for length in (8, 3, 5, 1):
        input_ids = prompt(length)
        expected = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS)
        tokens, _ = speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)
        assert tokens == expected


def test_no_step_is_compiled_after_the_warmup(target, draft) -> None:
    input_ids = prompt()
    capacity = 8 + MAX_NEW_TOKENS + GAMMA + 1
    target_runner = DecoderOnlyRunner(target, capacity, "default", "eager")
    draft_runner = DecoderOnlyRunner(draft, capacity, "default", "eager")
    target_runner.warmup(input_ids, [1, GAMMA + 1], repeats=1)
    draft_runner.warmup(input_ids, [1, 2], repeats=1)
    compiled = counters["stats"]["unique_graphs"]

    speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)
    speculative_decode(target_runner, draft_runner, prompt(3), 1.0, GAMMA, MAX_NEW_TOKENS)
    assert counters["stats"]["unique_graphs"] == compiled


def test_models_with_different_output_layer_sizes_sample_the_same_tokens(target) -> None:
    """Qwen2.5-0.5B has 151936 logits and Qwen2.5-32B 152064, for the same tokenizer."""
    small_draft = tiny_qwen(seed=2, vocab_size=VOCAB - 16)
    common = VOCAB - 16

    for temperature in (0.0, 1.0):
        torch.manual_seed(0)
        tokens, _ = speculative_decode(
            target, small_draft, prompt(), temperature, GAMMA, MAX_NEW_TOKENS, vocab_size=common
        )
        assert tokens and max(tokens) < common

    # Without an explicit size the smaller vocabulary is used.
    tokens, _ = speculative_decode(target, small_draft, prompt(), 1.0, GAMMA, MAX_NEW_TOKENS)
    assert tokens and max(tokens) < common

    baseline = autoregressive_decode(target, prompt(), 0.0, MAX_NEW_TOKENS, vocab_size=common)
    tokens, _ = speculative_decode(target, small_draft, prompt(), 0.0, GAMMA, MAX_NEW_TOKENS, vocab_size=common)
    assert tokens == baseline


def test_eos_token_ids_reads_the_generation_config_and_accepts_a_list(target) -> None:
    model = tiny_qwen(seed=3)
    assert eos_token_ids(model) == {VOCAB - 1}
    model.generation_config.eos_token_id = [7, 9]
    assert eos_token_ids(model) == {7, 9}
    model.generation_config.eos_token_id = None
    model.config.eos_token_id = None
    assert eos_token_ids(model) == set()


@pytest.mark.parametrize("use_runner", [False, True])
def test_every_eos_token_ends_the_generation(target, draft, use_runner: bool) -> None:
    input_ids = prompt()
    free = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS)
    stop = free[3]  # Make a token that greedy decoding emits an end token, as <|im_end|> is for chat models.
    expected = free[: free.index(stop) + 1]

    ended_target, ended_draft = tiny_qwen(seed=0, layers=3), tiny_qwen(seed=1)
    ended_target.generation_config.eos_token_id = [VOCAB - 1, stop]
    ended_draft.generation_config.eos_token_id = [VOCAB - 1, stop]
    if use_runner:
        capacity = 8 + MAX_NEW_TOKENS + GAMMA + 1
        ended_target, ended_draft = DecoderOnlyRunner(ended_target, capacity), DecoderOnlyRunner(ended_draft, capacity)

    assert autoregressive_decode(ended_target, input_ids, 0.0, MAX_NEW_TOKENS) == expected
    tokens, _ = speculative_decode(ended_target, ended_draft, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)
    assert tokens == expected


def test_the_runners_reject_the_other_kind_of_model(target) -> None:
    with pytest.raises(ValueError):
        ModelRunner(target, 16)

    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    t5 = AutoModelForSeq2SeqLM.from_config(
        AutoConfig.for_model("t5", d_model=8, d_ff=16, num_layers=1, num_heads=2, d_kv=4)
    )
    with pytest.raises(ValueError):
        DecoderOnlyRunner(t5, 16)


def test_a_chat_is_tokenized_once_with_the_template() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    messages = [{"role": "user", "content": "Hello there"}]

    ids = encode(tokenizer, messages, max_input_length=512)

    text = tokenizer.decode(ids[0])
    assert text.count("<|im_start|>system") == 1
    assert text.endswith("<|im_start|>assistant\n")

    cut = encode(tokenizer, messages, max_input_length=5)
    assert cut.shape == (1, 5) and cut[0].tolist() == ids[0, -5:].tolist()  # The generation prompt is kept.
