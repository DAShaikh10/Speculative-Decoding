"""
Tests for the runner that executes the decoding steps on static caches, with padded inputs, optionally compiled.

The CPU can only show that the compiled steps trace, that they give the same tokens, and that no step is compiled twice.
Whether CUDA graphs are recorded can only be seen on a GPU.
"""

import pytest
import torch
from torch._dynamo.utils import counters
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from speculative_decoding.decoding import DecoderState, autoregressive_decode, speculative_decode
from speculative_decoding.evaluation import EvaluationResult, evaluate
from speculative_decoding.runner import ModelRunner

MODEL_NAME: str = "google-t5/t5-small"
MAX_NEW_TOKENS: int = 12
GAMMA: int = 3
CAPACITY: int = 1 + MAX_NEW_TOKENS + GAMMA + 1
TEXTS: list[str] = [
    "translate English to German: The house is wonderful.",
    "translate English to German: I would like a cup of tea, please, and a slice of cake.",
    "translate English to German: Good morning.",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def target():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()


@pytest.fixture(scope="module")
def draft():
    """A randomly initialised model with the same architecture, which disagrees with the target."""
    torch.manual_seed(0)
    return AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(MODEL_NAME)).eval()


def encode(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt").input_ids


def graphs() -> int:
    return counters["stats"]["unique_graphs"]


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()


@pytest.mark.parametrize("compile_mode, backend", [(None, "inductor"), ("default", "eager")])
def test_speculative_decode_through_runners_equals_baseline(tokenizer, target, draft, compile_mode, backend) -> None:
    """At temperature 0 the output must be the baseline's, whatever the draft model says and the steps run as."""
    target_runner = ModelRunner(target, CAPACITY, compile_mode, backend, encoder_bucket=16)
    draft_runner = ModelRunner(draft, CAPACITY, compile_mode, backend, encoder_bucket=16)

    for text in TEXTS:
        input_ids: torch.Tensor = encode(tokenizer, text)
        expected: list[int] = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS, use_cache=False)
        actual, stats = speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)

        assert actual == expected
        assert stats.accepted < stats.drafted


def test_autoregressive_decode_through_a_runner_equals_the_plain_model(tokenizer, target) -> None:
    runner = ModelRunner(target, CAPACITY, "default", "eager", encoder_bucket=16)
    for text in TEXTS:
        input_ids: torch.Tensor = encode(tokenizer, text)
        assert autoregressive_decode(runner, input_ids, 0.0, MAX_NEW_TOKENS) == autoregressive_decode(
            target, input_ids, 0.0, MAX_NEW_TOKENS
        )


def test_nothing_is_compiled_after_the_warmup(tokenizer, target, draft) -> None:
    """The warmup has to cover every step shape and cache state that decoding uses, so that no compile is timed."""
    target_runner = ModelRunner(target, CAPACITY, "default", "eager", encoder_bucket=16)
    draft_runner = ModelRunner(draft, CAPACITY, "default", "eager", encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])
    draft_runner.warmup(input_ids, [1, 2])
    target_runner.warmup(input_ids, [1, GAMMA + 1])
    compiled: int = graphs()
    assert compiled > 0

    for temperature in (0.0, 1.0):  # Sampling accepts more or fewer guesses, so it uses other step lengths.
        for _ in range(3):
            speculative_decode(target_runner, draft_runner, input_ids, temperature, GAMMA, MAX_NEW_TOKENS)
        autoregressive_decode(target_runner, input_ids, temperature, MAX_NEW_TOKENS)

    assert graphs() == compiled


def test_one_runner_serves_every_input_of_a_bucket(tokenizer, target, draft) -> None:
    """Different inputs share the caches and the graphs of their bucket, and a new bucket gets its own."""
    target_runner = ModelRunner(target, CAPACITY, "default", "eager", encoder_bucket=16)
    draft_runner = ModelRunner(draft, CAPACITY, "default", "eager", encoder_bucket=16)
    short, long = encode(tokenizer, TEXTS[0]), encode(tokenizer, TEXTS[1])
    assert target_runner.bucket_for(short.shape[1]) < target_runner.bucket_for(long.shape[1])

    def decode(input_ids: torch.Tensor) -> list[int]:
        tokens, _ = speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)
        return tokens

    for input_ids in (short, long):
        for runner, lengths in ((draft_runner, [1, 2]), (target_runner, [1, GAMMA + 1])):
            runner.warmup(input_ids, lengths)
    compiled: int = graphs()

    for input_ids in (short, long, short, long, encode(tokenizer, TEXTS[2])):
        expected: list[int] = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS, use_cache=False)
        assert decode(input_ids) == expected
    assert graphs() == compiled


def test_compiled_with_inductor_equals_baseline(tokenizer, target, draft) -> None:
    """The real compiler, which on a GPU with mode='reduce-overhead' records CUDA graphs."""
    target_runner = ModelRunner(target, CAPACITY, "default", encoder_bucket=16)
    draft_runner = ModelRunner(draft, CAPACITY, "default", encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])
    draft_runner.warmup(input_ids, [1, 2], repeats=1)
    target_runner.warmup(input_ids, [1, GAMMA + 1], repeats=1)
    compiled: int = graphs()

    expected: list[int] = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS, use_cache=False)
    actual, _ = speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)

    assert actual == expected
    assert graphs() == compiled


def test_decoding_state_can_be_rolled_back_through_a_runner(tokenizer, target) -> None:
    runner = ModelRunner(target, CAPACITY, "default", "eager", encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])

    with torch.inference_mode():
        state: DecoderState = DecoderState(runner, input_ids)
        state.logits()
        state.extend([5, 6, 7, 8])
        state.logits()
        state.truncate(len(state) - 3)  # Reject the last three tokens.
        state.extend([9, 10])
        actual: torch.Tensor = state.logits().clone()

        reference: DecoderState = DecoderState(target, input_ids, use_cache=False)
        reference.tokens = list(state.tokens)
        expected: torch.Tensor = reference.logits()[-actual.shape[0] :]

    assert torch.allclose(actual, expected, atol=1e-3)


def test_the_runner_rejects_decoder_only_models() -> None:
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").eval()
    with pytest.raises(ValueError, match="encoder-decoder"):
        ModelRunner(model, 16)


def test_a_runner_and_a_plain_model_cannot_be_mixed(tokenizer, target, draft) -> None:
    runner = ModelRunner(target, CAPACITY, None, encoder_bucket=16)
    with pytest.raises(ValueError, match="both models or neither"):
        speculative_decode(runner, draft, encode(tokenizer, TEXTS[0]), 0.0, GAMMA, MAX_NEW_TOKENS)


def test_evaluate_with_compiled_runners(tokenizer, target, draft) -> None:
    result: EvaluationResult = evaluate(
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        inputs=TEXTS,
        temperature=0.0,
        gamma=GAMMA,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_length=64,
        compile_mode="default",
        encoder_bucket=16,
        compile_backend="eager",
    )

    assert result.compile_mode == "default"
    assert result.identical_outputs == len(TEXTS)
    assert result.speedup is not None
