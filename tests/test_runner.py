"""
Tests for the runner that executes the decoding steps on static caches, with padded inputs, optionally compiled.

The CPU can only show that the compiled steps trace, that they give the same tokens, and that no step is compiled twice.
Whether CUDA graphs are recorded can only be seen on a GPU.
"""

import copy

import pytest
import torch
from torch._dynamo.utils import counters
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from speculative_decoding.decoding import DecoderState, autoregressive_decode, speculative_decode
from speculative_decoding.evaluation import EvaluationResult, evaluate
from speculative_decoding.runner import ModelRunner
from speculative_decoding.sampling import sample_token, standardize

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


@pytest.mark.parametrize("compile_mode, backend", [(None, "inductor"), ("default", "eager")])
def test_fused_draft_equals_baseline_at_temperature_zero(tokenizer, target, draft, compile_mode, backend) -> None:
    """Sampling every guess in one call must not change what comes out, whatever the draft model says."""
    target_runner = ModelRunner(target, CAPACITY, compile_mode, backend, encoder_bucket=16)
    draft_runner = ModelRunner(draft, CAPACITY, compile_mode, backend, encoder_bucket=16)

    for text in TEXTS:
        input_ids: torch.Tensor = encode(tokenizer, text)
        expected: list[int] = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS, use_cache=False)
        actual, stats = speculative_decode(
            target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS, fuse_draft=True
        )

        assert actual == expected
        assert stats.accepted < stats.drafted


def test_fused_draft_matches_the_unfused_loop_at_temperature_zero(tokenizer, target, draft) -> None:
    """Both ways of drafting must propose the same guesses from the same state, and leave the same state behind."""
    runner = ModelRunner(draft, CAPACITY, None, encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])

    with torch.inference_mode():
        fused: DecoderState = DecoderState(runner, input_ids)
        guesses, probs = fused.sample_guesses(GAMMA, 0.0)

        plain: DecoderState = DecoderState(ModelRunner(draft, CAPACITY, None, encoder_bucket=16), input_ids)
        expected: list[int] = []
        for _ in range(GAMMA):
            distribution = standardize(plain.logits()[-1], 0.0)
            expected.append(int(distribution.argmax()))
            plain.extend([expected[-1]])

    assert guesses.tolist() == expected
    assert fused.tokens == plain.tokens
    assert fused.cached == plain.cached == len(plain) - 1
    assert probs.shape == (GAMMA, draft.config.vocab_size)


def test_fused_draft_draws_fresh_samples_on_every_call(tokenizer, target) -> None:
    """A loop that replayed the same random draws would bias the sampling, as a CUDA graph can if RNG is baked in."""
    runner = ModelRunner(target, CAPACITY, None, encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])

    with torch.inference_mode():
        state: DecoderState = DecoderState(runner, input_ids)
        draws: list[list[int]] = []
        for _ in range(8):
            state.truncate(1)
            state.cached = 0
            guesses, _ = state.sample_guesses(GAMMA, 1.0)
            draws.append(guesses.tolist())

    assert len({tuple(draw) for draw in draws}) > 1


def test_fused_draft_samples_from_the_draft_distribution(tokenizer, target) -> None:
    """The most likely first guess must come up as often as its probability says."""
    runner = ModelRunner(target, CAPACITY, None, encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])
    draws: int = 400

    with torch.inference_mode():
        state: DecoderState = DecoderState(runner, input_ids)
        first: list[int] = []
        probs = None
        for _ in range(draws):
            state.truncate(1)
            state.cached = 0
            guesses, probs = state.sample_guesses(2, 1.0)
            first.append(int(guesses[0]))

    top: int = int(probs[0].argmax())
    expected: float = float(probs[0, top])
    observed: float = first.count(top) / draws
    assert abs(observed - expected) < 4 * (expected * (1 - expected) / draws) ** 0.5 + 0.01


def test_sample_token_follows_the_distribution() -> None:
    torch.manual_seed(0)
    probs: torch.Tensor = torch.tensor([0.5, 0.3, 0.2, 0.0])
    draws: torch.Tensor = torch.stack([sample_token(probs) for _ in range(6000)]).flatten()
    frequencies: torch.Tensor = torch.bincount(draws, minlength=4).float() / 6000

    assert torch.allclose(frequencies, probs, atol=0.03)
    assert frequencies[3] == 0  # An impossible token is never drawn.


def test_sample_token_of_a_one_hot_distribution_returns_its_token() -> None:
    probs: torch.Tensor = torch.nn.functional.one_hot(torch.tensor(7), num_classes=32).float()
    assert all(int(sample_token(probs)) == 7 for _ in range(50))


def test_fused_draft_needs_runners(tokenizer, target, draft) -> None:
    with pytest.raises(ValueError, match="needs runners"):
        speculative_decode(target, draft, encode(tokenizer, TEXTS[0]), 0.0, GAMMA, MAX_NEW_TOKENS, fuse_draft=True)


def test_nothing_is_compiled_after_the_warmup_of_the_fused_loop(tokenizer, target, draft) -> None:
    target_runner = ModelRunner(target, CAPACITY, "default", "eager", encoder_bucket=16)
    draft_runner = ModelRunner(draft, CAPACITY, "default", "eager", encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])
    draft_runner.warmup_loop(input_ids, [1, 2], GAMMA, 1.0)
    target_runner.warmup(input_ids, [1, GAMMA + 1])
    compiled: int = graphs()

    for _ in range(3):
        speculative_decode(target_runner, draft_runner, input_ids, 1.0, GAMMA, MAX_NEW_TOKENS, fuse_draft=True)

    assert graphs() == compiled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs are only recorded on a GPU")
def test_fused_draft_with_cuda_graphs_draws_fresh_samples_and_equals_baseline(tokenizer, target, draft) -> None:
    device = torch.device("cuda")
    # The fixtures are shared with the CPU tests, so they must not be moved.
    target_gpu, draft_gpu = copy.deepcopy(target).to(device), copy.deepcopy(draft).to(device)
    target_runner = ModelRunner(target_gpu, CAPACITY, "reduce-overhead", encoder_bucket=16)
    draft_runner = ModelRunner(draft_gpu, CAPACITY, "reduce-overhead", encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0]).to(device)
    draft_runner.warmup_loop(input_ids, [1, 2], GAMMA, 1.0)
    draft_runner.warmup_loop(input_ids, [1, 2], GAMMA, 0.0)
    target_runner.warmup(input_ids, [1, GAMMA + 1])
    assert counters["inductor"]["cudagraph_skips"] == 0  # Every step, the fused loop included, was recorded as a graph.

    with torch.inference_mode():
        state: DecoderState = DecoderState(draft_runner, input_ids)
        draws: set[tuple[int, ...]] = set()
        for _ in range(8):
            state.truncate(1)
            state.cached = 0
            guesses, _ = state.sample_guesses(GAMMA, 1.0)
            draws.add(tuple(guesses.tolist()))
    assert len(draws) > 1  # The graph must renew its random draws on every replay.

    expected: list[int] = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS, use_cache=False)
    actual, _ = speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS, fuse_draft=True)
    assert actual == expected


def test_evaluate_stops_at_the_token_budget_and_logs_the_breakdown(tokenizer, target, draft) -> None:
    result: EvaluationResult = evaluate(
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        inputs=TEXTS * 4,
        temperature=0.0,
        gamma=GAMMA,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_length=64,
        compile_mode="default",
        encoder_bucket=16,
        compile_backend="eager",
        min_generated_tokens=20,
        fuse_draft=True,
    )

    assert 20 <= result.generated_tokens
    assert result.num_examples < len(TEXTS) * 4
    assert result.fuse_draft
    assert result.draft_loop_ms is not None
    assert result.draft_step_ms > 0 and result.target_step_ms > 0 and result.verify_step_ms > 0
    assert result.encoder_ms > 0
    assert 0 <= result.alpha_analytic <= 1
    assert result.identical_outputs == result.num_examples
    assert result.truncated_generations <= result.num_examples
    assert result.overhead_ms == pytest.approx(result.iteration_ms - result.modelled_iteration_ms)


def test_evaluate_rejects_fusing_without_runners(tokenizer, target, draft) -> None:
    with pytest.raises(ValueError, match="runners"):
        evaluate(
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            inputs=TEXTS,
            temperature=0.0,
            gamma=GAMMA,
            max_new_tokens=MAX_NEW_TOKENS,
            max_input_length=64,
            fuse_draft=True,
        )


def test_fused_draft_compiled_with_inductor_traces_and_equals_baseline(tokenizer, target, draft) -> None:
    """The real compiler must accept the whole loop, with the sampling in it, as one graph."""
    target_runner = ModelRunner(target, CAPACITY, "default", encoder_bucket=16)
    draft_runner = ModelRunner(draft, CAPACITY, "default", encoder_bucket=16)
    input_ids: torch.Tensor = encode(tokenizer, TEXTS[0])
    for temperature in (0.0, 1.0):
        draft_runner.warmup_loop(input_ids, [1, 2], GAMMA, temperature, repeats=1)
    target_runner.warmup(input_ids, [1, GAMMA + 1], repeats=1)
    compiled: int = graphs()

    expected: list[int] = autoregressive_decode(target, input_ids, 0.0, MAX_NEW_TOKENS, use_cache=False)
    actual, _ = speculative_decode(target_runner, draft_runner, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS, fuse_draft=True)
    speculative_decode(target_runner, draft_runner, input_ids, 1.0, GAMMA, MAX_NEW_TOKENS, fuse_draft=True)

    assert actual == expected
    assert graphs() == compiled
