"""
Tests for the unigram and bigram draft models: fitting the counts, the smoothed tables, and decoding with them.
The target models are tiny and randomly initialised, so nothing is downloaded.
"""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from speculative_decoding.decoding import autoregressive_decode, speculative_decode
from speculative_decoding.evaluation import evaluate, make_runner
from speculative_decoding.ngram import (
    CountAccumulator,
    NGramCounts,
    NGramRunner,
    build_log_probs,
    parse_ngram_spec,
)

VOCAB: int = 12
START, END = 0, 1
MAX_NEW_TOKENS: int = 12
GAMMA: int = 3
DOCUMENTS: list[list[int]] = [[2, 3, 4, 3], [2, 3, 5], [6, 7]]


def fitted() -> NGramCounts:
    accumulator = CountAccumulator(VOCAB, START, END)
    accumulator.add(DOCUMENTS)
    return accumulator.finish(tokenizer_size=VOCAB - 1, source="test")


def test_parse_ngram_spec() -> None:
    assert parse_ngram_spec("bigram:/cache/pile.pt")[0] == 2
    assert parse_ngram_spec("unigram:/cache/pile.pt")[0] == 1
    assert str(parse_ngram_spec("bigram:/cache/pile.pt")[1]) == "/cache/pile.pt"
    assert parse_ngram_spec("EleutherAI/pythia-70m") is None


def test_counts_do_not_pair_tokens_across_documents() -> None:
    counts = fitted()
    pairs = {
        (int(code) // VOCAB, int(code) % VOCAB): int(n) for code, n in zip(counts.bigram_codes, counts.bigram_counts)
    }
    assert pairs[(START, 2)] == 2 and pairs[(2, 3)] == 2 and pairs[(3, 4)] == 1 and pairs[(4, 3)] == 1
    assert pairs[(3, END)] == 1 and pairs[(5, END)] == 1 and pairs[(7, END)] == 1
    assert (END, START) not in pairs and (END, 2) not in pairs  # Nothing follows the end of a document.
    assert counts.unigram[3] == 3 and counts.unigram[2] == 2 and counts.unigram[START] == 0
    assert counts.num_tokens == sum(len(d) + 1 for d in DOCUMENTS)


def test_counts_of_many_batches_equal_the_counts_of_one() -> None:
    one = fitted()
    accumulator = CountAccumulator(VOCAB, START, END)
    for document in DOCUMENTS:
        accumulator.add([document])
    many = accumulator.finish(VOCAB - 1)
    assert torch.equal(one.unigram, many.unigram)
    assert torch.equal(one.bigram_codes, many.bigram_codes) and torch.equal(one.bigram_counts, many.bigram_counts)


def test_documents_longer_than_the_limit_are_cut_without_an_end() -> None:
    accumulator = CountAccumulator(VOCAB, START, END, max_document_tokens=2)
    accumulator.add([[2, 3, 4, 5]])
    counts = accumulator.finish(VOCAB - 1)
    assert counts.unigram[END] == 0 and counts.unigram[2] == 1 and counts.unigram[3] == 1 and counts.unigram[4] == 0


def test_a_token_outside_the_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        CountAccumulator(VOCAB, START, END).add([[VOCAB]])


def test_unigram_table_is_the_add_k_distribution() -> None:
    table = build_log_probs(fitted(), order=1, smoothing=0.5)
    assert table.shape == (1, VOCAB)
    expected = (fitted().unigram.double() + 0.5) / (fitted().unigram.sum() + 0.5 * VOCAB)
    assert torch.allclose(table[0].exp().double(), expected, atol=1e-6)
    assert torch.isfinite(table).all()


def test_bigram_table_backs_off_to_the_unigram() -> None:
    counts = fitted()
    unigram = build_log_probs(counts, order=1, smoothing=1.0)[0].exp()
    table = build_log_probs(counts, order=2, smoothing=1.0, backoff_strength=2.0)
    assert table.shape == (VOCAB, VOCAB)
    assert torch.allclose(table.exp().sum(dim=1), torch.ones(VOCAB), atol=1e-5)  # Every row is a distribution.
    # After a 2 there were 2 observed 3s: (2 + 2 * u(3)) / (2 + 2).
    assert torch.allclose(table[2, 3].exp(), (2 + 2 * unigram[3]) / 4, atol=1e-6)
    # A token that never occurred as a context gets the unigram distribution.
    assert torch.allclose(table[9].exp(), unigram, atol=1e-6)


def test_the_bigram_table_does_not_depend_on_the_chunking(monkeypatch) -> None:
    whole = build_log_probs(fitted(), order=2)
    monkeypatch.setattr("speculative_decoding.ngram.CHUNK_ENTRIES", 3 * VOCAB)  # Three rows at a time.
    assert torch.equal(whole, build_log_probs(fitted(), order=2))


def test_counts_survive_saving(tmp_path) -> None:
    counts = fitted()
    counts.save(tmp_path / "counts.pt")
    loaded = NGramCounts.load(tmp_path / "counts.pt")
    assert loaded.vocab_size == VOCAB and loaded.tokenizer_size == VOCAB - 1 and loaded.source == "test"
    assert torch.equal(loaded.bigram_counts, counts.bigram_counts)

    runner = NGramRunner.from_file(tmp_path / "counts.pt", 2, torch.float32, "cpu", tokenizer_size=VOCAB - 1)
    assert runner.config.vocab_size == VOCAB
    with pytest.raises(ValueError, match="tokenizer"):
        NGramRunner.from_file(tmp_path / "counts.pt", 2, torch.float32, "cpu", tokenizer_size=VOCAB)


def test_runner_returns_the_row_of_every_token() -> None:
    bigram = NGramRunner(build_log_probs(fitted(), 2), order=2)
    unigram = NGramRunner(build_log_probs(fitted(), 1), order=1)
    tokens = torch.tensor([[2, 5, 3]])
    slot = bigram.prepare(tokens)
    assert torch.equal(bigram.forward(slot, tokens, 0), bigram.table[[2, 5, 3]])
    logits = unigram.forward(unigram.prepare(tokens), tokens, 0)
    assert logits.shape == (3, VOCAB) and torch.equal(logits[0], logits[2])


def tiny_target() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        eos_token_id=VOCAB - 1,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("prompt", [[[0]], [[0, 2, 3, 4]]])
def test_speculative_decoding_with_an_ngram_draft_equals_the_baseline(order: int, prompt) -> None:
    target = tiny_target()
    input_ids = torch.tensor(prompt)
    draft = NGramRunner(build_log_probs(fitted(), order), order)
    target_runner = make_runner(target, 4 + MAX_NEW_TOKENS + GAMMA + 1, "eager")

    baseline = autoregressive_decode(target_runner, input_ids, 0.0, MAX_NEW_TOKENS)
    tokens, stats = speculative_decode(target_runner, draft, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)
    assert tokens == baseline
    assert stats.drafted > 0


@pytest.mark.parametrize("order", [1, 2])
def test_sampling_with_an_ngram_draft_runs_and_stays_in_the_vocabulary(order: int) -> None:
    target = tiny_target()
    draft = NGramRunner(build_log_probs(fitted(), order), order)
    target_runner = make_runner(target, 1 + MAX_NEW_TOKENS + GAMMA + 1, "eager")
    for seed in range(3):
        torch.manual_seed(seed)
        tokens, _ = speculative_decode(target_runner, draft, torch.tensor([[0]]), 1.0, GAMMA, MAX_NEW_TOKENS)
        assert 0 < len(tokens) <= MAX_NEW_TOKENS and all(0 <= token < VOCAB for token in tokens)


class StartTokenizer:
    bos_token_id = START
    eos_token_id = VOCAB - 1

    def __len__(self) -> int:
        return VOCAB


def test_evaluate_accepts_an_ngram_draft() -> None:
    target = tiny_target()
    draft = NGramRunner(build_log_probs(fitted(), 2), 2)
    target_runner = make_runner(target, 1 + MAX_NEW_TOKENS + GAMMA + 1, "eager")
    result = evaluate(
        target=target_runner,
        draft=draft,
        tokenizer=StartTokenizer(),
        inputs=["", ""],
        temperature=0.0,
        gamma=GAMMA,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_length=8,
        compile_mode="eager",
    )
    assert result.identical_outputs == 2
    assert 0.0 <= result.alpha <= 1.0


def test_documents_are_tokenized_in_batches() -> None:
    from speculative_decoding.ngram import documents_from

    tokenizer = lambda texts, add_special_tokens: {"input_ids": [[ord(c) % VOCAB for c in t] for t in texts]}  # noqa: E731
    batches = list(documents_from(["ab", "cd", "ef"], tokenizer, batch_size=2))
    assert [len(batch) for batch in batches] == [2, 1]


def tiny_t5(seed: int, layers: int):
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(seed)
    config = T5Config(
        vocab_size=VOCAB,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=layers,
        num_decoder_layers=layers,
        num_heads=4,
        decoder_start_token_id=START,
        eos_token_id=END,
        pad_token_id=START,
    )
    return T5ForConditionalGeneration(config).eval()


@pytest.mark.parametrize("order", [1, 2])
def test_ngram_draft_for_an_encoder_decoder_target_decodes_from_its_start_token(order: int) -> None:
    """A T5 target decodes from the pad token whatever the input is, so the draft must not take the input as prompt."""
    target = tiny_t5(0, layers=2)
    draft = NGramRunner(build_log_probs(fitted(), order), order, decoder_start_token_id=START)
    assert draft.config.is_encoder_decoder and draft.config.decoder_start_token_id == START
    target_runner = make_runner(target, 1 + MAX_NEW_TOKENS + GAMMA + 1, "eager", encoder_bucket=8)

    for input_ids in (torch.tensor([[5, 6, 7, 8, 9]]), torch.tensor([[7, 3]])):
        baseline = autoregressive_decode(target_runner, input_ids, 0.0, MAX_NEW_TOKENS)
        tokens, stats = speculative_decode(target_runner, draft, input_ids, 0.0, GAMMA, MAX_NEW_TOKENS)
        assert tokens == baseline
        assert stats.drafted > 0


def test_a_decoder_only_ngram_draft_is_not_an_encoder_decoder_model() -> None:
    draft = NGramRunner(build_log_probs(fitted(), 1), 1)
    assert not draft.config.is_encoder_decoder


def test_a_dotted_column_reaches_into_nested_columns() -> None:
    from speculative_decoding.ngram import get_column

    example = {"translation": {"en": "house", "de": "Haus"}, "article": "text"}
    assert get_column(example, "translation.de") == "Haus"
    assert get_column(example, "article") == "text"
