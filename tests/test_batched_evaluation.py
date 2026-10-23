"""
Tests for the batched benchmark and its command line. They use tiny randomly initialised T5 models and a tiny word level
tokenizer, so they need neither a GPU nor a download.
"""

import json
import sys

import pytest
import torch
from tiny_models import make_t5, perturbed_copy
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from speculative_decoding import batched_evaluate
from speculative_decoding.batched_evaluation import evaluate_batched, make_batches
from speculative_decoding.metrics import expected_tokens_per_iteration

WORDS: list[str] = [
    "the",
    "house",
    "is",
    "wonderful",
    "a",
    "cat",
    "sat",
    "on",
    "mat",
    "and",
    "dog",
    "ran",
    "far",
    "away",
]

INPUTS: list[str] = [
    "the house is wonderful",
    "a cat sat on the mat",
    "the dog ran far away and the cat sat",
    "house",
    "a dog",
    "the cat and the dog ran away",
    "wonderful house on a mat",
    "far away",
    "the mat is a house and a house is a mat",
    "cat dog",
    "the dog sat",
    "a house is far away from the cat",
]


def make_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary: dict[str, int] = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    vocabulary.update({word: index + 3 for index, word in enumerate(WORDS)})
    tokenizer: Tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", padding_side="right"
    )


@pytest.fixture(scope="module")
def models_and_tokenizer():
    target = make_t5(seed=1, vocab_size=40)
    return target, perturbed_copy(target, noise=0.08), make_tokenizer()


def test_make_batches_pads_masks_and_drops_the_remainder() -> None:
    tokenizer = make_tokenizer()
    batches = make_batches(tokenizer, INPUTS[:7], batch_size=3, max_input_length=6, device=torch.device("cpu"))

    assert len(batches) == 2  # 7 inputs make 2 batches of 3, the last one is dropped.
    for ids, mask in batches:
        assert ids.shape == mask.shape and ids.shape[0] == 3 and ids.shape[1] <= 6
        assert torch.equal(mask.bool(), ids != 0)  # 1 for tokens, 0 for the padding.
    assert batches[0][1].sum(dim=1).tolist() == [4, 6, 6]  # "the dog ran far away and the cat sat" is truncated to 6.

    with pytest.raises(ValueError):
        make_batches(tokenizer, INPUTS[:2], batch_size=3, max_input_length=6, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        make_batches(tokenizer, INPUTS, batch_size=0, max_input_length=6, device=torch.device("cpu"))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_greedy_benchmark_outputs_equal_the_baseline(models_and_tokenizer, batch_size: int) -> None:
    target, draft, tokenizer = models_and_tokenizer
    result = evaluate_batched(
        target, draft, tokenizer, INPUTS, batch_size, temperature=0.0, gamma=3, max_new_tokens=10, max_input_length=16
    )

    assert result.num_examples == 12 and result.num_batches == 12 // batch_size
    assert result.identical_outputs == 12  # Greedy decoding gives the same text as the baseline.
    assert 0 < result.alpha <= 1
    assert 1 <= result.tokens_per_iteration <= result.gamma + 1
    assert result.expected_tokens_per_iteration == pytest.approx(expected_tokens_per_iteration(result.alpha, 3))
    assert result.lockstep_efficiency == pytest.approx(
        result.tokens_per_iteration / result.expected_tokens_per_iteration
    )
    assert result.speedup == pytest.approx(result.baseline_seconds / result.speculative_seconds)
    assert result.generated_tokens > 0 and result.speculative_tokens_per_second > 0
    assert result.cost_coefficient is None and result.predicted_speedup is None  # Only with `predict`.
    if batch_size == 1:
        # One sequence is Equation (1) itself, as far as the sample allows. The lockstep formula agrees with it.
        assert result.expected_lockstep_tokens_per_iteration == pytest.approx(result.expected_tokens_per_iteration)


def test_benchmark_with_predictions(models_and_tokenizer) -> None:
    target, draft, tokenizer = models_and_tokenizer
    result = evaluate_batched(
        target,
        draft,
        tokenizer,
        INPUTS,
        4,
        temperature=1.0,
        gamma=3,
        max_new_tokens=8,
        max_input_length=16,
        predict=True,
    )

    assert result.cost_coefficient > 0 and result.verification_ratio > 0
    assert result.predicted_speedup_lockstep == pytest.approx(
        result.tokens_per_iteration / (3 * result.cost_coefficient + result.verification_ratio)
    )
    assert result.predicted_speedup_theorem_3_8 > 0 and result.predicted_speedup > 0


def test_benchmark_without_the_baseline(models_and_tokenizer) -> None:
    target, draft, tokenizer = models_and_tokenizer
    result = evaluate_batched(target, draft, tokenizer, INPUTS, 3, 1.0, 2, 6, 16, run_baseline=False)

    assert result.speedup is None and result.baseline_seconds is None and result.identical_outputs is None
    assert result.speculative_seconds > 0


# The command line -------------------------------------------------------------------------------------------------


def test_command_line_end_to_end(tmp_path, monkeypatch, capsys) -> None:
    target = make_t5(seed=1, vocab_size=40)
    target.save_pretrained(tmp_path / "target")
    perturbed_copy(target, noise=0.08).save_pretrained(tmp_path / "draft")
    make_tokenizer().save_pretrained(tmp_path / "target")
    output = tmp_path / "out" / "batched.json"

    monkeypatch.setattr(batched_evaluate, "load_inputs", lambda task, count: INPUTS[:count])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batched_evaluate",
            "-t",
            str(tmp_path / "target"),
            "-d",
            str(tmp_path / "draft"),
            "-g",
            "3",
            "-k",
            "translation",
            "--temperature",
            "0",
            "--batch-sizes",
            "1",
            "4",
            "64",  # 64 is more than the inputs and is skipped.
            "--num-examples",
            "12",
            "--max-new-tokens",
            "8",
            "--predict",
            "--output",
            str(output),
        ],
    )
    batched_evaluate.main()

    report = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert [result["batch_size"] for result in report["results"]] == [1, 4]
    assert all(result["identical_outputs"] == 12 for result in report["results"])
    assert report["task"] == "translation" and report["device"] in {"cpu", "cuda", "mps"}
    assert all(result["predicted_speedup_lockstep"] > 0 for result in report["results"])


@pytest.mark.parametrize(
    "extra",
    [["-g", "0"], ["--temperature", "-1"], ["--batch-sizes", "0"], ["--num-examples", "1", "--batch-sizes", "2"]],
)
def test_command_line_rejects_bad_arguments(monkeypatch, extra: list[str]) -> None:
    base = ["x", "-t", "t", "-d", "d", "-g", "3", "-k", "translation", "--temperature", "0"]
    # Later flags override earlier ones, so the extra ones replace the valid defaults.
    monkeypatch.setattr(sys, "argv", base + extra)
    with pytest.raises(SystemExit):
        batched_evaluate.parse_arguments()
