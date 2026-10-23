"""
Tests for batch_scaling.py. They use a tiny randomly initialised T5, so they need neither a GPU nor a download.
"""

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from speculative_decoding.batch_scaling import generalised_speedup, measure_batch_scaling, time_decoder_steps
from speculative_decoding.metrics import expected_speedup


def tiny_t5() -> T5ForConditionalGeneration:
    config: T5Config = T5Config(
        vocab_size=100,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return T5ForConditionalGeneration(config).eval()


@pytest.mark.parametrize(
    "alpha, gamma, c, expected",
    [
        # Rows of Table 4 of the paper (Appendix A.3), whose EXP column is Theorem 3.8 rounded to one decimal.
        (0.75, 7, 0.02, 3.2),  # EnDe, T5-small, temp 0
        (0.8, 7, 0.04, 3.3),  # EnDe, T5-base, temp 0
        (0.82, 7, 0.11, 2.5),  # EnDe, T5-large, temp 0
    ],
)
def test_reduces_to_theorem_3_8_with_the_paper_table(alpha: float, gamma: int, c: float, expected: float) -> None:
    assert generalised_speedup(alpha, gamma, c, r=1.0) == pytest.approx(expected_speedup(alpha, gamma, c))
    assert round(generalised_speedup(alpha, gamma, c, r=1.0), 1) == expected


def test_hand_computed_value_and_monotonicity() -> None:
    # alpha=0.8, gamma=5: (1 - 0.8**6) / 0.2 = 3.68928 tokens per iteration.
    assert generalised_speedup(0.8, 5, 0.02, 1.0) == pytest.approx(3.68928 / 1.1)
    assert generalised_speedup(0.8, 5, 0.02, 1.5) == pytest.approx(3.68928 / 1.6)
    assert generalised_speedup(0.8, 5, 0.02, 2.0) < generalised_speedup(0.8, 5, 0.02, 1.0)
    assert generalised_speedup(1.0, 5, 0.0, 1.0) == pytest.approx(6.0)  # Every guess accepted, free draft.


@pytest.mark.parametrize(
    "alpha, gamma, c, r", [(-0.1, 3, 0.1, 1), (1.1, 3, 0.1, 1), (0.5, 0, 0.1, 1), (0.5, 3, -1, 1), (0.5, 3, 0.1, 0)]
)
def test_rejects_invalid_arguments(alpha: float, gamma: int, c: float, r: float) -> None:
    with pytest.raises(ValueError):
        generalised_speedup(alpha, gamma, c, r)


def test_time_decoder_steps_leaves_the_cache_consistent() -> None:
    model: T5ForConditionalGeneration = tiny_t5()
    timings = time_decoder_steps(
        model, batch_size=3, step_lengths=[1, 4], input_length=5, prefix_length=7, repeats=2, trials=2, warmup=1
    )
    assert set(timings) == {1, 4}
    for timing in timings.values():
        assert 0 < timing.minimum <= timing.median <= timing.maximum


def test_time_decoder_steps_rejects_bad_models() -> None:
    model: T5ForConditionalGeneration = tiny_t5()
    model.train()
    with pytest.raises(ValueError):
        time_decoder_steps(model, 1, [1], 5, 7)


def test_measure_batch_scaling_on_a_tiny_t5() -> None:
    target: T5ForConditionalGeneration = tiny_t5()
    draft: T5ForConditionalGeneration = tiny_t5()
    points = measure_batch_scaling(
        target,
        draft,
        batch_sizes=[2, 1, 2],
        gamma=3,
        alpha=0.7,
        input_length=5,
        prefix_length=7,
        repeats=2,
        trials=2,
        warmup=1,
    )

    assert [point.batch_size for point in points] == [1, 2]  # Sorted and deduplicated.
    for point in points:
        assert point.cost_coefficient == pytest.approx(point.draft_step.median / point.target_step.median)
        assert point.verification_ratio == pytest.approx(point.target_verify.median / point.target_step.median)
        assert point.speedup == pytest.approx(
            generalised_speedup(0.7, 3, point.cost_coefficient, point.verification_ratio)
        )
        assert point.speedup_theorem_3_8 == pytest.approx(expected_speedup(0.7, 3, point.cost_coefficient))


def test_measure_batch_scaling_stops_at_the_first_out_of_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from speculative_decoding import batch_scaling

    real = batch_scaling.time_decoder_steps

    def fail_from_four(model, batch_size, *args, **kwargs):
        if batch_size >= 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(model, batch_size, *args, **kwargs)

    monkeypatch.setattr(batch_scaling, "time_decoder_steps", fail_from_four)
    points = measure_batch_scaling(
        tiny_t5(),
        tiny_t5(),
        batch_sizes=[1, 2, 4, 8],
        gamma=2,
        alpha=0.7,
        input_length=5,
        prefix_length=7,
        repeats=1,
        trials=1,
        warmup=0,
    )

    assert [point.batch_size for point in points] == [1, 2]  # 4 ran out of memory and 8 was skipped.
