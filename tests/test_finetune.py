import numpy as np
import pytest
import torch

from finetune.data import IGNORE_INDEX, TokenizedPairs, collate, make_batches, prefetch, train_batches
from finetune.train import seq2seq_loss, shift_right


def make_data(lengths: list[int]) -> TokenizedPairs:
    inputs = [np.arange(1, n + 1, dtype=np.int32) for n in lengths]
    targets = [np.arange(1, n // 2 + 2, dtype=np.int32) for n in lengths]
    return TokenizedPairs(inputs, targets)


def test_batches_cover_every_example_once_within_budget():
    lengths = list(np.random.default_rng(0).integers(5, 200, size=500))
    data = make_data(lengths)
    batches = make_batches(data, 1000, np.random.default_rng(1))

    assert sorted(np.concatenate(batches).tolist()) == list(range(500))
    for batch in batches:
        assert len(batch) * max(lengths[i] for i in batch) <= 1000 or len(batch) == 1


def test_train_batches_give_ranks_equal_counts():
    data = make_data([10] * 50)
    streams = [train_batches(data, 40, rank, 3, 0) for rank in range(3)]
    first = [[next(s).input_ids.shape for _ in range(5)] for s in streams]
    assert all(len(f) == 5 for f in first)


def test_collate_pads_labels_with_ignore_index():
    data = make_data([4, 8])
    batch = collate(data, np.array([0, 1]))
    assert batch.input_ids.shape == (2, 8)
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert batch.labels[0, -1] == IGNORE_INDEX
    assert batch.input_ids[0].tolist() == [1, 2, 3, 4, 0, 0, 0, 0]
    assert batch.input_ids[1].tolist() == list(range(1, 9))
    assert (batch.input_tokens, batch.target_tokens) == (12, 3 + 5)
    assert batch.padded_tokens == 16 + 2 * 5


def test_shift_right():
    labels = torch.tensor([[5, 6, 1, IGNORE_INDEX]])
    assert shift_right(labels, 0, 0).tolist() == [[0, 5, 6, 1]]


def test_loss_matches_cross_entropy_and_ignores_padding():
    logits = torch.randn(2, 4, 11)
    labels = torch.tensor([[1, 2, 3, IGNORE_INDEX], [4, 5, IGNORE_INDEX, IGNORE_INDEX]])
    loss, nll = seq2seq_loss(logits, labels, 0.0)
    expected = torch.nn.functional.cross_entropy(logits.view(-1, 11), labels.view(-1), ignore_index=IGNORE_INDEX)
    assert torch.allclose(nll, expected) and torch.allclose(loss, expected)

    with_z, nll_z = seq2seq_loss(logits, labels, 1e-4)
    assert torch.allclose(nll_z, expected) and with_z > nll_z


def test_sharded_adafactor_matches_hf_adafactor_on_unsharded_parameters():
    from transformers import Adafactor

    from finetune.optim import ShardedAdafactor

    torch.manual_seed(0)
    reference = torch.nn.Sequential(torch.nn.Linear(6, 9), torch.nn.LayerNorm(9), torch.nn.Linear(9, 3))
    model = torch.nn.Sequential(torch.nn.Linear(6, 9), torch.nn.LayerNorm(9), torch.nn.Linear(9, 3))
    model.load_state_dict(reference.state_dict())
    hf = Adafactor(reference.parameters(), lr=1e-3, beta1=None, scale_parameter=True, relative_step=False)
    mine = ShardedAdafactor(model.parameters(), lr=1e-3)

    x = torch.randn(5, 6)
    for _ in range(3):
        for net, optimizer in ((reference, hf), (model, mine)):
            net(x).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad()

    for a, b in zip(reference.parameters(), model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_tokenized_cache_roundtrip(tmp_path):
    data = make_data([4, 9, 6])
    data.save(tmp_path / "cache.npz")
    loaded = TokenizedPairs.load(tmp_path / "cache.npz")

    assert len(loaded) == 3
    for a, b in zip(data.inputs + data.targets, loaded.inputs + loaded.targets):
        assert a.tolist() == b.tolist()
    assert not list(tmp_path.glob("*.tmp"))


def reference_batches(lengths, order_rng_seed, budget):
    """The batching as a plain loop over examples."""
    data_lengths = np.asarray(lengths)
    order = np.random.default_rng(order_rng_seed).permutation(len(lengths))
    batches = []
    for start in range(0, len(order), 20_000):
        pool = order[start : start + 20_000]
        pool = pool[np.argsort(data_lengths[pool], kind="stable")]
        begin = 0
        for end in range(len(pool)):
            if (end + 1 - begin) * data_lengths[pool[end]] > budget and end > begin:
                batches.append(pool[begin:end])
                begin = end
        batches.append(pool[begin:])
    return batches


@pytest.mark.parametrize("budget", [7, 100, 1000])
def test_vectorized_batching_matches_the_loop(budget):
    lengths = np.random.default_rng(0).integers(1, 60, size=3000)
    data = TokenizedPairs([np.zeros(n, dtype=np.int32) for n in lengths], [np.zeros(1, dtype=np.int32)] * len(lengths))

    got = make_batches(data, budget, np.random.default_rng(5))
    # Unshuffle: compare as the set of batches, since the order of batches is shuffled with the same generator.
    want = reference_batches(lengths, 5, budget)
    assert sorted(map(tuple, got)) == sorted(map(tuple, want))


def test_prefetch_keeps_order_and_raises_producer_errors():
    data = make_data([5, 6, 7, 8])
    batches = [collate(data, np.array([i])) for i in range(4)]
    assert [b.input_tokens for b in prefetch(iter(batches), depth=2)] == [5, 6, 7, 8]

    def failing():
        yield batches[0]
        raise RuntimeError("boom")

    stream = prefetch(failing())
    next(stream)
    with pytest.raises(RuntimeError, match="boom"):
        next(stream)


def test_loss_normalizer_makes_micro_batches_weigh_tokens_equally():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    labels[1, 2:] = IGNORE_INDEX
    whole, _ = seq2seq_loss(logits, labels, 1e-4)

    # Split into two micro-batches of 5 and 2 tokens. Dividing by each one's own count would weigh them equally.
    total = (labels != IGNORE_INDEX).sum().item()
    parts = sum(seq2seq_loss(logits[i : i + 1], labels[i : i + 1], 1e-4, total)[0] for i in range(2))
    assert torch.allclose(whole, parts)


def test_bench_recommends_the_fastest_size_that_leaves_headroom():
    from finetune.bench import recommend, worst_case_batch

    results = [
        {"micro_batch_tokens": 2048, "worst_case_peak_gb": 60.0, "tokens_per_second": 100},
        {"micro_batch_tokens": 4096, "worst_case_peak_gb": 90.0, "tokens_per_second": 130},
        {"micro_batch_tokens": 8192, "worst_case_peak_gb": 170.0, "tokens_per_second": 140},
    ]
    assert recommend(results, 180.0, 0.1)["micro_batch_tokens"] == 4096  # 170 GB of 180 leaves under 10% free.
    assert recommend(results, 50.0, 0.1) is None
    batch = worst_case_batch(2048, 512, 100)
    assert batch.input_ids.shape == (4, 512) and batch.target_tokens == 2048


def test_make_contiguous_fixes_transposed_parameters():
    from finetune.train import make_contiguous

    layer = torch.nn.Linear(4, 6)
    layer.weight.data = layer.weight.data.t().contiguous().t()  # The layout of the hub copy of XXL.
    expected = layer.weight.detach().clone()
    assert not layer.weight.is_contiguous()

    assert make_contiguous(layer) == 1
    assert layer.weight.is_contiguous() and torch.equal(layer.weight, expected)
    assert make_contiguous(layer) == 0
