"""
Plot the acceptance rate alpha against the batch size, with bootstrap confidence intervals.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def alpha_of(counts: np.ndarray) -> np.ndarray:

    accepted = counts[..., 0].sum(axis=-1)
    decided = counts.sum(axis=(-1, -2))
    return accepted / decided


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("alpha_vs_batch_size.png"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = json.loads(args.results.read_text())
    results = sorted(report["results"], key=lambda result: result["batch_size"])
    if not results:
        raise SystemExit("The file has no results (did every batch size run out of memory?)")
    missing = [r["batch_size"] for r in results if not r.get("example_counts")]
    if missing:
        raise SystemExit(
            f"No `example_counts` for batch sizes {missing}. The evaluation has to record the accepted and rejected "
            "guesses of every example before error bars can be computed."
        )

    counts = [np.asarray(r["example_counts"], dtype=float) for r in results]
    sizes = {c.shape[0] for c in counts}
    if len(sizes) != 1:
        raise SystemExit(f"The batch sizes decoded different numbers of examples: {sorted(sizes)}")
    num_examples = sizes.pop()

    rng = np.random.default_rng(args.seed)
    picks = rng.integers(0, num_examples, size=(args.resamples, num_examples))
    low_q, high_q = (1 - args.confidence) / 2 * 100, (1 + args.confidence) / 2 * 100

    batch_sizes, points, lows, highs = [], [], [], []
    boot = []
    for result, c in zip(results, counts):
        boot_alpha = alpha_of(c[picks])
        boot.append(boot_alpha)
        batch_sizes.append(result["batch_size"])
        points.append(float(alpha_of(c)))
        lows.append(np.percentile(boot_alpha, low_q))
        highs.append(np.percentile(boot_alpha, high_q))

    print(f"alpha with a {args.confidence:.0%} bootstrap interval over {num_examples} examples")
    print(f"{'B':>4} {'alpha':>8} {'low':>8} {'high':>8} | {'alpha(B) - alpha(first)':>24}")
    for i, b in enumerate(batch_sizes):
        diff = boot[i] - boot[0]
        d_low, d_high = np.percentile(diff, [low_q, high_q])
        verdict = "" if i == 0 else ("excludes 0" if d_low > 0 or d_high < 0 else "includes 0")
        print(f"{b:>4} {points[i]:>8.4f} {lows[i]:>8.4f} {highs[i]:>8.4f} | "
              f"{points[i] - points[0]:+.4f} [{d_low:+.4f}, {d_high:+.4f}] {verdict}")

    x = np.arange(len(batch_sizes))
    err = np.array([np.array(points) - np.array(lows), np.array(highs) - np.array(points)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, points, yerr=err, fmt="o-", capsize=4)
    ax.axhline(points[0], linestyle="--", linewidth=0.8, color="gray", label=f"first batch size ({points[0]:.3f})")
    ax.set_xticks(x, [str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Acceptance rate alpha")
    ax.set_title(f"{report.get('task', '')} {report.get('target', '')}".strip())
    ax.set_ylim(min(lows) - 0.03, max(highs) + 0.03)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
