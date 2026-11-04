"""
Plot the measured wall-clock speedup against the batch size, next to the speedup the model predicts.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("speedup_vs_batch_size.png"))
    parser.add_argument("--no-predictions", action="store_true", help="Only draw the measured speedup.")
    parser.add_argument("--range", action="store_true", help="Error bars span min to max instead of +- 1 std.")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_sizes = set()

    for i, path in enumerate(args.results):
        report = json.loads(path.read_text())
        results = sorted(
            (r for r in report["results"] if r.get("speedup") is not None), key=lambda r: r["batch_size"]
        )
        if not results:
            print(f"{path}: no speedup recorded, skipped")
            continue
        label = report.get("device_name", path.stem)
        color = colors[i % len(colors)]
        b = [r["batch_size"] for r in results]
        all_sizes.update(b)

        print(f"== {path.name} ({label})")
        means, lows, highs = [], [], []
        print(f"{'B':>4} {'runs':>4} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'predicted':>10} {'lockstep':>9}")
        for r in results:
            runs = np.asarray(r.get("speedup_runs") or [r["speedup"]], dtype=float)
            mean = runs.mean()
            std = runs.std(ddof=1) if len(runs) > 1 else 0.0
            means.append(mean)
            if args.range:
                lows.append(mean - runs.min())
                highs.append(runs.max() - mean)
            else:
                lows.append(std)
                highs.append(std)
            print(f"{r['batch_size']:>4} {len(runs):>4} {mean:>7.3f} {std:>7.3f} {runs.min():>7.3f} {runs.max():>7.3f} "
                  f"{r.get('predicted_speedup', float('nan')):>10.3f} "
                  f"{r.get('predicted_speedup_lockstep', float('nan')):>9.3f}")

        n_runs = min(len(r.get("speedup_runs") or [0]) for r in results)
        spread = "" if n_runs < 2 else (f", min-max of {n_runs} runs" if args.range else f", mean ± std of {n_runs} runs")
        ax.errorbar(b, means, yerr=[lows, highs], fmt="o-", capsize=4, color=color,
                    label=f"{label} measured{spread}")
        if not args.no_predictions:
            if all("predicted_speedup_lockstep" in r for r in results):
                ax.plot(b, [r["predicted_speedup_lockstep"] for r in results], "s--", color=color, alpha=0.7,
                        markersize=4, label=f"{label} predicted (lockstep)")
            if all("predicted_speedup" in r for r in results):
                ax.plot(b, [r["predicted_speedup"] for r in results], ":", color=color, alpha=0.5,
                        label=f"{label} predicted (independent)")

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="-")
    ax.text(min(all_sizes), 1.02, "no speedup", color="gray", fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(all_sizes), [str(s) for s in sorted(all_sizes)])
    ax.minorticks_off()
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup over autoregressive decoding (x)")
    ax.set_ylim(bottom=0.8)
    first = json.loads(args.results[0].read_text())
    ax.set_title(f"{first.get('task', '')}: {first.get('target', '')} with {first.get('draft', '')}".strip(": "),
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
