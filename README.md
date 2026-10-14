# Speculative Decoding: replicating "Fast Inference from Transformers via Speculative Decoding"

This project is our attempt to reproduce, end to end, the paper by Leviathan, Kalman and Matias (Google Research):

> *Fast Inference from Transformers via Speculative Decoding* — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

There is no code here yet. This README is the plan: what the paper claims, what we want to rebuild, and how we'll know we got it right.

## The idea in plain words

Large language models generate text one token at a time, and every token needs a full forward pass through a huge model. That is slow, and it is slow for a frustrating reason: the hardware is mostly sitting idle waiting on memory, not doing useful math.

Speculative decoding exploits that idle time. The recipe is:

1. Let a much smaller, cheaper "draft" model guess the next few tokens (say 3 to 5) on its own.
2. Hand all of those guesses to the big "target" model and have it score them **in a single parallel forward pass**.
3. Keep the guesses the big model agrees with, fix the first one it disagrees with, and move on.

The part that makes this more than a heuristic is the acceptance rule. It is a modified rejection-sampling scheme, so the final output is *exactly* distributed as if the big model had generated it alone. No quality trade-off, no retraining, no change to the architecture. You just get the same text faster.

## What we want to replicate

The goal is the whole paper, not just the headline number. Concretely:

- **The algorithm itself**: the draft-then-verify loop, with the rejection-sampling acceptance step and the resampling from the adjusted distribution when a token is rejected. This should work for greedy decoding (temperature 0) and for sampling (temperature 1), as well as top-k / nucleus style settings.
- **The theory**: the acceptance rate α, the expected number of tokens produced per target-model call, `(1 - α^(γ+1)) / (1 - α)`, and the expected wall-time improvement as a function of α, the draft/target cost ratio c, and the number of guesses γ. I want to check that the measured numbers actually follow these formulas and not just that "it got faster".
- **The correctness claim**: an empirical sanity check that outputs from speculative decoding and plain autoregressive sampling really come from the same distribution (and are token-for-token identical under greedy decoding).
- **The experiments**: the paper's main results table, roughly:
  - T5-XXL (11B) as the target, with T5-small, T5-base and T5-large as drafts.
  - English-to-German translation (WMT) and CNN/DailyMail summarization, both at temperature 0 and temperature 1, at batch size 1.
  - Measuring α for each draft/target pairing, and the resulting speedup, which the paper reports as somewhere in the 2x to 3.5x range.
  - The sweep over γ to see where adding more guesses stops paying off.
- **The extras**: the paper also looks at dialogue-style and language modeling setups with other model sizes, and discusses very cheap draft models such as n-gram models. I'll try to cover those where it's feasible with open models.

## What will probably be different

We don't have TPUs or access to the proprietary models used in the paper (like LaMDA), so some things will have to be swapped:

- Open models stand in for the closed ones. The T5 family is available publicly, so the main experiments should be reproducible nearly as written. For anything else I'll pick the closest open equivalents and say so clearly.
- Absolute timings will be different because the hardware is different. What should carry over is the *shape* of the results: α values, relative speedups, and how they move with γ and model size.
- we'll note where the paper under-specifies a detail (exact checkpoints, decoding hyperparameters, how timing was measured) and what we assumed.

## How we'll judge success

- The acceptance rates we measure land close to the paper's.
- The speedups are in the same ballpark, and agree with the theoretical formula given the measured α and c.
- The distribution check passes: speculative and standard decoding are statistically indistinguishable.
- Everything is reproducible from a single config, so someone else can rerun it.

## Planned layout

Nothing built yet, but roughly:

- a minimal, readable implementation of the algorithm first (small models, easy to step through),
- then the T5 benchmark harness for the paper's tables,
- then analysis notebooks comparing measured results with the theory,
- and a short write-up of what matched, what didn't, and why.

## Notes

We are writing this for my own understanding as much as for reproduction, so I'll keep the implementation simple and well commented over fast and clever.
