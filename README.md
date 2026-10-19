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

## Models

The paper used T5 v1.1 checkpoints fine-tuned per task (target T5-XXL; drafts T5-small, T5-base, T5-large). We use the original `google-t5` checkpoints instead: they were pretrained on a multi-task mixture that includes WMT EnDe and CNN/DM, so they work out of the box with the `translate English to German:` and `summarize:` prefixes, and the whole family shares one tokenizer.

| Model | Params | Download (fp32) | In bf16/fp16 | Link |
|---|---|---|---|---|
| t5-small | 60M | 0.24 GB | ~0.12 GB | https://huggingface.co/google-t5/t5-small |
| t5-base | 220M | 0.89 GB | ~0.45 GB | https://huggingface.co/google-t5/t5-base |
| t5-large | 770M | 2.95 GB | ~1.5 GB | https://huggingface.co/google-t5/t5-large |
| t5-3b | 3B | 11.41 GB | ~5.7 GB | https://huggingface.co/google-t5/t5-3b |
| t5-11b (the "XXL") | 11B | 45.23 GB | ~22.6 GB | https://huggingface.co/google-t5/t5-11b |

Notes:

- Sizes are for a single weight file. The Hugging Face repo pages show a larger total (about 34 GB for t5-3b, about 90 GB for t5-11b) because each repo stores the same weights in several formats (safetensors, PyTorch `.bin`, TensorFlow `.h5`). `from_pretrained` downloads only one of them.
- t5-3b is the cheaper stand-in target for development; t5-11b is the faithful target.
- On V100s use fp16 (no native bf16), and watch for fp16 overflow/NaNs in the larger T5 models. A100s can use bf16.


## Hosting a model on the cluster

`serving/` has a small FastAPI server (`/health`, `/info`, `/generate`) that serves one T5 checkpoint, plus a Dockerfile. Everyone has their own Kubernetes namespace and registry path, so all per-user settings live in `config.json`; the manifests in `k8s/*.yaml.tmpl` are templates filled in from it.

1. Copy `config.json.example` to `config.json` (it is git-ignored, so it stays yours) and set `namespace` to yours (the image becomes `registry.datexis.com/<namespace>/<image_name>:<image_tag>`) and `gpu` to the node label you want (`v100`, `a100`, `b200`, ...).
2. Build and push the image: `scripts/build_push.sh`
3. Deploy: `scripts/apply.sh` (add `--dry-run` to check first). It generates `k8s/*.yaml` from `k8s/*.yaml.tmpl` and `config.json`, then runs `kubectl apply` on them. Use `python3 scripts/generate_manifests.py` to only generate the YAML. `registry` in `config.json` is optional; leave it empty if the image doesn't need one.
5. Try it: `kubectl -n <namespace> port-forward svc/t5-small-server 8000:8000`, then

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"text": "The house is wonderful.", "task": "translate_en_de"}'
```

The image is slim (just the server code). On first start the container installs its Python packages (torch etc.) into the PVC under `/cache/pylibs` and downloads the weights into `/cache/hf`; restarts reuse both.

**GPU type and `torch_cuda`:** `torch_cuda` in `config.json` selects the torch build that gets installed, and it has to match the GPU you pick in `gpu`:

| GPU (`gpu`) | `torch_cuda` |
|---|---|
| `b200`, `h100`, `h200`, `a100`, `l4`, `l40` | `cu128` (default; tested on B200) |
| `v100` | `cu126` (**not tested yet**) |

- **V100 (Volta):** the cu128 wheels have no kernels for V100, so the server would start but fail on the first `/generate` with `CUDA error: no kernel image is available`. Set `"torch_cuda": "cu126"` and `"gpu": "v100"`, run `scripts/apply.sh`. Also keep `dtype` at `float32`, since T5 can overflow in fp16.
- **B200 (Blackwell):** needs cu128. The default PyPI torch wheel is the CUDA 12.6 build and has no Blackwell kernels, which is why we install from PyTorch's own index.
- Changing `torch_cuda` makes the pod reinstall its packages on the next start (a few minutes).

## Datasets

The paper's T5 experiments use two tasks: English to German translation (WMT EnDe) and CNN/DailyMail summarization. It's GPT-like experiment uses lm1b. T5's own code loads the first two from TensorFlow Datasets (TFDS). WMT can be loaded at T5's pinned version. CNN/DM cannot: T5 pins version 3.1.0, but current TFDS (checked on 4.9.10) refuses to load it and offers only 3.4.0, so use 3.4.0. Sizes below are the ones TFDS lists, in its own units (GiB/MiB).

| Dataset | TFDS id (pinned) | Splits (examples) | Download | Prepared on disk | Link |
|---|---|---|---|---|---|
| WMT En-De (T2T config) | `wmt_t2t_translate/de-en:1.0.0` | train 4,592,289 / validation 3,000 / test 3,003 | 1.61 GiB | 1.39 GiB | https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate |
| CNN/DailyMail (non-anonymized) | `cnn_dailymail:3.4.0` | train 287,113 / validation 13,368 / test 11,490 | 558.32 MiB | 1.29 GiB | https://www.tensorflow.org/datasets/catalog/cnn_dailymail |

```python
import tensorflow_datasets as tfds

wmt = tfds.load("wmt_t2t_translate/de-en:1.0.0", split="test")
cnn = tfds.load("cnn_dailymail:3.4.0", split="test")
```

Notes:

- **Pins:** T5's `tasks.py` uses `wmt_t2t_translate/de-en:1.0.0` (task `wmt_t2t_ende_v003`) and `cnn_dailymail:3.1.0`. Current TFDS loads the WMT pin as-is, but for CNN/DM it rejects 3.1.0 with "cannot be loaded at version 3.1.0, only: 3.4.0". Use 3.4.0. TFDS's release notes say the changes after 3.1.0 only remove an extra space before the added sentence period (which should not affect ROUGE) and add `publisher` and `id` features, so the difference from the paper's setup should be negligible.
- **What the T5 paper says (Section 2.3):** all data comes from TFDS. WMT En-De uses the same training data as Vaswani et al. 2017 (News Commentary v13, Common Crawl, Europarl v7) with newstest2013 as validation. CNN/DM is the non-anonymized version from See et al. 2017.
- **WMT split names:** the TFDS page gives only the counts. The T2T config's script lists newstest2013 as validation and newstest2014 as test, and the 3,000 and 3,003 counts are consistent with that.


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
