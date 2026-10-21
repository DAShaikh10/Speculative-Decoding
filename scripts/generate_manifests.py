#!/usr/bin/env python3
"""
Generate `k8s/*.yaml` from `k8s/*.yaml.tmpl` using the values from `config.json`.

Every value (namespace, image, GPU, ...) comes from `config.json`. The generated .yaml
files are git-ignored since they contain environment-specific values.
    python3 scripts/generate_manifests.py                         # all templates -> k8s/<name>.yaml
    python3 scripts/generate_manifests.py --templates pvc server  # only these templates
    python3 scripts/generate_manifests.py --get image             # print one config/derived value
"""

import argparse
import json
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent

# 'image_pull_secret' and 'registry' are not strictly checked but are required for DATEXIS registry access.
# Please check that they are set in config.json if you are using the DATEXIS registry.
REQUIRED: list[str] = [
    "namespace",
    "image_name",
    "image_tag",
    "gpu",
    "draft_model_name",
    "target_model_name",
    "served_model_name",
    "dtype",
    "torch_cuda",
    "server_name",
    "pvc_name",
    "pvc_size",
    "storage_class",
    "access_mode",
]


# Values of the job templates that config.json does not have to set. `--set key=value` overrides any value.
DEFAULTS: dict[str, str] = {
    "job_name": "overfitclub-speculative-decoding-dialogue",
    "gamma": "5",
    "temperature": "0",
    "num_examples": "100",
    "memory_limit": "96Gi",  # The weights go straight to the GPU, but the safetensors files are memory mapped.
    "gpu_product": "",  # The `nvidia.com/gpu.product` node label, to tell GPUs of one `gpu` label apart (40GB/80GB A100).
}


def load_config(path: Path = ROOT / "config.json", overrides: dict[str, str] | None = None) -> dict:
    """
    Load the configuration from config.json.

    Args:
        path: The path to the config.json file.
        overrides: Values that replace the ones of config.json and the defaults.

    Returns:
        A dictionary containing the configuration.

    Raises:
        SystemExit: If any required key is missing or empty.
    """

    cfg = {**DEFAULTS, **json.loads(path.read_text()), **(overrides or {})}
    missing: list[str] = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        sys.exit(f"config.json is missing/empty: {', '.join(missing)}")

    # Registry layout on our cluster: [<registry>/]<your namespace>/<image>:<tag>
    # The registry is optional; leave it empty for images that don't need one.
    registry: str = cfg.get("registry", "").strip().rstrip("/")
    prefix: str = f"{registry}/" if registry else ""
    cfg["image"] = f"{prefix}{cfg['namespace']}/{cfg['image_name']}:{cfg['image_tag']}"
    # The image of the evaluation jobs, which is built from the Dockerfile in the repository root. It lives on Docker Hub
    # when `dockerhub_namespace` is set, and in the same registry as the server image otherwise.
    eval_image_name: str = "overfitclub-speculative-decoding:latest"
    cfg["eval_image"] = (
        f"{cfg['dockerhub_namespace']}/{eval_image_name}"
        if cfg.get("dockerhub_namespace")
        else f"{prefix}{cfg['namespace']}/{eval_image_name}"
    )
    # An empty line in the node selector when no GPU product is asked for.
    cfg["gpu_product_selector"] = f"nvidia.com/gpu.product: {cfg['gpu_product']}" if cfg["gpu_product"] else ""

    return cfg


def main() -> None:
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--templates", "-t", nargs="*", help="Template names (pvc, server); default: all")
    argparser.add_argument("--get", "-g", help="Print a single config/derived value and exit")
    argparser.add_argument(
        "--set", "-s", nargs="*", default=[], metavar="KEY=VALUE", help="Override values of config.json for this run"
    )

    args: argparse.Namespace = argparser.parse_args()
    config: dict = load_config(overrides=dict(item.split("=", 1) for item in args.set))

    if args.get:
        print(config[args.get])
        return

    k8s: Path = ROOT / "k8s"
    paths: list[Path] = sorted(k8s.glob("*.yaml.tmpl"))
    if args.templates:
        paths = [k8s / f"{name}.yaml.tmpl" for name in args.templates]
    for p in paths:
        out: Path = p.with_suffix("")
        out.write_text(Template(p.read_text()).substitute(config).strip() + "\n")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
