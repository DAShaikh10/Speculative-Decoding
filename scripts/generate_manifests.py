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
    "served_model_name",
    "dtype",
    "torch_cuda",
    "server_name",
    "pvc_name",
    "pvc_size",
    "storage_class",
    "access_mode",
]


def load_config(path: Path = ROOT / "config.json") -> dict:
    """
    Load the configuration from config.json.

    Args:
        path: The path to the config.json file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        SystemExit: If any required key is missing or empty.
    """

    cfg = json.loads(path.read_text())
    missing: list[str] = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        sys.exit(f"config.json is missing/empty: {', '.join(missing)}")

    # Registry layout on our cluster: [<registry>/]<your namespace>/<image>:<tag>
    # The registry is optional; leave it empty for images that don't need one.
    registry: str = cfg.get("registry", "").strip().rstrip("/")
    prefix: str = f"{registry}/" if registry else ""
    cfg["image"] = f"{prefix}{cfg['namespace']}/{cfg['image_name']}:{cfg['image_tag']}"

    return cfg


def main() -> None:
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--templates", "-t", nargs="*", help="Template names (pvc, server); default: all")
    argparser.add_argument("--get", "-g", help="Print a single config/derived value and exit")

    args: argparse.Namespace = argparser.parse_args()
    config: dict = load_config()

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
