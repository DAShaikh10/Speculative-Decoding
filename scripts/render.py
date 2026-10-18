#!/usr/bin/env python3
"""Render k8s/*.yaml.tmpl with the values from config.json and print the YAML to stdout.

Nothing is written to disk, so no manifest with a hardcoded namespace exists; every
value (namespace, image, GPU, ...) comes from config.json.
    python3 scripts/render.py                  # all templates, separated by ---
    python3 scripts/render.py pvc server       # only these templates
    python3 scripts/render.py --get image      # print one config/derived value
"""
import argparse
import json
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["namespace", "registry", "image_name", "image_tag", "gpu", "model_name",
            "served_model_name", "dtype", "torch_cuda", "server_name", "pvc_name", "pvc_size",
            "storage_class", "access_mode", "image_pull_secret"]


def load_config(path: Path = ROOT / "config.json") -> dict:
    cfg = json.loads(path.read_text())
    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        sys.exit(f"config.json is missing/empty: {', '.join(missing)}")
    # Registry layout on our cluster: <registry>/<your namespace>/<image>:<tag>
    cfg["image"] = f"{cfg['registry']}/{cfg['namespace']}/{cfg['image_name']}:{cfg['image_tag']}"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("templates", nargs="*", help="template names (pvc, server); default: all")
    ap.add_argument("--get", help="print a single config/derived value and exit")
    args = ap.parse_args()
    cfg = load_config()

    if args.get:
        print(cfg[args.get])
        return

    paths = sorted((ROOT / "k8s").glob("*.yaml.tmpl"))
    if args.templates:
        paths = [ROOT / "k8s" / f"{name}.yaml.tmpl" for name in args.templates]
    docs = [Template(p.read_text()).substitute(cfg) for p in paths]  # KeyError on typos
    print("\n---\n".join(d.strip() for d in docs))


if __name__ == "__main__":
    main()
