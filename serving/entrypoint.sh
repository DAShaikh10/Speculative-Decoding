#!/bin/sh
# Install the Python dependencies into the PVC on first start, then run the server.
# The install is keyed on requirements.txt + the torch CUDA build, so it is skipped on
# restarts and redone automatically when either changes.
#
# TORCH_CUDA picks the torch wheel: cu128 for Blackwell (B200) / Hopper / Ampere,
# cu126 for older GPUs such as V100 (cu128 builds dropped Volta).
set -eu

TORCH_VERSION=2.7.1
TORCH_CUDA="${TORCH_CUDA:-cu128}"
LIBS=/cache/pylibs
STAMP="$LIBS/.installed-$(cat /app/requirements.txt | md5sum | cut -d' ' -f1)-$TORCH_CUDA"

if [ ! -f "$STAMP" ]; then
  echo "Installing dependencies (torch $TORCH_VERSION+$TORCH_CUDA) into $LIBS (first start, takes a few minutes) ..."
  rm -rf "$LIBS"
  mkdir -p "$LIBS" /cache/tmp
  export TMPDIR=/cache/tmp
  pip install --no-cache-dir --target "$LIBS" "torch==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
    --extra-index-url https://pypi.org/simple
  pip install --no-cache-dir --target "$LIBS" -r /app/requirements.txt
  touch "$STAMP"
fi

export PYTHONPATH="$LIBS:${PYTHONPATH:-}"
exec python3 /app/server.py
