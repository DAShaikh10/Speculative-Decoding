#!/bin/sh
set -eu

export UV_PROJECT_ENVIRONMENT=/cache/venv
export UV_CACHE_DIR=/cache/uv  # same filesystem as the venv, so uv hardlinks instead of copying.
export TMPDIR=/cache/tmp
# Compiled kernels of torch.compile are cached here, so a later run doesn't compile the decoding steps again.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/triton}"
# TORCH_CUDA selects the torch build via the pyproject extra of the same name: cu128 for
# Blackwell / Hopper / Ampere, cu126 for older GPUs such as V100.
TORCH_CUDA="${TORCH_CUDA:-cu128}"
STAMP="$UV_PROJECT_ENVIRONMENT/.synced-$(md5sum /app/uv.lock | cut -d' ' -f1)-$TORCH_CUDA"

if [ ! -f "$STAMP" ]; then
  echo "Syncing dependencies ($TORCH_CUDA) into $UV_PROJECT_ENVIRONMENT (first start, takes a few minutes) ..."
  mkdir -p /cache/tmp
  uv sync --project /app --locked --no-dev --extra "$TORCH_CUDA" --no-install-project
  touch "$STAMP"
fi

export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$VIRTUAL_ENV/bin:$PATH"

exec "$@"
