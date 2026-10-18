#!/usr/bin/env bash
# Render the k8s templates from config.json and apply them (nothing is written to disk).
#   scripts/apply.sh            # PVC + server
#   scripts/apply.sh server     # only the Deployment + Service
#   scripts/apply.sh --dry-run  # server-side dry run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY=()
TEMPLATES=()
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY=(--dry-run=server); else TEMPLATES+=("$a"); fi
done
python3 "$ROOT/scripts/render.py" ${TEMPLATES[@]+"${TEMPLATES[@]}"} | kubectl apply ${DRY[@]+"${DRY[@]}"} -f -
