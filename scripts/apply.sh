#!/usr/bin/env bash
# Generate k8s/*.yaml from the templates and config.json, then kubectl apply them.
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
python3 "$ROOT/scripts/generate_manifests.py" ${TEMPLATES[@]+"${TEMPLATES[@]}"}
if [ ${#TEMPLATES[@]} -eq 0 ]; then
  TEMPLATES=(pvc server)
fi
for t in "${TEMPLATES[@]}"; do
  kubectl apply ${DRY[@]+"${DRY[@]}"} -f "$ROOT/k8s/$t.yaml"
done
