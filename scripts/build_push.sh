#!/usr/bin/env bash
# Build the server image and push it to the private registry, using config.json.
# Image name: [<registry>/]<namespace>/<image_name>:<image_tag>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="$(python3 "$ROOT/scripts/generate_manifests.py" --get image)"

echo "Building $IMAGE"
# linux/amd64: the cluster nodes are x86 even if you build on an Apple Silicon Mac.
docker build --platform linux/amd64 -t "$IMAGE" "$ROOT/serving"
# Large layers sometimes drop mid-upload ("broken pipe"); finished layers are skipped on retry.
for attempt in 1 2 3 4 5; do
  docker push "$IMAGE" && exit 0
  echo "push attempt $attempt failed, retrying..." >&2
  sleep 5
done
exit 1
