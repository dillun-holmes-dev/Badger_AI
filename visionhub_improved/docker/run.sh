#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-visionhub:tensorrt-py3.11.9}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -eq 0 ]; then
  set -- bash
fi

docker run \
  --rm \
  -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${REPO_ROOT}:/workspace/VisionHub" \
  -w /workspace/VisionHub \
  "${IMAGE_NAME}" \
  "$@"
