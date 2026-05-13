#!/usr/bin/env bash
# Run PINNACLE training inside the standard NVIDIA PyTorch container.
#
# Usage:
#   ./scripts/docker_train.sh                              # default config
#   ./scripts/docker_train.sh training.max_steps=100      # any Hydra overrides
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image to use      (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "REPO_DIR     : $REPO_DIR"
echo "DOCKER_IMAGE : $DOCKER_IMAGE"
echo

# Extra positional args are forwarded verbatim as Hydra overrides.
# The canonical bash -c / -- pattern keeps quoting correct.
docker run --rm --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  "$DOCKER_IMAGE" \
  bash -c '
    pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas -q &&
    python -m pinnacle.main "$@"
  ' -- "$@"
