#!/usr/bin/env bash
# Single base training run — same as one strategy run in run_e1.sh.
#
# Usage:
#   ./scripts/run_base_training.sh
#   STRAT=brdr ./scripts/run_base_training.sh
#   STEPS=50000 ./scripts/run_base_training.sh
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image             (default: nvcr.io/nvidia/pytorch:26.01-py3)
#   STRAT        – weight strategy: ntk, brdr, uniform, batch_size (default: ntk)
#   STEPS        – max training steps (default: 50000, same as E1)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
STRAT="${STRAT:-ntk}"
STEPS="${STEPS:-50000}"

echo "================================================================"
echo " Base training run (single, mirrors run_e1.sh)"
echo " REPO_DIR     : $REPO_DIR"
echo " DOCKER_IMAGE : $DOCKER_IMAGE"
echo " STRAT        : $STRAT"
echo " STEPS        : $STEPS"
echo "================================================================"

docker run --rm -i --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  -e "STRAT=$STRAT" \
  -e "STEPS=$STEPS" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

echo "--- base training: weight_strat=${STRAT} ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main 

echo "Done. Results in outputs/experiments/base_training_${STRAT}/"
CONTAINER_EOF
