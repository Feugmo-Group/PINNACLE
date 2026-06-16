#!/usr/bin/env bash
# Verify original-paper accuracy: hybrid NTK training, 50 000 steps,
# empirical anchor (t=76 000 s, E=0.4 V) — same setup as hybrid_training_final.
#
# Usage:
#   ./scripts/run_original_check.sh
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image             (default: nvcr.io/nvidia/pytorch:26.01-py3)
#   STEPS        – max training steps (default: 50000)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
STEPS="${STEPS:-50000}"

echo "================================================================"
echo " Original-check training run"
echo " REPO_DIR     : $REPO_DIR"
echo " DOCKER_IMAGE : $DOCKER_IMAGE"
echo " STEPS        : $STEPS"
echo " anchor_mode  : default (empirical: t=76000 s, E=0.4 V)"
echo "================================================================"

docker run --rm -i --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  -e "STEPS=$STEPS" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

echo "--- original_check: ntk hybrid, ${STEPS} steps, empirical anchor ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat=ntk \
    training.max_steps="${STEPS}" \
    hybrid.use_data=true \
    hybrid.anchor_mode=default \
    experiment.name=original_check

echo "Done. Results in outputs/experiments/original_check/"
CONTAINER_EOF
