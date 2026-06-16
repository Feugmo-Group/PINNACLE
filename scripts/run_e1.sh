#!/usr/bin/env bash
# E1 — Ablation: NTK vs uniform vs batch_size (BRDR dropped: not implemented)
# Outputs: timing.json, loss_landscape_*.png, training_losses.png per run
# Paper placement: Table II + companion loss-curve figure (Sec III.D, V.B)
#
# Runs all 4 strategies in a single Docker container (install once).
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image to use      (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E1 — Loss-weighting ablation study (4 strategies × 50k steps)"
echo " REPO_DIR     : $REPO_DIR"
echo " DOCKER_IMAGE : $DOCKER_IMAGE"
echo "================================================================"

docker run --rm -i --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  --env CUDA_MEM_FRACTION="${CUDA_MEM_FRACTION:-0.5}" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=50000

echo "================================================================"
echo " E1 — Loss-weighting ablation study (${STEPS} steps each)"
echo "================================================================"

for STRAT in ntk uniform batch_size; do
    echo ""
    echo "--- E1: weight_strat=${STRAT} ---"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat="${STRAT}" \
        training.max_steps="${STEPS}" \
        "experiment.name=e1_ablation_${STRAT}"
done

echo ""
echo "================================================================"
echo " E1 done. Results in outputs/experiments/e1_ablation_*/"
echo " Compare timing.json across strategies for Table II."
echo " Compare loss_landscape_components.png for landscape comparison."
echo "================================================================"
CONTAINER_EOF
