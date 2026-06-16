#!/usr/bin/env bash
# W7 — Multi-seed Table I: 3 independent seeds of the hybrid NTK baseline.
# Runs seeds 1, 2, 3 (seed 0 == the main paper result already exists).
# Paper change: replace single-number Table I with mean +/- sigma across seeds.
#
# ~3 GPU-hours total (1 h per seed).
#
# Optional env vars:
#   REPO_DIR     – repo root (default: auto-detected)
#   DOCKER_IMAGE – container image (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " W7 — Multi-seed Table I (seeds 1, 2, 3; seed 0 already exists)"
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

for SEED in 1 2 3; do
    echo "--- W7 hybrid NTK seed=${SEED} ---"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk \
        training.max_steps="${STEPS}" \
        precision=float64 \
        seed="${SEED}" \
        experiment.name="w7_hybrid_seed${SEED}" || true
done

echo ""
echo "W7 done. Results in outputs/experiments/w7_hybrid_seed{1,2,3}/"
echo "Aggregate final film-thickness errors across seeds 0-3 for Table I mean+/-sigma."
CONTAINER_EOF
