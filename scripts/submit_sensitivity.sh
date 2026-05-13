#!/usr/bin/env bash
# Sensitivity analysis — sweeps random seeds for FEM anchor-point selection.
# Mirrors pinnacle/submit_sensitivity.sh (SLURM original) but runs in Docker.
# Calls pinnacle.main directly with hybrid.random_seed, one run per seed.
#
# Usage:
#   ./scripts/submit_sensitivity.sh                   # defaults: seeds 42-51, ntk, 20000 steps
#   ./scripts/submit_sensitivity.sh 42 51             # seeds 42..51 inclusive
#   ./scripts/submit_sensitivity.sh 42 46 brdr        # seeds 42..46, brdr weighting
#   STEPS=50000 ./scripts/submit_sensitivity.sh       # override step count
#
# Positional args (all optional):
#   $1  SEED_START  first seed (default: 42)
#   $2  SEED_END    last seed  (default: 51)
#   $3  STRAT       weight_strat (default: ntk)
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image             (default: nvcr.io/nvidia/pytorch:26.01-py3)
#   STEPS        – training steps per run      (default: 20000)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

SEED_START="${1:-42}"
SEED_END="${2:-51}"
STRAT="${3:-ntk}"
STEPS="${STEPS:-20000}"

echo "================================================================"
echo " Sensitivity analysis — hybrid PINN anchor-seed sweep"
echo " REPO_DIR     : $REPO_DIR"
echo " DOCKER_IMAGE : $DOCKER_IMAGE"
echo " SEEDS        : ${SEED_START}..${SEED_END}"
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
  -e "SEED_START=$SEED_START" \
  -e "SEED_END=$SEED_END" \
  -e "STRAT=$STRAT" \
  -e "STEPS=$STEPS" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

echo "================================================================"
echo " Sensitivity: seeds ${SEED_START}..${SEED_END}, strat=${STRAT}, steps=${STEPS}"
echo "================================================================"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    echo ""
    echo "--- seed=${SEED} ---"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat="${STRAT}" \
        training.max_steps="${STEPS}" \
        precision=float64 \
        hybrid.use_data=true \
        hybrid.fem_batch_size=1 \
        hybrid.fem_data_dir="pinnacle/FEM" \
        hybrid.anchor_seed="${SEED}" \
        "experiment.name=sensitivity_${STRAT}_seed${SEED}"
    echo "Completed seed ${SEED}"
done

echo ""
echo "================================================================"
echo " All seeds done. Results in outputs/experiments/sensitivity_${STRAT}_seed*/"
echo "================================================================"
CONTAINER_EOF
