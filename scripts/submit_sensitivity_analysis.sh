#!/usr/bin/env bash
# Sensitivity analysis with automatic FEM metric aggregation.
# Calls pinnacle/sensitivity_analysis.py which trains N seeds then computes
# RMSE/MAE/R² vs FEM ground truth and writes aggregate_statistics.json +
# sensitivity_report.txt.
#
# Usage:
#   ./scripts/submit_sensitivity_analysis.sh              # defaults: 10 seeds from 42, ntk
#   ./scripts/submit_sensitivity_analysis.sh 10           # 10 seeds
#   ./scripts/submit_sensitivity_analysis.sh 10 0         # 10 seeds starting from seed 0
#   ./scripts/submit_sensitivity_analysis.sh 10 0 brdr    # brdr weighting
#   STEPS=50000 ./scripts/submit_sensitivity_analysis.sh
#
# Positional args (all optional):
#   $1  NUM_SEEDS   number of seeds (default: 10)
#   $2  START_SEED  first seed      (default: 42)
#   $3  STRAT       weight_strat    (default: ntk)
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image             (default: nvcr.io/nvidia/pytorch:26.01-py3)
#   STEPS        – training steps per seed     (default: 20000)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

NUM_SEEDS="${1:-10}"
START_SEED="${2:-42}"
STRAT="${3:-ntk}"
STEPS="${STEPS:-20000}"

echo "================================================================"
echo " Sensitivity analysis (train + FEM metrics + aggregation)"
echo " REPO_DIR     : $REPO_DIR"
echo " DOCKER_IMAGE : $DOCKER_IMAGE"
echo " NUM_SEEDS    : $NUM_SEEDS"
echo " START_SEED   : $START_SEED"
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
  -e "NUM_SEEDS=$NUM_SEEDS" \
  -e "START_SEED=$START_SEED" \
  -e "STRAT=$STRAT" \
  -e "STEPS=$STEPS" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

echo "================================================================"
echo " Sensitivity: ${NUM_SEEDS} seeds from ${START_SEED}, strat=${STRAT}, steps=${STEPS}"
echo "================================================================"

PYTHONPATH=/app/pinnacle python -m pinnacle.sensitivity_analysis \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    hybrid.use_data=true \
    hybrid.fem_batch_size=1 \
    hybrid.fem_data_dir="pinnacle/FEM" \
    hybrid.noise_sigma=0.0 \
    hybrid.n_data_points=1 \
    "experiment.name=sensitivity_${STRAT}" \
    "+sensitivity.num_seeds=${NUM_SEEDS}" \
    "+sensitivity.start_seed=${START_SEED}" \
    "+sensitivity.output_base=outputs/experiments/sensitivity_analysis_${STRAT}"

echo ""
echo "================================================================"
echo " Done. Results in outputs/experiments/sensitivity_analysis_${STRAT}_*/"
echo "   aggregate_statistics.json   — mean/std/min/max per voltage"
echo "   sensitivity_report.txt      — human-readable summary"
echo "   seed_*/loss_history.json    — per-seed loss curves"
echo "   seed_*/fem_comparison_metrics.json  — per-seed FEM metrics"
echo "================================================================"
CONTAINER_EOF
