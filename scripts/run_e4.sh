#!/usr/bin/env bash
# E4 — Data-count sweep: N_data in {0,1,2,3,5,10,20,50}
# 5 seeds × 8 N values × NTK only = 40 runs (50k steps each).
# Paper placement: new Sec. IV.E "Data Efficiency Analysis"
#
# All sub-runs execute inside a single Docker container (install once).
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image to use      (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E4 — Data-count sweep (8 N values × 5 seeds × 2 strategies = 40 runs)"
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
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=50000
N_VALUES=(0 1 2 3 5 10 20 50)
SEEDS=(0 1 2 3 4)

echo "================================================================"
echo " E4 — Data-count sweep (${STEPS} steps each)"
echo "================================================================"

for N in "${N_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
            echo "--- E4: N=${N} seed=${SEED} strat=ntk ---"
            PYTHONPATH=/app/pinnacle python -m pinnacle.main \
                training.weight_strat="ntk" \
                training.max_steps="${STEPS}" \
                hybrid.n_data_points="${N}" \
                hybrid.anchor_seed="${SEED}" \
                "experiment.name=e4_N${N}_seed${SEED}_ntk"
    done
done

echo ""
echo "================================================================"
echo " E4 done. Results in outputs/experiments/e4_N*/"
echo " Aggregate relative errors by N to build data-efficiency figure."
echo "================================================================"
CONTAINER_EOF
