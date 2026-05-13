#!/usr/bin/env bash
# E5 — Noise-robustness study: noise_sigma in {0, 0.01, 0.05, 0.10, 0.20, 0.50}
# 5 seeds × 6 sigma levels × NTK only = 30 runs (50k steps each).
# Paper placement: Sec. IV.D or new Sec. IV.F "Noise Robustness"
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
echo " E5 — Noise-robustness sweep (6 sigma × 5 seeds × 2 strategies = 30 runs)"
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
SIGMAS=(0.0 0.01 0.05 0.10 0.20 0.50)
SEEDS=(0 1 2 3 4)

echo "================================================================"
echo " E5 — Noise-robustness sweep (${STEPS} steps each)"
echo "================================================================"

for SIGMA in "${SIGMAS[@]}"; do
    # Build a filename-safe label: 0.05 → s005
    LABEL="s$(echo "$SIGMA" | tr -d '.')"
    for SEED in "${SEEDS[@]}"; do
            echo "--- E5: sigma=${SIGMA} seed=${SEED} strat=ntk ---"
            PYTHONPATH=/app/pinnacle python -m pinnacle.main \
                training.weight_strat="ntk" \
                training.max_steps="${STEPS}" \
                hybrid.anchor_mode=seed \
                hybrid.noise_sigma="${SIGMA}" \
                hybrid.anchor_seed="${SEED}" \
                "experiment.name=e5_${LABEL}_seed${SEED}_ntk"
    done
done

echo ""
echo "================================================================"
echo " E5 done. Results in outputs/experiments/e5_s*/"
echo " Key message: error vs sigma — physics denoises up to ~10% noise."
echo "================================================================"
CONTAINER_EOF
