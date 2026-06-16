#!/usr/bin/env bash
# E5-v2 — Noise robustness with corrected random-anchor sampling.
# Noise is applied to each randomly drawn FEM point at every step.
# sigma in {0, 0.01, 0.05, 0.10, 0.20, 0.50} × 5 seeds
# Experiment names: e5_s{SIGMA_TAG}_seed{S}_ntk_random
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E5-v2 — Noise robustness (random anchor)"
echo "================================================================"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env CUDA_MEM_FRACTION="${CUDA_MEM_FRACTION:-0.5}" \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=50000
declare -A SIGMA_TAGS=([0.00]=s000 [0.01]=s001 [0.05]=s005 [0.10]=s010 [0.20]=s020 [0.50]=s050)
SEEDS=(0 1 2 3 4)

for SIGMA in 0.00 0.01 0.05 0.10 0.20 0.50; do
    TAG="${SIGMA_TAGS[$SIGMA]}"
    for S in "${SEEDS[@]}"; do
        echo "--- E5 sigma=${SIGMA} seed=${S} ---"
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=random \
            hybrid.noise_sigma="${SIGMA}" \
            seed="${S}" \
            "experiment.name=e5_${TAG}_seed${S}_ntk_random" || true
    done
done

echo "E5-v2 done."
CONTAINER_EOF
