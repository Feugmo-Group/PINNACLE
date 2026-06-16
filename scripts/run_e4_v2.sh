#!/usr/bin/env bash
# E4-v2 — Data efficiency with corrected random-anchor sampling.
#
# Main: N in {0,1,2,3,5,10,20,50} data points drawn randomly per step × 5 seeds
#   Experiment names: e4_N{N}_seed{S}_ntk_random
#
# Fixed-anchor comparison: N in {1,5} × 5 seeds
#   Experiment names: e4_N{N}_seed{S}_ntk_fixed
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E4-v2 — Data efficiency (random anchor)"
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
N_VALUES=(0 1 2 3 5 10 20 50)
SEEDS=(0 1 2 3 4)

echo "=== E4-v2 main: random anchor ==="
for N in "${N_VALUES[@]}"; do
    for S in "${SEEDS[@]}"; do
        echo "--- E4 N=${N} seed=${S} ---"
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=random \
            hybrid.n_data_points="${N}" \
            seed="${S}" \
            "experiment.name=e4_N${N}_seed${S}_ntk_random" || true
    done
done

echo "=== E4-v2 comparison: fixed anchor N=1 and N=5 ==="
for N in 1 5; do
    for S in "${SEEDS[@]}"; do
        echo "--- E4 fixed N=${N} seed=${S} ---"
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=seed \
            hybrid.anchor_seed="${S}" \
            hybrid.n_data_points="${N}" \
            seed="${S}" \
            "experiment.name=e4_N${N}_seed${S}_ntk_fixed" || true
    done
done

echo "E4-v2 done."
CONTAINER_EOF
