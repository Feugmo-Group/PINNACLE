#!/usr/bin/env bash
# E1-v2 — NTK ablation with corrected random-anchor FEM sampling.
# Re-runs the 3-strategy ablation (ntk / uniform / batch_size) using
# anchor_mode=random (one random FEM point per step) which reproduces
# the published <2.2% errors. Previous E1 runs used a fixed empirical
# anchor and produced wrong-branch convergence.
#
# Output: outputs/experiments/e1_ablation_{ntk,uniform,batch_size}_v2/
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E1-v2 — Loss-weighting ablation (random anchor)"
echo " REPO_DIR : $REPO_DIR"
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

for STRAT in ntk uniform batch_size; do
    echo "--- E1-v2 strategy=${STRAT} ---"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat="${STRAT}" \
        training.max_steps="${STEPS}" \
        hybrid.anchor_mode=random \
        "experiment.name=e1_ablation_${STRAT}_v2" || true
done

echo "E1-v2 done. Results in outputs/experiments/e1_ablation_*_v2/"
CONTAINER_EOF
