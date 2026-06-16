#!/usr/bin/env bash
# W7 seeds 3-4 (fresh) — Table I reproducibility, same random-anchor NTK recipe
# as the converged seed 1 (lr=1e-3, 50k steps, float64). Names match existing
# w7_hybrid_seed{N}_random_anchor dirs.
set -uo pipefail
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
docker run --rm -i --init --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$REPO_DIR:/app" --workdir="/app" \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -uo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
for SEED in 3 4; do
    echo "--- W7 hybrid NTK seed=${SEED} (random anchor) ---"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk training.max_steps=50000 \
        precision=float64 seed="${SEED}" \
        experiment.name="w7_hybrid_seed${SEED}_random_anchor" || true
done
echo "W7 seeds 3-4 done."
CONTAINER_EOF
