#!/usr/bin/env bash
# E1-BRDR — Add BRDR strategy to the E1 ablation table.
# Runs two configurations:
#   brdr_random : random FEM anchor (matches E1-v2 setup)
#   brdr_nofem  : no FEM data (pure PINN, matches E1 NTK baseline)
#
# Output: outputs/experiments/e1_ablation_brdr_{random,nofem}/
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E1-BRDR — BRDR weighting ablation"
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

wall_start() { date +%s; }
wall_end() {
    local t0=$1 name=$2
    local t1; t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    echo "WALL_TIME ${name} ${elapsed}s ($(( elapsed / 60 ))m $(( elapsed % 60 ))s)"
    # Append to timing.json if it exists
    local latest
    latest=$(ls -td /app/outputs/experiments/${name}/*/timing.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        python3 -c "
import json, sys
p = '$latest'
with open(p) as f: d = json.load(f)
d['wall_clock_s'] = $elapsed
with open(p, 'w') as f: json.dump(d, f, indent=2)
print('  wall_clock_s written to', p)
"
    fi
}

echo "--- E1-BRDR: with FEM (random anchor) ---"
T0=$(wall_start)
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat=brdr \
    training.max_steps="${STEPS}" \
    hybrid.anchor_mode=random \
    experiment.name=e1_ablation_brdr_random || true
wall_end "$T0" e1_ablation_brdr_random

echo "--- E1-BRDR: without FEM (pure PINN) ---"
T0=$(wall_start)
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat=brdr \
    training.max_steps="${STEPS}" \
    hybrid.use_data=false \
    experiment.name=e1_ablation_brdr_nofem || true
wall_end "$T0" e1_ablation_brdr_nofem

echo "E1-BRDR done. Results in outputs/experiments/e1_ablation_brdr_*/"
CONTAINER_EOF
