#!/usr/bin/env bash
# E3-v2 — Anchor robustness with corrected random-anchor sampling.
#
# Part A (main): 30 seeds × anchor_mode=random + seed=N
#   Tests: do different random FEM sequences give similar errors?
#   Experiment names: e3a_seed{0..29}_ntk_random
#
# Part B (systematic): 5 (t,E) positions × anchor_mode=random
#   Experiment names: e3b_{label}_ntk_random
#
# Part C (fixed-anchor comparison, 10 seeds):
#   Uses anchor_mode=seed (same fixed point each step).
#   Shows that fixed single anchor fails → motivates random.
#   Experiment names: e3c_seed{0..9}_ntk_fixed
#
# Optional: pass "A", "B", "C", or "AB" to run only those parts.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
PART="${1:-ABC}"

echo "================================================================"
echo " E3-v2 — Anchor robustness (part(s): $PART)"
echo "================================================================"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env CUDA_MEM_FRACTION="${CUDA_MEM_FRACTION:-0.5}" \
  "$DOCKER_IMAGE" bash -s << CONTAINER_EOF
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
PART="${PART}"
STEPS=50000

if [[ "\$PART" == *A* ]]; then
    echo "=== E3a-v2: 30 random-sequence seeds ==="
    for SEED in \$(seq 0 29); do
        echo "--- E3a seed=\${SEED} ---"
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \\
            training.weight_strat=ntk \\
            training.max_steps=\${STEPS} \\
            hybrid.anchor_mode=random \\
            seed="\${SEED}" \\
            "experiment.name=e3a_seed\${SEED}_ntk_random" || true
    done
fi

if [[ "\$PART" == *B* ]]; then
    echo "=== E3b-v2: 5 systematic positions (random sampling) ==="
    declare -A POSITIONS=(
        [early_low]="2000 0.1"
        [mid_low]="200000 0.1"
        [late_low]="380000 0.1"
        [mid_mid]="200000 1.0"
        [mid_high]="200000 1.8"
    )
    SEED_IDX=0
    for LABEL in early_low mid_low late_low mid_mid mid_high; do
        read T_VAL E_VAL <<< "\${POSITIONS[\$LABEL]}"
        echo "--- E3b \${LABEL} ---"
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \\
            training.weight_strat=ntk \\
            training.max_steps=\${STEPS} \\
            hybrid.anchor_mode=random \\
            seed=\${SEED_IDX} \\
            "experiment.name=e3b_\${LABEL}_ntk_random" || true
        SEED_IDX=\$((SEED_IDX+1))
    done
fi

if [[ "\$PART" == *C* ]]; then
    echo "=== E3c-v2: 10 fixed-anchor seeds (comparison) ==="
    for SEED in \$(seq 0 9); do
        echo "--- E3c fixed seed=\${SEED} ---"
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \\
            training.weight_strat=ntk \\
            training.max_steps=\${STEPS} \\
            hybrid.anchor_mode=seed \\
            hybrid.anchor_seed="\${SEED}" \\
            "experiment.name=e3c_seed\${SEED}_ntk_fixed" || true
    done
fi

echo "E3-v2 done."
CONTAINER_EOF
