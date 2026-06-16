#!/usr/bin/env bash
# run_all_v2.sh — Master script: E1 → E3 → E4 → E5 (all with random anchor).
# Runs all experiments sequentially inside one Docker session to avoid
# repeated image-pull and pip-install overhead.
# W7 seeds 1-4 run separately via run_w7.sh (already in progress).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
LOG="$REPO_DIR/revision_v2.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Revision v2 started (random-anchor FEM sampling) ==="
log "REPO_DIR=$REPO_DIR"

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
REPO=/app

# ── E1-v2: 3 strategies ──────────────────────────────────────────────────────
echo "======== E1-v2: loss-weighting ablation ========"
for STRAT in ntk uniform batch_size; do
    echo "--- E1 ${STRAT} ---"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat="${STRAT}" \
        training.max_steps="${STEPS}" \
        hybrid.anchor_mode=random \
        "experiment.name=e1_ablation_${STRAT}_v2" || true
done

# ── E3a-v2: 30 seeds (random sequence) ──────────────────────────────────────
echo "======== E3a-v2: 30-seed anchor robustness ========"
for SEED in $(seq 0 29); do
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk \
        training.max_steps="${STEPS}" \
        hybrid.anchor_mode=random \
        seed="${SEED}" \
        "experiment.name=e3a_seed${SEED}_ntk_random" || true
done

# ── E3b-v2: 5 systematic positions (random sampling) ────────────────────────
echo "======== E3b-v2: systematic positions ========"
LABELS=(early_low mid_low late_low mid_mid mid_high)
SEED_IDX=0
for LABEL in "${LABELS[@]}"; do
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk \
        training.max_steps="${STEPS}" \
        hybrid.anchor_mode=random \
        seed="${SEED_IDX}" \
        "experiment.name=e3b_${LABEL}_ntk_random" || true
    SEED_IDX=$((SEED_IDX+1))
done

# ── E3c-v2: 10 fixed-anchor seeds (comparison) ───────────────────────────────
echo "======== E3c-v2: fixed-anchor comparison ========"
for SEED in $(seq 0 9); do
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk \
        training.max_steps="${STEPS}" \
        hybrid.anchor_mode=seed \
        hybrid.anchor_seed="${SEED}" \
        "experiment.name=e3c_seed${SEED}_ntk_fixed" || true
done

# ── E4-v2: data efficiency ────────────────────────────────────────────────────
echo "======== E4-v2: data efficiency ========"
for N in 0 1 2 3 5 10 20 50; do
    for S in 0 1 2 3 4; do
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=random \
            hybrid.n_data_points="${N}" \
            seed="${S}" \
            "experiment.name=e4_N${N}_seed${S}_ntk_random" || true
    done
done
for N in 1 5; do
    for S in 0 1 2 3 4; do
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

# ── E5-v2: noise robustness ───────────────────────────────────────────────────
echo "======== E5-v2: noise robustness ========"
declare -A SIGMA_TAGS=([0.00]=s000 [0.01]=s001 [0.05]=s005 [0.10]=s010 [0.20]=s020 [0.50]=s050)
for SIGMA in 0.00 0.01 0.05 0.10 0.20 0.50; do
    TAG="${SIGMA_TAGS[$SIGMA]}"
    for S in 0 1 2 3 4; do
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=random \
            hybrid.noise_sigma="${SIGMA}" \
            seed="${S}" \
            "experiment.name=e5_${TAG}_seed${S}_ntk_random" || true
    done
done

echo "======== All v2 experiments complete ========"
CONTAINER_EOF

log "=== Revision v2 complete ==="
