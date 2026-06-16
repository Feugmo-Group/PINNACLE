#!/usr/bin/env bash
# run_all_v2_resume_full_gpu.sh — Resume from E3a seed6.
# Completed: E1-v2, E3a seeds 0-5, E3b early_low, E3b mid_low.
# Remaining: E3a seed6, E3b late_low/mid_mid/mid_high, E3c, E4, E5.
# Uses CUDA_MEM_FRACTION=1.0 (full GPU). All experiments run sequentially.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
LOG="$REPO_DIR/revision_v2.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Revision v2 resumed (full GPU, CUDA_MEM_FRACTION=1.0) ==="
log "REPO_DIR=$REPO_DIR"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env CUDA_MEM_FRACTION=1.0 \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=50000
REPO=/app

# Skip experiment if timing.json already exists in its output dir
skip_if_done() {
    local name="$1"
    local found
    found=$(find "$REPO/outputs/experiments/$name" -name "timing.json" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "--- SKIP $name (already has timing.json) ---"
        return 0
    fi
    return 1
}

# ── E3a seed6 (retry) ────────────────────────────────────────────────────────
echo "======== E3a seed6 retry ========"
NAME="e3a_seed6_ntk_random"
skip_if_done "$NAME" || \
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat=ntk \
    training.max_steps="${STEPS}" \
    hybrid.anchor_mode=random \
    seed=6 \
    "experiment.name=${NAME}" || true

# ── E3b-v2: remaining positions ──────────────────────────────────────────────
echo "======== E3b-v2: remaining positions ========"
declare -A E3B_SEEDS=([late_low]=2 [mid_mid]=3 [mid_high]=4)
for LABEL in late_low mid_mid mid_high; do
    SEED_IDX="${E3B_SEEDS[$LABEL]}"
    NAME="e3b_${LABEL}_ntk_random"
    echo "--- E3b ${LABEL} (seed=${SEED_IDX}) ---"
    skip_if_done "$NAME" || \
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk \
        training.max_steps="${STEPS}" \
        hybrid.anchor_mode=random \
        seed="${SEED_IDX}" \
        "experiment.name=${NAME}" || true
done

# ── E3c-v2: 10 fixed-anchor seeds ────────────────────────────────────────────
echo "======== E3c-v2: fixed-anchor comparison ========"
for SEED in $(seq 0 9); do
    NAME="e3c_seed${SEED}_ntk_fixed"
    echo "--- E3c seed ${SEED} ---"
    skip_if_done "$NAME" || \
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk \
        training.max_steps="${STEPS}" \
        hybrid.anchor_mode=seed \
        hybrid.anchor_seed="${SEED}" \
        "experiment.name=${NAME}" || true
done

# ── E4-v2: data efficiency ────────────────────────────────────────────────────
echo "======== E4-v2: data efficiency ========"
for N in 0 1 2 3 5 10 20 50; do
    for S in 0 1 2; do
        NAME="e4_N${N}_seed${S}_ntk_random"
        echo "--- E4 N=${N} seed=${S} random ---"
        skip_if_done "$NAME" || \
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=random \
            hybrid.n_data_points="${N}" \
            seed="${S}" \
            "experiment.name=${NAME}" || true
    done
done
for N in 1 5; do
    for S in 0 1 2; do
        NAME="e4_N${N}_seed${S}_ntk_fixed"
        echo "--- E4 N=${N} seed=${S} fixed ---"
        skip_if_done "$NAME" || \
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=seed \
            hybrid.anchor_seed="${S}" \
            hybrid.n_data_points="${N}" \
            seed="${S}" \
            "experiment.name=${NAME}" || true
    done
done

# ── E5-v2: noise robustness ───────────────────────────────────────────────────
echo "======== E5-v2: noise robustness ========"
declare -A SIGMA_TAGS=([0.00]=s000 [0.01]=s001 [0.05]=s005 [0.10]=s010 [0.20]=s020 [0.50]=s050)
for SIGMA in 0.00 0.01 0.05 0.10 0.20 0.50; do
    TAG="${SIGMA_TAGS[$SIGMA]}"
    for S in 0 1 2; do
        NAME="e5_${TAG}_seed${S}_ntk_random"
        echo "--- E5 sigma=${SIGMA} seed=${S} ---"
        skip_if_done "$NAME" || \
        PYTHONPATH=/app/pinnacle python -m pinnacle.main \
            training.weight_strat=ntk \
            training.max_steps="${STEPS}" \
            hybrid.anchor_mode=random \
            hybrid.noise_sigma="${SIGMA}" \
            seed="${S}" \
            "experiment.name=${NAME}" || true
    done
done

echo "======== Revision v2 complete ========"
CONTAINER_EOF

log "=== Revision v2 complete ==="
