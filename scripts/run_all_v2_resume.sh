#!/usr/bin/env bash
# run_all_v2_resume.sh — Resume/restart-safe sweep for E4 and E5.
# Completed: E1-v2, E3a (7 seeds), E3b (5 positions), E3c (10 seeds), E4 N=0/N=1 (6 runs).
# CUDA_MEM_FRACTION hardcoded to 0.5 to prevent OOM.
# Resume logic: skips experiments with timing.json; resumes interrupted ones from latest checkpoint.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
LOG="$REPO_DIR/revision_v2.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Revision v2 RESUMED (checkpoint-aware, CUDA_MEM_FRACTION=0.5) ==="
log "REPO_DIR=$REPO_DIR"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env CUDA_MEM_FRACTION=0.5 \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=5000
NTK_FREQ=1000

# ── Helper: run one experiment with automatic skip/resume logic ───────────────
# Usage: run_exp <experiment_name> <extra hydra args...>
# - If timing.json already exists → skip (already done).
# - If a partial checkpoint exists → resume from the latest model_step_*.pt.
# - Otherwise → start fresh.
run_exp() {
    local NAME="$1"; shift
    local EXTRA_ARGS=("$@")

    # Check if already completed
    if ls /app/outputs/experiments/"${NAME}"/*/timing.json 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] ${NAME} — timing.json found, already complete"
        return 0
    fi

    # Find latest checkpoint from any previous (interrupted) run
    local RESUME_ARG=""
    local LATEST_CKPT=""
    if [ -d "/app/outputs/experiments/${NAME}" ]; then
        LATEST_CKPT=$(find /app/outputs/experiments/"${NAME}" -name "model_step_*.pt" 2>/dev/null \
            | sort -t_ -k3 -n | tail -1) || true
    fi
    if [ -n "$LATEST_CKPT" ]; then
        echo "  [RESUME] ${NAME} — resuming from ${LATEST_CKPT}"
        RESUME_ARG="training.resume_checkpoint=${LATEST_CKPT}"
    else
        echo "  [START] ${NAME} — starting fresh"
    fi

    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.max_steps="${STEPS}" \
        training.ntk_update_freq="${NTK_FREQ}" \
        "experiment.name=${NAME}" \
        ${RESUME_ARG:+"${RESUME_ARG}"} \
        "${EXTRA_ARGS[@]}" || true
}

# ── E3a/E3b/E3c: DONE ────────────────────────────────────────────────────────
echo "======== E3a/E3b/E3c: already done, skipping ========"

# ── E4-v2: data efficiency ────────────────────────────────────────────────────
echo "======== E4-v2: data efficiency ========"
for N in 0 1 2 3 5 10 20 50; do
    for S in 0 1 2; do
        echo "--- E4 N=${N} seed=${S} random ---"
        run_exp "e4_N${N}_seed${S}_ntk_random" \
            training.weight_strat=ntk \
            hybrid.anchor_mode=random \
            hybrid.n_data_points="${N}" \
            seed="${S}"
    done
done
for N in 1 5; do
    for S in 0 1 2; do
        echo "--- E4 N=${N} seed=${S} fixed ---"
        run_exp "e4_N${N}_seed${S}_ntk_fixed" \
            training.weight_strat=ntk \
            hybrid.anchor_mode=seed \
            hybrid.anchor_seed="${S}" \
            hybrid.n_data_points="${N}" \
            seed="${S}"
    done
done

# ── E5-v2: noise robustness ───────────────────────────────────────────────────
echo "======== E5-v2: noise robustness ========"
declare -A SIGMA_TAGS=([0.00]=s000 [0.01]=s001 [0.05]=s005 [0.10]=s010 [0.20]=s020 [0.50]=s050)
for SIGMA in 0.00 0.01 0.05 0.10 0.20 0.50; do
    TAG="${SIGMA_TAGS[$SIGMA]}"
    for S in 0 1 2; do
        echo "--- E5 sigma=${SIGMA} seed=${S} ---"
        run_exp "e5_${TAG}_seed${S}_ntk_random" \
            training.weight_strat=ntk \
            hybrid.anchor_mode=random \
            hybrid.noise_sigma="${SIGMA}" \
            seed="${S}"
    done
done

echo "======== All v2 experiments complete ========"
CONTAINER_EOF

log "=== Revision v2 complete ==="
