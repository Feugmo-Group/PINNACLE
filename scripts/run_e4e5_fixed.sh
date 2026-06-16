#!/usr/bin/env bash
# run_e4e5_fixed.sh — Corrected E4 (data efficiency) and E5 (noise robustness).
#
# Why this exists: the original E4/E5 runs were invalid. training.py:353 passed
# hybrid.fem_batch_size (=1) to sample_fem_data(), which overrode hybrid.n_data_points,
# so EVERY run trained with exactly one anchor regardless of N or sigma. Fixed in
# training.py + sampling.py (sample_fem_data now respects n_data_points; N=0 => None).
#
# Design corrections vs the old sweep:
#   - anchor_mode=seed (a FIXED N-point anchor set held constant across all steps),
#     not random (which redraws every step and defeats the "only N measurements" premise).
#   - anchor_seed=S so each replicate uses a distinct anchor subset AND init.
#   - consistent STEPS across all runs (no 50k/5k mix).
#
# Env: STEPS (default 50000), NTK_FREQ (default 100).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
LOG="$REPO_DIR/e4e5_fixed.log"
STEPS="${STEPS:-50000}"
NTK_FREQ="${NTK_FREQ:-100}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== E4/E5 corrected re-run (STEPS=$STEPS, NTK_FREQ=$NTK_FREQ) ==="

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env STEPS="$STEPS" --env NTK_FREQ="$NTK_FREQ" \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

run_exp() {
    local NAME="$1"; shift
    if ls /app/outputs/experiments/"${NAME}"/*/timing.json 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] ${NAME} — timing.json found"; return 0
    fi
    echo "  [START] ${NAME}"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.max_steps="${STEPS}" \
        training.ntk_update_freq="${NTK_FREQ}" \
        "experiment.name=${NAME}" \
        "$@" || true
}

echo "======== E4 (fixed): data efficiency, anchor_mode=seed ========"
for N in 0 1 2 3 5 10 20 50; do
    for S in 0 1 3; do
        run_exp "e4fix_N${N}_seed${S}" \
            training.weight_strat=ntk \
            hybrid.anchor_mode=seed \
            hybrid.anchor_seed="${S}" \
            hybrid.n_data_points="${N}" \
            seed="${S}"
    done
done

echo "======== E5 (fixed): noise robustness, single fixed anchor ========"
for SIGMA in 0.00 0.01 0.05 0.10 0.20 0.50; do
    TAG=$(printf "s%03d" "$(python3 -c "print(int(${SIGMA}*100))")")
    for S in 0 1 3; do
        run_exp "e5fix_${TAG}_seed${S}" \
            training.weight_strat=ntk \
            hybrid.anchor_mode=seed \
            hybrid.anchor_seed="${S}" \
            hybrid.n_data_points=1 \
            hybrid.noise_sigma="${SIGMA}" \
            seed="${S}"
    done
done

echo "======== E1-pure: weighting ablation, PURE PHYSICS (use_data=False) ========"
# Isolates NTK's value on the hard problem (no anchor doing the work). R2.2.
for STRAT in ntk uniform batch_size; do
    run_exp "e1pure_${STRAT}" \
        training.weight_strat="${STRAT}" \
        hybrid.use_data=False
done

echo "======== E2: BC-residual diagnostic (standard hybrid NTK) ========"
# bc_*_rms keys are logged to loss_history every run; figure generated from checkpoint. R2.3.
run_exp "e2_bc_residuals" \
    training.weight_strat=ntk \
    hybrid.anchor_mode=seed \
    hybrid.anchor_seed=0 \
    hybrid.n_data_points=1

echo "======== E4/E5/E1-pure/E2 corrected re-run complete ========"
CONTAINER_EOF

log "=== E4/E5 corrected re-run complete ==="
