#!/usr/bin/env bash
# run_e4e5_parallel.sh — MAXIMAL-parallelism corrected sweep on the GB10 (128 GB).
# No CUDA_MEM_FRACTION cap; every pending run launches concurrently in one container.
# Each run uses ~1 GB, so the GPU memory easily holds all of them; the denormal
# flush (main.py) keeps the formerly-slow seed-2 runs fast.
#
# Runs: E4 N∈{0,1,2,3,5,10,20,50}×seed{0,1,2}, E5 σ∈{0,.01,.05,.1,.2,.5}×seed{0,1,2}
#       (anchor_mode=seed, fixed N-point set), E1-pure {ntk,uniform,batch_size}
#       (use_data=False), E2 bc_residuals. Skip logic via timing.json.
# Env: STEPS (default 35000), NTK_FREQ (default 100), MAXJOBS (0 = unlimited/all).
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
LOG="$REPO_DIR/e4e5_parallel.log"
STEPS="${STEPS:-35000}"
NTK_FREQ="${NTK_FREQ:-100}"
MAXJOBS="${MAXJOBS:-0}"

echo "[$(date '+%H:%M:%S')] === PARALLEL corrected sweep (STEPS=$STEPS, MAXJOBS=$MAXJOBS) ===" | tee -a "$LOG"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env STEPS="$STEPS" --env NTK_FREQ="$NTK_FREQ" --env MAXJOBS="$MAXJOBS" \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -uo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

mkdir -p /app/outputs/parlogs

done_already() { ls /app/outputs/experiments/"$1"/*/timing.json >/dev/null 2>&1; }

launch() {
    local NAME="$1"; shift
    if done_already "$NAME"; then echo "  [SKIP] $NAME"; return 0; fi
    echo "  [LAUNCH] $NAME"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.max_steps="${STEPS}" training.ntk_update_freq="${NTK_FREQ}" \
        "experiment.name=${NAME}" "$@" \
        > "/app/outputs/parlogs/${NAME}.log" 2>&1 &
}

throttle() {
    # If MAXJOBS>0, block until running background jobs drop below the cap.
    [ "${MAXJOBS}" -gt 0 ] || return 0
    while [ "$(jobs -rp | wc -l)" -ge "${MAXJOBS}" ]; do wait -n 2>/dev/null || sleep 5; done
}

# ── E4: data efficiency (fixed N-point anchor set) ───────────────────────────
for N in 0 1 2 3 5 10 20 50; do for S in 0 1 2; do
    throttle
    launch "e4fix_N${N}_seed${S}" \
        training.weight_strat=ntk hybrid.anchor_mode=seed \
        hybrid.anchor_seed="${S}" hybrid.n_data_points="${N}" seed="${S}"
done; done

# ── E5: noise robustness (single fixed anchor) ───────────────────────────────
for SIGMA in 0.00 0.01 0.05 0.10 0.20 0.50; do
    TAG=$(printf "s%03d" "$(python3 -c "print(int(${SIGMA}*100))")")
    for S in 0 1 2; do
        throttle
        launch "e5fix_${TAG}_seed${S}" \
            training.weight_strat=ntk hybrid.anchor_mode=seed \
            hybrid.anchor_seed="${S}" hybrid.n_data_points=1 \
            hybrid.noise_sigma="${SIGMA}" seed="${S}"
    done
done

# ── E1-pure: weighting ablation, pure physics (R2.2) ─────────────────────────
for STRAT in ntk uniform batch_size; do
    throttle
    launch "e1pure_${STRAT}" training.weight_strat="${STRAT}" hybrid.use_data=False
done

# ── E2: BC-residual diagnostic (R2.3) ────────────────────────────────────────
throttle
launch "e2_bc_residuals" training.weight_strat=ntk \
    hybrid.anchor_mode=seed hybrid.anchor_seed=0 hybrid.n_data_points=1

echo "  All runs launched; waiting for completion..."
wait
echo "======== PARALLEL corrected sweep complete ========"
CONTAINER_EOF

echo "[$(date '+%H:%M:%S')] === PARALLEL sweep complete ===" | tee -a "$LOG"
