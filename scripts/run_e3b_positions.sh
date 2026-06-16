#!/usr/bin/env bash
# run_e3b_positions.sh — CORRECTED E3-B systematic anchor-position sweep.
# The earlier run_e3_v2.sh used anchor_mode=random and never passed the (t,E)
# position, so its "e3b_*" runs were just duplicates of E3a seeds 0-4. This
# script uses anchor_mode=position with explicit hybrid.anchor_t/anchor_E so each
# run genuinely pins the single FEM anchor to a distinct (t,E) location.
# Output dirs: e3b_pos_<label>.  Env: STEPS (default 35000), NTK_FREQ (default 100).
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
LOG="$REPO_DIR/e3b_positions.log"
STEPS="${STEPS:-35000}"
NTK_FREQ="${NTK_FREQ:-100}"

echo "[$(date '+%H:%M:%S')] === E3-B position sweep (STEPS=$STEPS) ===" | tee -a "$LOG"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env STEPS="$STEPS" --env NTK_FREQ="$NTK_FREQ" \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -uo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

run_pos() {
    local LABEL="$1" T="$2" E="$3"
    local NAME="e3b_pos_${LABEL}"
    if ls /app/outputs/experiments/"${NAME}"/*/timing.json 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] ${NAME}"; return 0; fi
    echo "  [START] ${NAME}  (t=${T}s, E=${E}V)"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat=ntk training.max_steps="${STEPS}" \
        training.ntk_update_freq="${NTK_FREQ}" \
        hybrid.anchor_mode=position hybrid.anchor_t="${T}" hybrid.anchor_E="${E}" \
        "experiment.name=${NAME}" || true
}

echo "======== E3-B: five systematic anchor positions ========"
run_pos early_low 2000   0.1
run_pos mid_low   200000 0.1
run_pos late_low  380000 0.1
run_pos mid_mid   200000 1.0
run_pos mid_high  200000 1.8
echo "======== E3-B position sweep complete ========"
CONTAINER_EOF

echo "[$(date '+%H:%M:%S')] === E3-B position sweep complete ===" | tee -a "$LOG"
