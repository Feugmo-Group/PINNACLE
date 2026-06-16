#!/usr/bin/env bash
# run_e5_n10.sh — E5 noise robustness redone with N=10 fixed anchors.
# The original E5 used 1 anchor, whose sigma=0 baseline was already ~65% (broken),
# making the noise trend uninterpretable. With 10 fixed anchors the sigma=0 baseline
# is ~18%, so multiplicative Gaussian noise on the measurements has a real baseline
# to degrade -> a meaningful "physics-as-denoiser" result. Output dirs: e5n10_*.
# Env: STEPS (default 35000), NTK_FREQ (default 100).
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
LOG="$REPO_DIR/e5_n10.log"
STEPS="${STEPS:-35000}"
NTK_FREQ="${NTK_FREQ:-100}"

echo "[$(date '+%H:%M:%S')] === E5 N=10 noise re-run (STEPS=$STEPS) ===" | tee -a "$LOG"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env STEPS="$STEPS" --env NTK_FREQ="$NTK_FREQ" \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -uo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

run_exp() {
    local NAME="$1"; shift
    if ls /app/outputs/experiments/"${NAME}"/*/timing.json 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] ${NAME}"; return 0; fi
    echo "  [START] ${NAME}"
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.max_steps="${STEPS}" training.ntk_update_freq="${NTK_FREQ}" \
        "experiment.name=${NAME}" "$@" || true
}

echo "======== E5 N=10: noise robustness with 10 fixed anchors ========"
for SIGMA in 0.00 0.01 0.05 0.10 0.20 0.50; do
    TAG=$(printf "s%03d" "$(python3 -c "print(int(${SIGMA}*100))")")
    for S in 0 1 3; do
        run_exp "e5n10_${TAG}_seed${S}" \
            training.weight_strat=ntk \
            hybrid.anchor_mode=seed \
            hybrid.anchor_seed="${S}" \
            hybrid.n_data_points=10 \
            hybrid.noise_sigma="${SIGMA}" \
            seed="${S}"
    done
done
echo "======== E5 N=10 re-run complete ========"
CONTAINER_EOF

echo "[$(date '+%H:%M:%S')] === E5 N=10 re-run complete ===" | tee -a "$LOG"
