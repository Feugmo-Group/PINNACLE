#!/usr/bin/env bash
# E6 — single-parameter inverse recovery from film-thickness L(t) observations.
#
# Recover ONE physical parameter at a time (joint recovery is ill-posed). The
# experiment contrasts how a parameter's "stiffness" — which loss term it
# dominates — controls its identifiability from L(t) data:
#   k2_0  : boundary-stiff. Enters dL/dt = Omega*(k2 - k5) through an exponential
#           Butler-Volmer term -> strongly identifiable from L(t).
#   k5_0  : mild/linear in dL/dt -> moderately identifiable.
#   D_cv  : interior-stiff (cation-vacancy transport); only weakly couples to
#           L(t) -> expected to be poorly identifiable from L(t) alone.
#
# Network lr is lowered to 2e-4 and inverse-param gradients are clipped
# (training.py) to keep the PDE solve stable past the early-divergence regime.
#
# Env: PARAM (k2_0|k5_0|D_cv, default k2_0), STEPS (default 30000),
#      OBS_V (observation voltages, default '[0.1]'; e.g. '[0.1,1.0,1.8]'),
#      EXP_SUFFIX (appended to experiment.name, default empty), DOCKER_IMAGE.
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
STEPS="${STEPS:-30000}"
PARAM="${PARAM:-k2_0}"
OBS_V="${OBS_V:-[0.1]}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

case "$PARAM" in
  k2_0) INIT_FLAG="inverse.k2_0_init=3.6e-5"  ; TRUE="3.6e-6"  ;;  # 10x true
  k5_0) INIT_FLAG="inverse.k5_0_init=7.65e-8" ; TRUE="7.65e-9" ;;  # 10x true
  D_cv) INIT_FLAG="inverse.D_cv_init=5.0e-21" ; TRUE="1.0e-21" ;;  # ~5x true
  *) echo "Unknown PARAM=$PARAM (use k2_0|k5_0|D_cv)"; exit 1 ;;
esac

echo "[$(date '+%H:%M:%S')] === E6 $PARAM recovery (STEPS=$STEPS, true=$TRUE) ==="

docker run --rm -i --init --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$REPO_DIR:/app" --workdir="/app" \
  --env STEPS="$STEPS" --env PARAM="$PARAM" --env INIT_FLAG="$INIT_FLAG" --env TRUE="$TRUE" \
  --env OBS_V="$OBS_V" --env EXP_SUFFIX="$EXP_SUFFIX" \
  "$DOCKER_IMAGE" bash -s << 'EOF'
set -uo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat=ntk training.max_steps="${STEPS}" \
    optimizer.adam.lr=2.0e-4 \
    inverse.enabled=true "inverse.unknown_params=[${PARAM}]" \
    ${INIT_FLAG} "inverse.obs_voltages=${OBS_V}" \
    inverse.stage2_start_step=$(( STEPS / 2 )) \
    experiment.name="e6_recover_${PARAM}${EXP_SUFFIX}" || true
echo "Done. True ${PARAM} = ${TRUE}; recovered value logged each step in loss_history (key '${PARAM}')."
EOF
echo "[$(date '+%H:%M:%S')] === E6 $PARAM recovery complete ==="
