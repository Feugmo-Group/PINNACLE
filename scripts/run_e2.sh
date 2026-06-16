#!/usr/bin/env bash
# E2 — Re-log E1-A (NTK, 50k steps) to capture per-component BC-RMS residuals.
# The bc_*_rms keys are already computed in compute_boundary_loss() and stored
# in loss_history; this script simply runs NTK training so those keys are
# populated in a fresh checkpoint for the Sec. V.C diagnostic figure.
#
# Output: outputs/experiments/e2_bc_residuals/<timestamp>/
#   - checkpoints/best_model.pt
#   - loss_history contains bc_cv_mf_rms, bc_av_mf_rms, bc_u_mf_rms,
#                            bc_cv_fs_rms, bc_av_fs_rms, bc_u_fs_rms, bc_h_fs_rms
#
# Optional env vars:
#   REPO_DIR     – repo root (default: auto-detected)
#   DOCKER_IMAGE – container image (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E2 — NTK re-run with per-BC RMS residual logging"
echo " REPO_DIR     : $REPO_DIR"
echo " DOCKER_IMAGE : $DOCKER_IMAGE"
echo "================================================================"

docker run --rm -i --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  --env CUDA_MEM_FRACTION="${CUDA_MEM_FRACTION:-0.5}" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat=ntk \
    training.max_steps=50000 \
    precision=float64 \
    experiment.name=e2_bc_residuals

echo ""
echo "E2 done. Checkpoint in outputs/experiments/e2_bc_residuals/"
echo "loss_history keys bc_*_rms are now populated for the Sec. V.C figure."
CONTAINER_EOF
