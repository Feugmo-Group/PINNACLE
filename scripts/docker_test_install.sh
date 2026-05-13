#!/usr/bin/env bash
# Validate the full PINNACLE stack inside the standard NVIDIA PyTorch container.
#
# Two-stage check:
#   1. test_install.py  — package imports, module imports, forward/backward pass,
#                         all aggregators, PINNTrainer 5 steps (programmatic).
#   2. python -m pinnacle.main — real Hydra entry point, 20 optimization steps
#                         so the actual config loading, sampling, and progress
#                         printing are exercised end-to-end.
#
# Usage:
#   ./scripts/docker_test_install.sh
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image to use      (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "REPO_DIR     : $REPO_DIR"
echo "DOCKER_IMAGE : $DOCKER_IMAGE"
echo

docker run --rm -i --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

# ── install extra packages (PyTorch already in base image) ────────────────────
echo ">>> Installing extra packages..."
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
echo ">>> Packages installed."
echo

# ── Stage 1: unit + integration tests (programmatic) ─────────────────────────
echo "============================================================"
echo "  STAGE 1 — test_install.py (imports + PINNTrainer 5 steps)"
echo "============================================================"
PYTHONPATH=/app/pinnacle python scripts/test_install.py

echo
echo "============================================================"
echo "  STAGE 2 — python -m pinnacle.main (Hydra, 20 real steps)"
echo "============================================================"

# Run 20 optimizer steps through the real entry point.
# Overrides keep it fast: tiny batches, uniform weighting, no FEM data.
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.max_steps=20 \
    training.rec_results_freq=5 \
    training.rec_inference_freq=1000 \
    training.save_network_freq=20 \
    training.weight_strat=uniform \
    batch_size.BC=32 \
    batch_size.interior=64 \
    batch_size.IC=32 \
    batch_size.L=32 \
    batch_size.inference=32 \
    sampling.adaptive.interior_points=64 \
    sampling.adaptive.boundary_points=32 \
    sampling.adaptive.initial_points=32 \
    sampling.adaptive.film_points=32 \
    hybrid.use_data=False \
    experiment.name=docker_install_check

echo
echo "============================================================"
echo "  ALL CHECKS PASSED — PINNACLE is ready to run."
echo "============================================================"
CONTAINER_EOF
