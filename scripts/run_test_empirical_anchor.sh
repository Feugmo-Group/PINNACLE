#!/usr/bin/env bash
# Test: anchor at (174000 s, 1.0 V) — the empirical "good" anchor from the
# pre-regression run. Verifies whether the anchor is the only differentiator
# between the OLD good predictions_overview and the current bad one.
#
# Usage: ./scripts/run_test_empirical_anchor.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

docker run --rm -i --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  "$DOCKER_IMAGE" \
  bash -s <<'EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
  hybrid.anchor_mode=position \
  hybrid.anchor_t=174000 \
  hybrid.anchor_E=1.0 \
  experiment.name=test_empirical_anchor
EOF

echo ""
echo "Done. Check outputs/experiments/test_empirical_anchor/<timestamp>/plots/predictions_overview.png"
