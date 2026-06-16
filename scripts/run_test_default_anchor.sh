#!/usr/bin/env bash
# Test: new default anchor_mode=default → snaps to (76000 s, 0.4 V, 3.64 nm).
# Confirms the configured default reproduces the OLD-GOOD predictions_overview.
#
# Usage: ./scripts/run_test_default_anchor.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

docker run --rm -i --init --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$REPO_DIR:/app" --workdir="/app" \
  "$DOCKER_IMAGE" bash -s <<'EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
  experiment.name=test_default_anchor
EOF

echo ""
echo "Done. Check outputs/experiments/test_default_anchor/<timestamp>/plots/predictions_overview.png"
