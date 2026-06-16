#!/usr/bin/env bash
# Test: anchor_mode=seed, anchor_seed=0 — exactly the random-draw branch
# that the OLD GOOD run (commit 88d0517) fell through to. If this matches
# the OLD GOOD predictions_overview, the position-mode lookup is the only
# code that differs in effect. If it ALSO produces the bad shape, the
# regression is elsewhere (RNG consumption order, library version, etc.).
#
# Usage: ./scripts/run_test_seed0.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

docker run --rm -i --init \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$REPO_DIR:/app" --workdir="/app" \
  "$DOCKER_IMAGE" \
  bash -s <<'EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
  hybrid.anchor_mode=seed \
  hybrid.anchor_seed=0 \
  experiment.name=test_seed0
EOF

echo ""
echo "Done. Check outputs/experiments/test_seed0/<timestamp>/plots/predictions_overview.png"
