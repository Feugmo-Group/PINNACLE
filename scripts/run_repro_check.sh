#!/usr/bin/env bash
# Run the same default-anchor config twice back-to-back. If predictions_overview
# differs between the two runs, CUDA training is non-deterministic for this
# problem at the trajectory level — that explains why OLD_GOOD's solution cannot
# be reproduced bit-for-bit even with the same anchor and seed.
#
# Usage: ./scripts/run_repro_check.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

docker run --rm -i --init --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$REPO_DIR:/app" --workdir="/app" \
  "$DOCKER_IMAGE" bash -s <<'EOF'
set -euo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q
PYTHONPATH=/app/pinnacle python -m pinnacle.main experiment.name=repro_check_A
PYTHONPATH=/app/pinnacle python -m pinnacle.main experiment.name=repro_check_B
EOF

echo ""
echo "Compare outputs/experiments/repro_check_A/<ts>/plots/predictions_overview.png"
echo "    vs outputs/experiments/repro_check_B/<ts>/plots/predictions_overview.png"
