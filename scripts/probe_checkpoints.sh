#!/usr/bin/env bash
# Quick diagnostic: dump hybrid_data_point from each checkpoint we care
# about so we can compare anchors numerically.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

docker run --rm -i --init --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$REPO_DIR:/app" --workdir="/app" \
  "$DOCKER_IMAGE" bash -s <<'EOF'
python - <<'PY'
import torch
paths = {
    "OLD_GOOD": "outputs/experiments/hybrid_training_final/2026-05-13_13-39-48/checkpoints/best_model.pt",
    "NEW_paper": "outputs/experiments/hybrid_training_final/2026-05-14_09-50-58/checkpoints/best_model.pt",
    "TEST_position_174k_1V": "outputs/experiments/test_empirical_anchor/2026-05-14_11-13-49/checkpoints/best_model.pt",
    "TEST_seed0": "outputs/experiments/test_seed0/2026-05-14_14-04-35/checkpoints/best_model.pt",
    "TEST_default_anchor": "outputs/experiments/test_default_anchor/2026-05-14_15-06-46/checkpoints/best_model.pt",
}
for name, p in paths.items():
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        hdp = ckpt.get("hybrid_data_point", None)
        keys = list(ckpt.keys())
        print(f"\n=== {name} ===")
        print(f"  path: {p}")
        print(f"  ckpt keys: {keys}")
        print(f"  hybrid_data_point: {hdp}")
    except Exception as e:
        print(f"\n=== {name} ===\n  ERROR: {e}")
PY
EOF
