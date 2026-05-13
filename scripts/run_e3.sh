#!/usr/bin/env bash
# E3 — Anchor-point robustness study
#   Part A: 30 random anchor seeds (NTK only)  = 30 runs (50k steps)
#   Part B:  5 systematic (t,E) positions (NTK only) = 5 runs
# Paper placement: expanded Sec. IV.D
#
# All sub-runs execute inside a single Docker container (install once).
#
# Usage:
#   ./scripts/run_e3.sh        # both parts (A and B)
#   ./scripts/run_e3.sh A      # Part A only (30 seeds)
#   ./scripts/run_e3.sh B      # Part B only (systematic positions)
#   ./scripts/run_e3.sh AB     # both parts explicitly
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image to use      (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
PART="${1:-AB}"

echo "================================================================"
echo " E3 — Anchor-point robustness (Part ${PART})"
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
  -e "PART=$PART" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=50000

# ── Part A: 30 random seeds ───────────────────────────────────────────────────
if [[ "$PART" == *A* ]]; then
    echo "================================================================"
    echo " E3-A: 30 random anchor seeds (NTK only) = 30 runs"
    echo "================================================================"
    for SEED in $(seq 0 29); do
            echo "--- E3-A seed=${SEED} strat=ntk ---"
            PYTHONPATH=/app/pinnacle python -m pinnacle.main \
                training.weight_strat="ntk" \
                training.max_steps="${STEPS}" \
                hybrid.anchor_seed="${SEED}" \
                "experiment.name=e3a_seed${SEED}_ntk"
    done
fi

# ── Part B: systematic (t, E) positions ──────────────────────────────────────
if [[ "$PART" == *B* ]]; then
    echo "================================================================"
    echo " E3-B: systematic anchor positions (NTK only) = 5 runs"
    echo "================================================================"

    # Format: "label:t_value:E_value"
    # Note: the planned "two_pt" multi-anchor entry was dropped from this pass
    # because the sampler currently accepts a single (anchor_t, anchor_E) only.
    # True two-anchor support is deferred unless a reviewer requests it; see
    # paper/revision_plan_v2.md for the rationale.
    declare -a ANCHORS=(
        "early_low:5000:0.1"
        "mid_low:150000:0.1"
        "late_low:800000:0.1"
        "mid_mid:150000:1.0"
        "mid_high:150000:1.8"
    )

    for ENTRY in "${ANCHORS[@]}"; do
        LABEL="${ENTRY%%:*}"; REST="${ENTRY#*:}"
        T_VAL="${REST%%:*}"; E_VAL="${REST#*:}"
            echo "--- E3-B ${LABEL} (t=${T_VAL}, E=${E_VAL}) strat=ntk ---"
            PYTHONPATH=/app/pinnacle python -m pinnacle.main \
                training.weight_strat="ntk" \
                training.max_steps="${STEPS}" \
                hybrid.anchor_t="${T_VAL}" \
                hybrid.anchor_E="${E_VAL}" \
                "experiment.name=e3b_${LABEL}_ntk"
    done
fi

echo ""
echo "E3 done. Results in outputs/experiments/e3a_* and e3b_*"
CONTAINER_EOF
