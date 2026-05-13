#!/usr/bin/env bash
# Master revision script — runs all APL revision experiments in priority order.
# Each experiment script spawns its own Docker container and runs all its
# sub-experiments inside (packages installed once per experiment group).
#
# Usage:
#   ./scripts/run_revision.sh            # all experiments
#   ./scripts/run_revision.sh E1         # single experiment
#   ./scripts/run_revision.sh E1 E6      # subset
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image to use      (default: nvcr.io/nvidia/pytorch:26.01-py3)
#
# Recommended order (from revision_plan.md):
#   1. E1  —   4 runs (fast, sets baseline for Table II)
#   2. E6  —  15 runs (headline result, start early)
#   3. E4  —  80 runs (sweep, run overnight)
#   4. E5  —  60 runs (sweep, run overnight)
#   5. E3  —  72 runs (robustness, lowest priority)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ALL=false
if [ $# -eq 0 ]; then
    RUN_ALL=true
fi

want() {
    local target="$1"; shift
    $RUN_ALL && return 0
    for arg in "$@"; do [[ "$arg" == "$target" ]] && return 0; done
    return 1
}

ARGS=("$@")

# ── E1: ablation (HIGH) ───────────────────────────────────────────────────────
if want E1 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E1 — Ablation study ##########"
    "${SCRIPT_DIR}/run_e1.sh"
fi

# ── E6: inverse problem (VERY HIGH) ──────────────────────────────────────────
if want E6 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E6 — Inverse problem ##########"
    "${SCRIPT_DIR}/run_e6.sh"
fi

# ── E4: data-count sweep (HIGH) ──────────────────────────────────────────────
if want E4 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E4 — Data-count sweep ##########"
    "${SCRIPT_DIR}/run_e4.sh"
fi

# ── E5: noise sweep (HIGH) ────────────────────────────────────────────────────
if want E5 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E5 — Noise robustness ##########"
    "${SCRIPT_DIR}/run_e5.sh"
fi

# ── E3: anchor robustness (MEDIUM) ───────────────────────────────────────────
if want E3 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E3 — Anchor robustness (Parts A+B) ##########"
    "${SCRIPT_DIR}/run_e3.sh" AB
fi

echo ""
echo "================================================================"
echo " All requested revision experiments completed."
echo " Outputs: outputs/experiments/e{1,3,4,5,6}_*/"
echo " Each run contains:"
echo "   timing.json                 — ms/step, peak GPU MB"
echo "   plots/loss_landscape_*.png  — per-component loss landscape"
echo "   plots/training_losses.png   — NTK vs BRDR convergence"
echo "   checkpoints/best_model.pt   — best checkpoint"
echo "================================================================"
