#!/usr/bin/env bash
# Master revision script — runs all pending APL revision experiments sequentially.
# Designed for a single GPU; each group installs packages once inside Docker
# and runs all its sub-experiments back-to-back.
#
# Usage:
#   ./scripts/run_revision.sh            # all pending experiments in order
#   ./scripts/run_revision.sh E2 W7      # specific experiments only
#
# Priority order (single GPU, ~80 h total):
#   E2  —   1 run  (~1 h)   BC-residual diagnostic figure (Sec. V.C)
#   W7  —   3 runs (~3 h)   multi-seed Table I
#   E4  —  40 runs (~40 h)  data-efficiency sweep (overnight)
#   E5  —  30 runs (~30 h)  noise-robustness sweep (overnight)
#   E3  —  35 runs (~35 h)  anchor-validation sweep (overnight)
#
# Already done (skip automatically):
#   E1  — ablation timing data exists in outputs/experiments/e1_ablation_*/
#   E7  — cost table already generated (run_e7.sh needs no GPU)
#   E6  — needs inverse-param gradient-flow fix before re-running
#
# Optional env vars:
#   REPO_DIR     – repo root (default: auto-detected via git)
#   DOCKER_IMAGE – container image (default: nvcr.io/nvidia/pytorch:26.01-py3)

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

START_TIME=$(date +%s)

# ── E6: inverse problem (Group 4, ~15 h) — run early, highest paper value ────
if want E6 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E6 — Inverse problem k3_0+D_CV recovery (~15 h) ##########"
    "${SCRIPT_DIR}/run_e6.sh"
fi

# ── E2: BC-residual re-run (Group 4, ~1 h) ───────────────────────────────────
if want E2 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E2 — BC-residual diagnostic run (~1 h) ##########"
    "${SCRIPT_DIR}/run_e2.sh"
fi

# ── W7: multi-seed Table I (Group 5, ~3 h) ───────────────────────────────────
if want W7 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## W7 — Multi-seed Table I, seeds 1-3 (~3 h) ##########"
    "${SCRIPT_DIR}/run_w7.sh"
fi

# ── E4: data-count sweep (Group 5, ~40 h) ────────────────────────────────────
if want E4 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E4 — Data-efficiency sweep (~40 h) ##########"
    "${SCRIPT_DIR}/run_e4.sh"
fi

# ── E5: noise-robustness sweep (Group 5, ~30 h) ──────────────────────────────
if want E5 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E5 — Noise-robustness sweep (~30 h) ##########"
    "${SCRIPT_DIR}/run_e5.sh"
fi

# ── E3: anchor-validation sweep (Group 5, ~35 h) ─────────────────────────────
if want E3 "${ARGS[@]}" 2>/dev/null || $RUN_ALL; then
    echo ""
    echo "########## E3 — Anchor-validation sweep (~35 h) ##########"
    "${SCRIPT_DIR}/run_e3.sh" AB
fi

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "================================================================"
echo " All requested experiments completed in ${ELAPSED} minutes."
echo " Outputs: outputs/experiments/"
echo "   e2_bc_residuals/          — per-BC RMS curves (Sec. V.C fig)"
echo "   w7_hybrid_seed{1,2,3}/    — seeds for Table I mean+/-sigma"
echo "   e4_N*_seed*_ntk/          — data-efficiency sweep"
echo "   e5_s*_seed*_ntk/          — noise-robustness sweep"
echo "   e3a_seed*_ntk/ e3b_*/     — anchor-validation sweep"
echo "================================================================"
