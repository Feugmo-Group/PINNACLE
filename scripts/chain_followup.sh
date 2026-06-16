#!/usr/bin/env bash
# chain_followup.sh — wait for the running E4/E5 sweep to finish, then re-run
# run_e4e5_fixed.sh so the second pass picks up: the 3 N=0 retries (now fixed),
# E1-pure (3 runs), and E2 (1 run). Skip logic handles everything already done.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

WAIT_PID="${1:-2804160}"
echo "[chain] waiting for sweep PID ${WAIT_PID} to finish..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 120; done
echo "[chain] sweep PID ${WAIT_PID} done at $(date). Waiting 60s for GPU to settle."
sleep 60

echo "[chain] launching second pass (N=0 retries + E1-pure + E2)."
STEPS=35000 NTK_FREQ=100 ./scripts/run_e4e5_fixed.sh
echo "[chain] second pass complete at $(date)."
