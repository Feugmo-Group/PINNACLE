#!/usr/bin/env bash
# E7 — Build Table IV: PINN vs FEM computational cost comparison.
# Parses existing E1 timing.json files (no new training required).
# FEM wall-clock estimates are taken from Bosing 2023 (COMSOL, adaptive mesh).
#
# Output: printed to stdout + written to outputs/e7_cost_table.txt
#
# Run from the repo root:
#   bash scripts/run_e7.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
OUT="$REPO_DIR/outputs/e7_cost_table.txt"
mkdir -p "$REPO_DIR/outputs"

python3 - "$REPO_DIR" "$OUT" << 'PYEOF'
import json, sys, pathlib, statistics

repo = pathlib.Path(sys.argv[1])
out  = pathlib.Path(sys.argv[2])

STEPS = 50_000          # training steps per run
FEM_MINUTES_PER_VOLTAGE = 60.0   # Bosing 2023: ~1 h per forward solve in COMSOL
N_VOLTAGES = 5          # paper evaluates 5 applied potentials

results = {}
for strat in ("ntk", "uniform", "batch_size"):
    jsons = sorted(
        (repo / "outputs" / "experiments" / f"e1_ablation_{strat}").rglob("timing.json")
    )
    # Keep only runs that have STEPS steps (50k) to exclude early smoke tests
    full = [j for j in jsons
            if json.loads(j.read_text()).get("n_steps_total", 0) == STEPS]
    if not full:
        print(f"[WARN] no {STEPS}-step timing.json found for {strat}", flush=True)
        continue
    # Use latest complete run
    data = json.loads(full[-1].read_text())
    results[strat] = data

lines = []
lines.append("=" * 72)
lines.append("  Table IV — Computational cost: PINN (this work) vs FEM (Bosing 2023)")
lines.append("=" * 72)
lines.append(f"  Training steps per run: {STEPS:,}")
lines.append("")
lines.append(f"  {'Strategy':<18}  {'ms/step (med)':<16}  {'Total (min)':<14}  {'Peak GPU (MB)'}")
lines.append(f"  {'-'*18}  {'-'*16}  {'-'*14}  {'-'*14}")

for strat, d in results.items():
    med  = d["median_ms_per_step"]
    total_min = STEPS * d["mean_ms_per_step"] / 60_000
    mem  = d["peak_mem_mb_max"]
    lines.append(f"  {strat:<18}  {med:<16.1f}  {total_min:<14.1f}  {mem:.0f}")

lines.append("")
lines.append(f"  {'FEM (1 voltage)':<18}  {'—':<16}  {FEM_MINUTES_PER_VOLTAGE:<14.0f}  n/a")
lines.append(f"  {'FEM (full curve)':<18}  {'—':<16}  {FEM_MINUTES_PER_VOLTAGE*N_VOLTAGES:<14.0f}  n/a")
lines.append("")
lines.append("  Notes:")
lines.append("  - PINN 'total' = mean_ms_per_step × 50,000 steps.")
lines.append("  - After training, the full polarisation curve (all voltages)")
lines.append("    is obtained in a single inference pass (< 1 s); FEM requires")
lines.append(f"    {N_VOLTAGES} independent runs (~{N_VOLTAGES*FEM_MINUTES_PER_VOLTAGE/60:.0f} h total).")
lines.append("  - FEM estimate from Bosing (2023), COMSOL Multiphysics,")
lines.append("    adaptive meshing, moving boundary.")
lines.append("=" * 72)

text = "\n".join(lines)
print(text)
out.write_text(text + "\n")
print(f"\nSaved to {out}")
PYEOF
