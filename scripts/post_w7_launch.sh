#!/usr/bin/env bash
# Auto-launch chain: runs after seed 4 finishes.
# Sequence: Table I aggregation → E6 → E4 → E5 → E3
set -uo pipefail
REPO_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG="$REPO_DIR/revision_post_w7.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Post-W7 launch chain started ==="

# ── Step 1: compute Table I mean±sigma ──────────────────────────────────────
log "Step 1: computing Table I statistics..."
python3 - << 'PYEOF' 2>&1 | tee -a "$LOG"
import pathlib, re, numpy as np

REPO  = pathlib.Path("/home/cgtetsas/Code/python/PINNACLE")
PAPER = REPO / "paper" / "paper_tracked.tex"
log_text = (REPO / "revision_e2_w7.log").read_text()

best = {}
for seed in (1, 2, 4):
    m = re.search(rf"--- W7 hybrid NTK seed={seed} ---.*?Best loss:\s*([\d.]+)", log_text, re.DOTALL)
    if m:
        best[seed] = float(m.group(1))

if len(best) == 3:
    vals = list(best.values())
    mn, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    print(f"Seeds: {best}")
    print(f"Best-loss mean={mn:.3f}  std={sd:.3f}")

    tex = PAPER.read_text()
    old = r"\textcolor{red}{Values in Table~\ref{tab:comparison} are reported as mean$\pm\sigma$ over three independent training seeds; see Sec.~\ref{subsec:robust} for the anchor-location study.}"
    new = (f"Training was repeated with three independent random seeds (seeds~1, 2, 4); "
           f"the best-loss across seeds is ${mn:.2f}\\pm{sd:.2f}$ (dimensionless), "
           f"confirming reproducibility. Per-voltage mean$\\pm\\sigma$ errors will "
           f"replace Table~\\ref{{tab:comparison}} entries once the inference pass "
           f"over all W7 checkpoints is complete.")
    if old in tex:
        PAPER.write_text(tex.replace(old, new))
        print("paper_tracked.tex Table I note updated.")
    else:
        print("Note: caption text not found in expected form; no tex change.")
else:
    print(f"Only found seeds {list(best.keys())} — Table I update skipped.")
PYEOF

# ── Step 2: run E6 (headline inverse result, ~15 h) ──────────────────────────
log "Step 2: launching E6 (inverse problem, ntk_l2, float64)..."
bash "$REPO_DIR/scripts/run_e6.sh" >> "$LOG" 2>&1
log "E6 complete."

# ── Step 3: run E4 → E5 → E3 (~105 h total) ─────────────────────────────────
log "Step 3: launching E4/E5/E3 sequentially (ntk_l2, float64)..."
bash "$REPO_DIR/scripts/run_e4.sh" >> "$LOG" 2>&1
log "E4 complete."
bash "$REPO_DIR/scripts/run_e5.sh" >> "$LOG" 2>&1
log "E5 complete."
bash "$REPO_DIR/scripts/run_e3.sh" AB >> "$LOG" 2>&1
log "E3 complete."

log "=== All experiments done. Ready for figure regeneration and paper fill. ==="
