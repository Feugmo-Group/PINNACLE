#!/usr/bin/env python3
"""
Compute Table I mean±sigma from W7 seeds (1, 2, 4) and update paper_tracked.tex.

Reads the best_model.pt loss_history from each seed checkpoint, extracts
the final hybrid film-thickness errors at each voltage, and writes mean±sigma
into the tab:comparison table in paper_tracked.tex.
"""
import pathlib, json, sys, re, torch, numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "paper_tracked.tex"

# ── locate seed checkpoints ──────────────────────────────────────────────────
seed_dirs = {}
for seed in (1, 2, 4):
    pattern = list((REPO / "outputs" / "experiments").glob(f"w7_hybrid_seed{seed}/*/checkpoints/best_model.pt"))
    if pattern:
        seed_dirs[seed] = sorted(pattern)[-1]   # latest run

print(f"Found checkpoints for seeds: {sorted(seed_dirs.keys())}")
for s, p in seed_dirs.items():
    print(f"  seed {s}: {p}")

if len(seed_dirs) < 3:
    print("ERROR: need seeds 1, 2, 4 to compute statistics.", file=sys.stderr)
    sys.exit(1)

# ── extract per-voltage final errors ─────────────────────────────────────────
VOLTAGES = [0.1, 0.4, 1.0, 1.6, 1.8]

# FEM reference final L values (nm) from Table I in the original paper
FEM_REF = {0.1: 1.27, 0.4: 2.53, 1.0: 5.78, 1.6: 11.4, 1.8: 16.1}

def load_errors(ckpt_path: pathlib.Path):
    """Load checkpoint and return dict voltage → final relative error (%)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The checkpoint stores loss_history and model state.
    # Try to find film_thickness predictions stored in the checkpoint.
    # Fall back to reading timing.json + final_stats.json if available.
    errors = {}

    timing_path = ckpt_path.parents[1] / "timing.json"
    if timing_path.exists():
        data = json.loads(timing_path.read_text())
        # timing.json may have per-voltage errors stored
        if "per_voltage_error" in data:
            return data["per_voltage_error"]

    # Try loss_history for film error proxy
    lh = ckpt.get("loss_history", {})
    # If we have the full history, the final loss gives us a proxy but not
    # per-voltage error directly. Return None to signal we need inference.
    return None

# Try loading errors from timing.json first
errors_by_seed = {}
for seed, ckpt_path in seed_dirs.items():
    result = load_errors(ckpt_path)
    if result is not None:
        errors_by_seed[seed] = result
        print(f"  seed {seed}: loaded from timing.json")
    else:
        print(f"  seed {seed}: timing.json has no per-voltage errors — will use best_loss proxy")

# If we couldn't get per-voltage errors, fall back to best_loss from the log
# and report that Table I update requires a proper inference pass.
if len(errors_by_seed) < 3:
    print("\nPer-voltage inference not available from checkpoints alone.")
    print("Extracting best_loss from log as proxy for Table I caption update...")

    # Extract best_loss from the revision log
    log = (REPO / "revision_e2_w7.log").read_text()
    best_losses = {}
    for seed in (1, 2, 4):
        pattern = rf"--- W7 hybrid NTK seed={seed} ---.*?Best loss:\s*([\d.]+)"
        m = re.search(pattern, log, re.DOTALL)
        if m:
            best_losses[seed] = float(m.group(1))
            print(f"  seed {seed}: best_loss = {best_losses[seed]:.4f}")

    if len(best_losses) == 3:
        vals = list(best_losses.values())
        mean_bl = np.mean(vals)
        std_bl = np.std(vals, ddof=1)
        print(f"\nBest-loss statistics: {mean_bl:.3f} ± {std_bl:.3f}")
        print("NOTE: These are scalar best-losses, not per-voltage errors.")
        print("Full per-voltage Table I update requires running inference on each checkpoint.")
        print("Writing a note to paper_tracked.tex that per-voltage stats are pending.")

        # Update the caption note in paper_tracked.tex
        tex = PAPER.read_text()
        old = r"\textcolor{red}{Values in Table~\ref{tab:comparison} are reported as mean$\pm\sigma$ over three independent training seeds; see Sec.~\ref{subsec:robust} for the anchor-location study.}"
        new = (f"Values in Table~\\ref{{tab:comparison}} are from the single primary run "
               f"(seed~0); mean$\\pm\\sigma$ over seeds~1, 2, 4 "
               f"(best-loss: ${mean_bl:.2f}\\pm{std_bl:.2f}$) will replace these once "
               f"per-voltage inference is aggregated.")
        if old in tex:
            tex = tex.replace(old, new)
            PAPER.write_text(tex)
            print(f"Updated paper_tracked.tex caption note.")
        else:
            print("Caption note not found in expected form — skipping tex update.")
        sys.exit(0)
    else:
        print("Could not extract best_loss for all seeds.", file=sys.stderr)
        sys.exit(1)
