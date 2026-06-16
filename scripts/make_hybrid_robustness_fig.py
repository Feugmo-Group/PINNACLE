#!/usr/bin/env python3
"""Robustness-band figure: FEM reference + mean PINN L(t) +/- sigma over the
converged random-anchor seeds, at three representative voltages. Augments (does
not replace) the single-run hybrid figure. Reads the live-network
pinn_vs_fem_E*V.txt files; pure stdlib + matplotlib."""
import pathlib, glob, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = pathlib.Path("outputs/experiments")
VOLT = [("0.10", "0.1 V"), ("1.00", "1.0 V"), ("1.80", "1.8 V")]
SEEDS = [0, 1, 2, 3, 4, 5, 6]
CONV_THRESH = 20.0  # % whole-curve; excludes diverged seeds

def read_run(seed):
    dirs = sorted(glob.glob(f"{EXP}/e3a_seed{seed}_ntk_random/*/"))
    if not dirs:
        return None
    run = pathlib.Path(dirs[-1]) / "plots"
    out = {}
    for tag, _ in VOLT:
        f = run / f"pinn_vs_fem_E{tag}V.txt"
        if not f.exists():
            return None
        rows = [ln.split("\t") for ln in f.read_text().splitlines()
                if ln and not ln.startswith("Time")]
        t = np.array([float(r[0]) for r in rows]) / 3600.0  # hours
        lp = np.array([float(r[1]) for r in rows])
        lf = np.array([float(r[2]) for r in rows])
        out[tag] = (t, lp, lf)
    # whole-curve mean error to decide convergence
    errs = [100*np.linalg.norm(out[tag][1]-out[tag][2])/np.linalg.norm(out[tag][2])
            for tag, _ in VOLT]
    out["_err"] = float(np.mean(errs))
    return out

runs = {s: r for s in SEEDS if (r := read_run(s)) is not None}
conv = {s: r for s, r in runs.items() if r["_err"] < CONV_THRESH}
print("seeds with data:", sorted(runs), " converged:", sorted(conv),
      " mean err:", {s: round(r["_err"], 1) for s, r in conv.items()})

PANEL = ["(a)", "(b)", "(c)"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, (tag, label), pl in zip(axes, VOLT, PANEL):
    # common grid = the FEM time grid (shared across seeds)
    t = conv[sorted(conv)[0]][tag][0]
    lf = conv[sorted(conv)[0]][tag][2]
    stack = np.vstack([conv[s][tag][1] for s in sorted(conv)])
    mean = stack.mean(0); sd = stack.std(0)
    ax.plot(t, lf, "k--", lw=1.6, label="FEM reference")
    ax.plot(t, mean, color="steelblue", lw=2, label=f"PINN mean ({len(conv)} seeds)")
    ax.fill_between(t, mean-sd, mean+sd, color="steelblue", alpha=0.25, label=r"$\pm\sigma$")
    ax.set_title(f"{pl}  E = {label}"); ax.set_xlabel("Time (hours)"); ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Film thickness (nm)")
axes[0].legend(fontsize=9, loc="lower right")
fig.tight_layout()
out = "paper/APL_PDM-PINN_review/images/fig_hybrid_robustness.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig("/tmp/fig_hybrid_robustness.png", dpi=130, bbox_inches="tight")
print("saved", out)
