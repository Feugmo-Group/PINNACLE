#!/usr/bin/env python3
"""
Build the E2 boundary-condition residual figure (Fig.~\\ref{fig:bc_residuals}).

Reads paper/figures/e2_bc_residuals.csv (extracted from the e2_bc_residuals
checkpoint's loss_history) and plots the six per-interface weighted BC loss
components over training. The figure visualises the open/stiff boundary-condition
problem: the anion-vacancy metal-film and film-solution residuals episodically
blow up by tens of orders of magnitude, while the potential and cation-vacancy
BCs stay bounded. A rolling median (robust to the transient spikes) is overlaid
on the raw trace so the qualitative trend is legible despite the huge dynamic range.

Outputs (paper/figures/ and paper/images/): fig_bc_residuals.{png,pdf}
Stdlib + matplotlib only.
"""
import csv
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
CSV = REPO / "paper" / "figures" / "e2_bc_residuals.csv"
OUTDIRS = [REPO / "paper" / "figures", REPO / "paper" / "images"]

KEYS = ["weighted_cv_mf_bc", "weighted_av_mf_bc", "weighted_u_mf_bc",
        "weighted_cv_fs_bc", "weighted_av_fs_bc", "weighted_u_fs_bc"]
LABELS = {
    "weighted_cv_mf_bc": r"$c_\mathrm{CV}$ metal--film",
    "weighted_av_mf_bc": r"$c_\mathrm{AV}$ metal--film",
    "weighted_u_mf_bc":  r"$\phi$ metal--film",
    "weighted_cv_fs_bc": r"$c_\mathrm{CV}$ film--soln.",
    "weighted_av_fs_bc": r"$c_\mathrm{AV}$ film--soln.",
    "weighted_u_fs_bc":  r"$\phi$ film--soln.",
}
COLORS = {
    "weighted_cv_mf_bc": "#1f77b4", "weighted_av_mf_bc": "#d62728", "weighted_u_mf_bc": "#2ca02c",
    "weighted_cv_fs_bc": "#1f77b4", "weighted_av_fs_bc": "#d62728", "weighted_u_fs_bc": "#2ca02c",
}
STYLE = {k: ("-" if "_mf_" in k else "--") for k in KEYS}

FLOOR = 1e-20  # log-plot floor; values below are clipped


def rolling_median(xs, win=201):
    n = len(xs)
    half = win // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(statistics.median(xs[lo:hi]))
    return out


# ---- load ----
rows = list(csv.DictReader(CSV.open()))
steps = [int(r["idx"]) for r in rows]
series = {k: [max(FLOOR, abs(float(r[k]))) for r in rows] for k in KEYS}

# ---- plot ----
fig, ax = plt.subplots(figsize=(6.6, 4.3))
for k in KEYS:
    med = rolling_median(series[k])
    ax.plot(steps, med, STYLE[k], color=COLORS[k], lw=1.8, label=LABELS[k], alpha=0.9)

ax.set_yscale("log")
ax.set_ylim(1e-12, 1e12)
ax.set_xlabel("Training step")
ax.set_ylabel("Weighted BC residual loss")
ax.set_title("Per-interface boundary-condition residuals")
ax.grid(True, which="major", ls="--", alpha=0.35)
# legend: split metal-film (solid) vs film-solution (dashed)
ax.legend(ncol=2, fontsize=8, framealpha=0.9, loc="upper right", title="solid: metal--film   dashed: film--soln.")
ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
ax.text(steps[-1], 1.3, "$O(1)$", color="0.4", fontsize=8, ha="right", va="bottom")
fig.tight_layout()

for d in OUTDIRS:
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"fig_bc_residuals.{ext}", dpi=200)
plt.close(fig)

# ---- console summary ----
print("E2 BC-residual components (rolling-median end value | raw peak):")
for k in KEYS:
    print(f"  {LABELS[k]:<22} end~{rolling_median(series[k])[-1]:.2e}   peak {max(series[k]):.2e}")
print("\nfigures written:")
for d in OUTDIRS:
    print(f"  {(d / 'fig_bc_residuals.pdf').relative_to(REPO)}")
