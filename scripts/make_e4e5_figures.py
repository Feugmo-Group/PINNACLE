#!/usr/bin/env python3
"""
Build the E4 (data efficiency) and E5 (noise robustness) revision figures.

Reads each run's plots/pinn_vs_fem_E*.txt (already produced by analysis.py) and
computes the WHOLE-CURVE relative-L2 error per voltage, averaged over the five
voltages, then mean +- std over the seed replicates. This is the honest metric
(captures shape mismatch, not just final-time error). See feedback-visual-fem-match.

E4 dirs:  e4fix_N{N}_seed{S}   N in {0,1,2,3,5,10,20,50}, S in {0,1,3}
E5 dirs:  e5n10_s{SSS}_seed{S} sigma in {0,1,5,10,20,50}%, N=10 fixed anchors, S in {0,1,3}

Outputs (paper/figures/):
  fig_E4_data_efficiency.{png,pdf}
  fig_E5_noise_robustness.{png,pdf}

Stdlib + matplotlib only. Runs locally, no GPU/Docker.
"""
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
EXP = REPO / "outputs" / "experiments"
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
VOLT_TAGS = ["0.10", "0.40", "1.00", "1.60", "1.80"]
SEEDS = [0, 1, 3]


def latest_run(name):
    dirs = sorted([p for p in (EXP / name).glob("*") if p.is_dir()]) if (EXP / name).exists() else []
    return dirs[-1] if dirs else None


def curve_relL2(rows):
    """Whole-curve relative L2 error [%]: ||L_PINN - L_FEM||_2 / ||L_FEM||_2."""
    num = den = 0.0
    for r in rows:
        lp, lf = float(r[1]), float(r[2])
        num += (lp - lf) ** 2
        den += lf ** 2
    return 100.0 * (num ** 0.5) / (den ** 0.5) if den > 0 else None


def mean_err(name):
    """Mean whole-curve relL2 [%] over the five voltages for one run."""
    run = latest_run(name)
    if run is None:
        return None
    vals = []
    for tag in VOLT_TAGS:
        f = run / "plots" / f"pinn_vs_fem_E{tag}V.txt"
        if not f.exists():
            continue
        rows = [ln.split("\t") for ln in f.read_text().splitlines()
                if ln and not ln.startswith("Time")]
        e = curve_relL2(rows)
        if e is not None:
            vals.append(e)
    return statistics.mean(vals) if vals else None


def aggregate(name_fn, levels):
    """For each level, gather per-seed mean_err -> (xs, means, stds, per_seed)."""
    xs, means, stds, raw = [], [], [], {}
    for lv in levels:
        vals = [me for s in SEEDS if (me := mean_err(name_fn(lv, s))) is not None]
        if vals:
            xs.append(lv)
            means.append(statistics.mean(vals))
            stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
            raw[lv] = vals
    return xs, means, stds, raw


# ----------------------------------------------------------------------------
# E4 — data efficiency
# ----------------------------------------------------------------------------
N_LEVELS = [0, 1, 2, 3, 5, 10, 20, 50]
e4_x, e4_m, e4_s, e4_raw = aggregate(lambda N, S: f"e4fix_N{N}_seed{S}", N_LEVELS)

# N=0 (pure physics, no anchor) plotted separately as a reference band, since
# log-x cannot show N=0 and it is a categorical "no data" baseline.
e4_phys = None
if 0 in e4_raw:
    e4_phys = statistics.mean(e4_raw[0])
plot_x = [n for n in e4_x if n > 0]
plot_m = [m for n, m in zip(e4_x, e4_m) if n > 0]
plot_s = [s for n, s in zip(e4_x, e4_s) if n > 0]

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.errorbar(plot_x, plot_m, yerr=plot_s, marker="o", ms=6, lw=2,
            capsize=4, color="#2c6fbb", label="PINN (NTK + hybrid)", zorder=3)
if e4_phys is not None:
    ax.axhline(e4_phys, ls=":", lw=1.5, color="#b03030",
               label=f"pure physics (N=0): {e4_phys:.0f}%")
ax.set_xscale("log")
ax.set_yscale("log")  # N=0 ref (~518%) and N>=1 detail (7-66%) span >2 decades
ax.set_xticks(plot_x)
ax.set_xticklabels([str(n) for n in plot_x])
ax.set_yticks([5, 10, 20, 50, 100, 200, 500])
ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500"])
ax.set_xlabel("Number of FEM anchor points $N$")
ax.set_ylabel("Whole-curve relative $L_2$ error (%)")
ax.set_title("Data efficiency")
ax.grid(True, which="both", ls="--", alpha=0.35)
ax.legend(frameon=True, loc="upper right")
for n, m in zip(plot_x, plot_m):
    ax.annotate(f"{m:.0f}", (n, m), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="#2c6fbb")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_E4_data_efficiency.{ext}", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------
# E5 — noise robustness (N=10 fixed anchors)
# ----------------------------------------------------------------------------
SIG_TAGS = [(0, "s000"), (1, "s001"), (5, "s005"), (10, "s010"), (20, "s020"), (50, "s050")]
sig_lookup = dict(SIG_TAGS)
e5_x, e5_m, e5_s, e5_raw = aggregate(
    lambda pct, S: f"e5n10_{sig_lookup[pct]}_seed{S}", [p for p, _ in SIG_TAGS])

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.errorbar(e5_x, e5_m, yerr=e5_s, marker="s", ms=6, lw=2,
            capsize=4, color="#2e8b57", label="PINN ($N=10$ anchors)", zorder=3)
if e5_x:
    ax.axhline(e5_m[0], ls=":", lw=1.3, color="#777",
               label=f"noise-free baseline: {e5_m[0]:.0f}%")
ax.set_xlabel("Measurement noise $\\sigma$ on anchors (%)")
ax.set_ylabel("Whole-curve relative $L_2$ error (%)")
ax.set_title("Noise robustness")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(frameon=True)
for x, m in zip(e5_x, e5_m):
    ax.annotate(f"{m:.0f}", (x, m), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color="#2e8b57")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_E5_noise_robustness.{ext}", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------
# Console summary (for the paper tables / response letter)
# ----------------------------------------------------------------------------
print("=" * 60)
print("E4 — data efficiency (whole-curve relL2, mean+-std, seeds {0,1,3})")
print("=" * 60)
for n, m, s in zip(e4_x, e4_m, e4_s):
    print(f"  N={n:<3} {m:6.1f}% +- {s:4.1f}   per-seed: "
          + "  ".join(f"{v:.1f}" for v in e4_raw[n]))
print()
print("=" * 60)
print("E5 — noise robustness (N=10 anchors, mean+-std, seeds {0,1,3})")
print("=" * 60)
for x, m, s in zip(e5_x, e5_m, e5_s):
    print(f"  sigma={x:<3}% {m:6.1f}% +- {s:4.1f}   per-seed: "
          + "  ".join(f"{v:.1f}" for v in e5_raw[x]))
print()
print(f"figures written to {OUT}/")
for f in sorted(OUT.glob("fig_E[45]_*")):
    print(f"  {f.relative_to(REPO)}")
