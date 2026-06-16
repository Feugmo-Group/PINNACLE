#!/usr/bin/env python3
"""
Aggregate final-time relative film-thickness errors for the revision v2 sweep.

Reads each experiment's plots/pinn_vs_fem_E*.txt (already produced by analysis.py)
and the timing.json, then prints summary tables for E1, E3a, E3b, E3c, E4, E5,
W7, and E6. Stdlib only — runs locally, no GPU/Docker needed.
"""
import json, pathlib, statistics, glob, re

REPO = pathlib.Path(__file__).resolve().parents[1]
EXP = REPO / "outputs" / "experiments"
VOLT_TAGS = ["0.10", "0.40", "1.00", "1.60", "1.80"]


def latest_run(name):
    dirs = sorted([p for p in (EXP / name).glob("*") if p.is_dir()]) if (EXP / name).exists() else []
    return dirs[-1] if dirs else None


def _curve_relL2(rows):
    """Whole-curve relative L2 error [%]: ||L_PINN - L_FEM||_2 / ||L_FEM||_2.
    Far more honest than final-time RelError — captures shape mismatch and is
    not fooled by a single PINN/FEM crossing point. (User directive: always
    judge the FULL FEM-vs-PINN curve match, not a single error number.)"""
    num = den = 0.0
    for r in rows:
        lp, lf = float(r[1]), float(r[2])
        num += (lp - lf) ** 2
        den += lf ** 2
    return 100.0 * (num ** 0.5) / (den ** 0.5) if den > 0 else None


def final_errors(name):
    """Return dict volt_tag -> WHOLE-CURVE relative-L2 error [%] per voltage."""
    run = latest_run(name)
    if run is None:
        return None
    out = {}
    for tag in VOLT_TAGS:
        f = run / "plots" / f"pinn_vs_fem_E{tag}V.txt"
        if not f.exists():
            continue
        rows = [ln.split("\t") for ln in f.read_text().splitlines() if ln and not ln.startswith("Time")]
        if not rows:
            continue
        e = _curve_relL2(rows)
        if e is not None:
            out[tag] = e
    return out or None


def mean_err(name):
    e = final_errors(name)
    if not e:
        return None
    return statistics.mean(e.values())


def timing(name):
    run = latest_run(name)
    if run is None:
        return None
    t = run / "timing.json"
    return json.loads(t.read_text()) if t.exists() else None


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if x is not None else "—"


print("=" * 70)
print("E1 — Loss-weighting ablation (Table II)")
print("=" * 70)
print(f"{'strategy':<12}{'med ms/step':>12}{'peak MB':>10}{'wall s':>9}{'mean err%':>11}")
for strat, name in [("ntk", "e1_ablation_ntk_v2"), ("uniform", "e1_ablation_uniform_v2"),
                    ("batch_size", "e1_ablation_batch_size_v2")]:
    t = timing(name)
    me = mean_err(name)
    if t:
        print(f"{strat:<12}{fmt(t['median_ms_per_step'],1):>12}{fmt(t['peak_mem_mb_max'],0):>10}"
              f"{t['wall_clock_s']:>9}{fmt(me):>11}")
    e = final_errors(name)
    if e:
        print(f"   per-V: " + "  ".join(f"{k}V={fmt(v)}" for k, v in e.items()))

print()
print("=" * 70)
print("E3a — Random-anchor seed robustness (7 seeds)")
print("=" * 70)
e3a = []
for s in range(7):
    me = mean_err(f"e3a_seed{s}_ntk_random")
    if me is not None:
        e3a.append(me)
        print(f"  seed{s}: mean_err = {fmt(me)}%")
if e3a:
    print(f"  >>> E3a across {len(e3a)} seeds: mean={fmt(statistics.mean(e3a))}%  "
          f"std={fmt(statistics.pstdev(e3a))}%  min={fmt(min(e3a))}  max={fmt(max(e3a))}")

print()
print("=" * 70)
print("E3b — Systematic anchor positions (5)")
print("=" * 70)
e3b = []
for pos in ["early_low", "mid_low", "late_low", "mid_mid", "mid_high"]:
    me = mean_err(f"e3b_{pos}_ntk_random")
    if me is not None:
        e3b.append(me)
        print(f"  {pos:<10}: mean_err = {fmt(me)}%")
if e3b:
    print(f"  >>> E3b worst-case mean_err = {fmt(max(e3b))}%  (range {fmt(min(e3b))}–{fmt(max(e3b))})")

print()
print("=" * 70)
print("E3c — Fixed-anchor seed robustness (10 seeds)")
print("=" * 70)
e3c = []
for s in range(10):
    me = mean_err(f"e3c_seed{s}_ntk_fixed")
    if me is not None:
        e3c.append(me)
if e3c:
    print(f"  >>> E3c across {len(e3c)} seeds: mean={fmt(statistics.mean(e3c))}%  "
          f"std={fmt(statistics.pstdev(e3c))}%  min={fmt(min(e3c))}  max={fmt(max(e3c))}")
    print(f"  per-seed: " + "  ".join(fmt(x) for x in e3c))

print()
print("=" * 70)
print("E4 — Data efficiency (Table IV): mean+-std over 3 seeds, random anchor")
print("=" * 70)
print(f"{'N':<6}{'mean err%':>11}{'std':>9}   per-seed")
for N in [0, 1, 2, 3, 5, 10, 20, 50]:
    vals = []
    for s in [0, 1, 2]:
        me = mean_err(f"e4_N{N}_seed{s}_ntk_random")
        if me is not None:
            vals.append(me)
    if vals:
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{N:<6}{fmt(mu):>11}{fmt(sd):>9}   " + "  ".join(fmt(v) for v in vals))

print()
print("--- E4 fixed-anchor (N=1,5) ---")
for N in [1, 5]:
    vals = []
    for s in [0, 1, 2]:
        me = mean_err(f"e4_N{N}_seed{s}_ntk_fixed")
        if me is not None:
            vals.append(me)
    if vals:
        print(f"  N={N} fixed: mean={fmt(statistics.mean(vals))}%  "
              f"std={fmt(statistics.pstdev(vals) if len(vals)>1 else 0)}  vals=" + " ".join(fmt(v) for v in vals))

print()
print("=" * 70)
print("E5 — Noise robustness (Table V): mean+-std over 3 seeds")
print("=" * 70)
print(f"{'sigma':<8}{'mean err%':>11}{'std':>9}   per-seed")
for tag, sig in [("s000", "0%"), ("s001", "1%"), ("s005", "5%"),
                 ("s010", "10%"), ("s020", "20%"), ("s050", "50%")]:
    vals = []
    for s in [0, 1, 2]:
        me = mean_err(f"e5_{tag}_seed{s}_ntk_random")
        if me is not None:
            vals.append(me)
    if vals:
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{sig:<8}{fmt(mu):>11}{fmt(sd):>9}   " + "  ".join(fmt(v) for v in vals))

print()
print("=" * 70)
print("W7 — Main result (Table I): per-voltage final error")
print("=" * 70)
for s in [1, 2, 3]:
    e = final_errors(f"w7_hybrid_seed{s}_random_anchor")
    if e:
        print(f"  seed{s}: " + "  ".join(f"{k}V={fmt(v)}" for k, v in e.items())
              + f"   | mean={fmt(statistics.mean(e.values()))}")

print()
print("=" * 70)
print("E6 — Inverse problem")
print("=" * 70)
for name in sorted([p.name for p in EXP.glob("e6_*")]):
    t = timing(name)
    e = final_errors(name)
    line = f"  {name:<22}"
    if e:
        line += f"mean_err={fmt(statistics.mean(e.values()))}%"
    print(line)
