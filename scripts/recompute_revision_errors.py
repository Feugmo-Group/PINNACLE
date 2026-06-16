#!/usr/bin/env python3
"""Recompute E3a/E3b/E4/E5 whole-curve relative-L2 errors from the live-network
pinn_vs_fem_E*V.txt files (the same valid source make_e4e5_figures.py uses).
No checkpoint loading -> immune to the state_dict-key bug. Pure stdlib."""
import pathlib, statistics, re

EXP = pathlib.Path("outputs/experiments")
VOLT_TAGS = ["0.10", "0.40", "1.00", "1.60", "1.80"]

def curve_relL2(rows):
    num = den = 0.0
    for r in rows:
        lp, lf = float(r[1]), float(r[2])
        num += (lp - lf) ** 2
        den += lf ** 2
    return 100.0 * (num ** 0.5) / (den ** 0.5) if den > 0 else None

def latest_run(name):
    d = EXP / name
    subs = sorted([p for p in d.glob("*") if p.is_dir()]) if d.exists() else []
    return subs[-1] if subs else None

def mean_err(name):
    run = latest_run(name)
    if run is None: return None
    vals = []
    for tag in VOLT_TAGS:
        f = run / "plots" / f"pinn_vs_fem_E{tag}V.txt"
        if not f.exists(): continue
        rows = [ln.split("\t") for ln in f.read_text().splitlines()
                if ln and not ln.startswith("Time")]
        e = curve_relL2(rows)
        if e is not None: vals.append(e)
    return statistics.mean(vals) if vals else None

def agg(names_seeds):
    """names_seeds: list of run names -> return (mean, std, raw list) over runs."""
    vals = [v for n in names_seeds if (v := mean_err(n)) is not None]
    if not vals: return None
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) == 1 else statistics.stdev(vals)
    return m, s, vals

print("="*70)
print("E3a — random-anchor robustness (per seed, mean over 5 voltages)")
e3a = {}
for s in range(7):
    e = mean_err(f"e3a_seed{s}_ntk_random")
    e3a[s] = e
    print(f"  seed{s}: {e:8.2f}%" if e is not None else f"  seed{s}: (none)")
conv = {s:e for s,e in e3a.items() if e is not None and e < 20}
div  = {s:e for s,e in e3a.items() if e is not None and e >= 20}
print(f"  -> {len(conv)}/{len(e3a)} converge (<20%); mean={statistics.mean(conv.values()):.2f}% "
      f"std={statistics.stdev(conv.values()):.2f}% ; diverged seeds={list(div.keys())}")

print("="*70)
print("E3b — single fixed-anchor position sweep (placement sensitivity)")
for pos in ["early_low","mid_low","late_low","mid_mid","mid_high"]:
    e = mean_err(f"e3b_pos_{pos}")
    print(f"  {pos:10s}: {e:8.2f}%" if e is not None else f"  {pos:10s}: (none)")

print("="*70)
print("E4 — data efficiency (mean +/- std over seeds 0,1,3)")
for N in [0,1,2,3,5,10,20,50]:
    r = agg([f"e4fix_N{N}_seed{s}" for s in (0,1,3)])
    if r: print(f"  N={N:2d}: {r[0]:8.2f}% +/- {r[1]:6.2f}%   (raw={[f'{x:.1f}' for x in r[2]]})")
    else: print(f"  N={N:2d}: (none)")

print("="*70)
print("E5 — anchor-noise robustness, N=10 (mean +/- std over seeds 0,1,3)")
for pct,tag in [(0,"s000"),(1,"s001"),(5,"s005"),(10,"s010"),(20,"s020"),(50,"s050")]:
    r = agg([f"e5n10_{tag}_seed{s}" for s in (0,1,3)])
    if r: print(f"  sigma={pct:2d}%: {r[0]:8.2f}% +/- {r[1]:6.2f}%   (raw={[f'{x:.1f}' for x in r[2]]})")
    else: print(f"  sigma={pct:2d}%: (none)")
