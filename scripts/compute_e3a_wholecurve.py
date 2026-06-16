#!/usr/bin/env python3
"""Whole-curve relative-L2 film-thickness error for the E3a random-anchor seeds.

For each e3a_seed*_ntk_random run: load best_model.pt, predict L(t,E) at the FEM
time points for all 5 voltages, compute  ||L_pinn - L_fem||_2 / ||L_fem||_2  per
voltage, then average over voltages. Reports per-seed and across-seed mean+/-std,
and the converged/diverged split. CPU-only.
"""
import sys, re, pathlib
import numpy as np
import torch

REPO = pathlib.Path("/app")
EXP  = REPO / "outputs" / "experiments"
sys.path.insert(0, str(REPO / "pinnacle"))

VOLTAGES = [0.1, 0.4, 1.0, 1.6, 1.8]
FEM_DIR  = REPO / "pinnacle" / "FEM"

def load_fem_curves():
    curves = {}
    for v in VOLTAGES:
        fpath = FEM_DIR / f"{v} V.txt"
        if not fpath.exists():
            continue
        data = np.genfromtxt(fpath, delimiter='\t', names=True)
        t_col = 'Times' if 'Times' in data.dtype.names else data.dtype.names[0]
        L_col = 'Filmthicknessm' if 'Filmthicknessm' in data.dtype.names else data.dtype.names[2]
        valid = ~(np.isnan(data[t_col]) | np.isnan(data[L_col]))
        curves[v] = (data[t_col][valid], data[L_col][valid] * 1e9)
    return curves

FEM = load_fem_curves()

from omegaconf import OmegaConf
from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics

def seed_error(exp_dir):
    ckpts = sorted(exp_dir.glob("*/checkpoints/best_model.pt"))
    if not ckpts:
        return None
    ckpt_path = ckpts[-1]
    cfg_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    config = OmegaConf.load(cfg_path)
    nets = NetworkManager(config, torch.device("cpu"))
    phys = ElectrochemicalPhysics(config, torch.device("cpu"))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("networks", ck.get("model_state_dict", ck.get("state_dict", {})))
    nets.load_state_dict(sd); nets.eval()
    dt = next(iter(nets.get_all_parameters())).dtype
    sc = phys.scales
    per_v = {}
    with torch.no_grad():
        for v in VOLTAGES:
            if v not in FEM: continue
            t_s, L_fem = FEM[v]
            t_hat = torch.tensor(t_s / sc.tc, dtype=dt).unsqueeze(1)
            E_hat = torch.full((len(t_s), 1), v / sc.phic, dtype=dt)
            L_hat = nets['film_thickness'](torch.cat([t_hat, E_hat], dim=1)).squeeze().cpu().numpy()
            L_pinn = L_hat * sc.lc * 1e9
            relL2 = np.linalg.norm(L_pinn - L_fem) / (np.linalg.norm(L_fem) + 1e-12) * 100.0
            per_v[v] = relL2
    mean = float(np.mean(list(per_v.values()))) if per_v else float("nan")
    return {"best_loss": float(ck.get("best_loss", float("nan"))), "per_v": per_v, "mean": mean}

rows = []
for d in sorted(EXP.glob("e3a_seed*_ntk_random")):
    m = re.match(r"e3a_seed(\d+)_ntk_random", d.name)
    if not m: continue
    seed = int(m.group(1))
    r = seed_error(d)
    if r is None:
        print(f"  seed{seed}: NO checkpoint"); continue
    pv = "  ".join(f"{v}V={r['per_v'].get(v,float('nan')):6.1f}" for v in VOLTAGES)
    print(f"  seed{seed}: mean_relL2={r['mean']:8.2f}%   [{pv}]   best_loss={r['best_loss']:.3g}")
    rows.append((seed, r['mean']))

print("\n=== SUMMARY (whole-curve relative-L2, averaged over 5 voltages) ===")
means = np.array([m for _, m in rows])
conv = means[means < 20.0]   # converged threshold: 20%
div  = means[means >= 20.0]
print(f"total seeds: {len(means)}")
print(f"converged (<20%): {len(conv)}  -> mean={conv.mean():.2f}%  std={conv.std(ddof=1) if len(conv)>1 else 0:.2f}%")
print(f"diverged  (>=20%): {len(div)}  values={[f'{x:.1f}' for x in div]}")
print(f"converged seeds: {[s for s,m in rows if m<20]}")
print(f"diverged  seeds: {[s for s,m in rows if m>=20]}")
