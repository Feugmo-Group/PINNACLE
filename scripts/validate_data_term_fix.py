#!/usr/bin/env python3
"""Validate that sample_fem_data now respects hybrid.n_data_points.

Run inside Docker:  PYTHONPATH=/app/pinnacle python scripts/validate_data_term_fix.py
"""
import torch
from hydra import initialize, compose
from physics.physics import ElectrochemicalPhysics
from sampling.sampling import CollocationSampler

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")

device = torch.device("cpu")
physics = ElectrochemicalPhysics(cfg, device)
sampler = CollocationSampler(cfg, physics, device)

print("anchor_mode =", cfg.hybrid.anchor_mode, " fem_batch_size =", cfg.hybrid.fem_batch_size)
print(f"{'n_data_points':>14} | {'returned anchors':>16}")
print("-" * 35)
for N in [0, 1, 2, 5, 10, 50]:
    cfg.hybrid.n_data_points = N
    fem = sampler.sample_fem_data()
    n = 0 if fem is None else len(fem["t"])
    flag = "  <-- pure physics (None)" if fem is None else ""
    print(f"{N:>14} | {n:>16}{flag}")

# Sanity: distinct anchor sets for distinct seeds in 'seed' mode
cfg.hybrid.anchor_mode = "seed"
cfg.hybrid.n_data_points = 5
print("\nseed-mode reproducibility / distinctness (N=5):")
for s in [0, 1, 2]:
    cfg.hybrid.anchor_seed = s
    a = sampler.sample_fem_data()["t"].tolist()
    b = sampler.sample_fem_data()["t"].tolist()
    print(f"  anchor_seed={s}: same set across two calls? {a == b} | t={[round(x,4) for x in a]}")
