#!/usr/bin/env python3
"""Diagnostic: confirm the learnable inverse parameter now receives gradient.

Recovers ONE parameter at a time. Default target k2_0 (boundary-stiff, drives
dL/dt). CPU-only, no training, no GPU.

Run in Docker:  PYTHONPATH=/app/pinnacle python scripts/diag_inverse_grad.py
"""
import torch
from omegaconf import OmegaConf

from physics.physics import ElectrochemicalPhysics
from networks.networks import NetworkManager
from losses.losses import compute_interior_loss, compute_film_physics_loss

torch.manual_seed(0)
dev = "cpu"

cfg = OmegaConf.load("conf/config.yaml")
cfg.inverse = OmegaConf.create({
    "enabled": True,
    "unknown_params": ["k2_0"],          # the simple, identifiable one
    "k2_0_init": 3.6e-5, "D_cv_init": 5.0e-21,
    "stage2_start_step": 10**9,
})
cfg.device = dev

physics = ElectrochemicalPhysics(cfg, dev)
nm = NetworkManager(cfg, dev)
networks = nm.networks if hasattr(nm, "networks") else nm
ip = physics.inverse_params

print("inverse_enabled:", physics.inverse_enabled, " unknown:", ip.unknown)
print("learnable params:", [n for n, _ in ip.named_parameters()])

N = 64
t = torch.rand(N, 1, device=dev, requires_grad=True)
E = torch.rand(N, 1, device=dev) * 1.8
tf = torch.rand(N, 1, device=dev, requires_grad=True)
Ef = torch.rand(N, 1, device=dev) * 1.8
x = torch.rand(N, 1, device=dev, requires_grad=True)


def fmt(g):
    return "None" if g is None else f"{float(g):.3e}"


def check(loss, tag):
    ip.zero_grad(set_to_none=True)
    loss = loss[0] if isinstance(loss, tuple) else loss
    loss.backward(retain_graph=True)
    g = ip.log_k2_0.grad if hasattr(ip, "log_k2_0") else None
    print(f"[{tag}] loss={float(loss.detach()):.3e}  grad log_k2_0={fmt(g)}")


check(compute_interior_loss(x, t, E, networks, physics), "interior")
check(compute_film_physics_loss(tf, Ef, networks, physics), "film_physics")
print("\nExpected: k2_0 now gets NON-None gradient from film_physics "
      "(k2 enters dL/dt = Omega*(k2-k5)).")
