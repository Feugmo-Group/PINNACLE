#!/usr/bin/env python3
"""
Smoke test: verifies the environment, module imports, and a minimal
forward + backward pass through all five PINNACLE networks.

Run directly:  python scripts/smoke_test.py
In Docker:     see scripts/docker_smoke_test.sh
"""
import sys
import os

# Allow running from repo root without a full package install.
# The docker scripts also set PYTHONPATH=/app/pinnacle, which takes precedence.
_pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pinnacle")
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

import torch

print(f"[ok] torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"[ok] CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
else:
    print("[warn] CUDA not available — using CPU")

# ── module imports ────────────────────────────────────────────────────────────
from networks.networks import FFN, NetworkManager       # noqa: E402
print("[ok] networks imported")

from physics.physics import ElectrochemicalPhysics      # noqa: E402,F401
print("[ok] physics imported")

from losses.losses import compute_total_loss            # noqa: E402,F401
print("[ok] losses imported")

from gradients.gradients import GradientComputer        # noqa: E402,F401
print("[ok] gradients imported")

# ── minimal config (no omegaconf / Hydra needed) ─────────────────────────────
class _Cfg(dict):
    """Dict that exposes keys as dot-notation attributes, recursively."""
    def __getattr__(self, k):
        try:
            v = self[k]
            return _Cfg(v) if isinstance(v, dict) else v
        except KeyError:
            raise AttributeError(k) from None

    def get(self, k, default=None):
        v = super().get(k, default)
        return _Cfg(v) if isinstance(v, dict) else v


_layer = {"hidden_layers": 2, "layer_size": 8}
cfg = _Cfg({
    "arch": {name: dict(_layer) for name in ["potential", "CV", "AV", "h", "L"]},
    "networks": {"type": "FFN", "initialize": True, "soft": False},
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── build all five networks ───────────────────────────────────────────────────
nm = NetworkManager(cfg, device)
print(f"[ok] NetworkManager on {device}, networks: {list(nm.networks.keys())}")

# ── forward pass ─────────────────────────────────────────────────────────────
xte = torch.randn(8, 3, device=device)   # (x, t, E) — inputs to φ, c_cv, c_av, c_h
te  = torch.randn(8, 2, device=device)   # (t, E)    — inputs to L

outs = [nm.networks[k](xte) for k in ("potential", "cv", "av", "h")]
outs.append(nm.networks["film_thickness"](te))

assert all(o.shape == torch.Size([8, 1]) for o in outs), \
    f"unexpected output shapes: {[o.shape for o in outs]}"
print("[ok] forward pass — all 5 networks, output shapes (8, 1) ✓")

# ── backward pass ─────────────────────────────────────────────────────────────
loss = sum(o.sum() for o in outs)
loss.backward()

first_param = next(nm.networks["potential"].parameters())
assert first_param.grad is not None, "gradient not populated after backward()"
print("[ok] backward pass — gradients populated ✓")

print("\n=== Smoke test PASSED ===")
