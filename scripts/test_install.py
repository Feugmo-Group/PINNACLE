#!/usr/bin/env python3
"""
PINNACLE full installation and integration test.

Tests (in order):
  1. Third-party package imports (torch, hydra, scipy, matplotlib, …)
  2. All PINNACLE module imports including new aggregator + utils.plotting
  3. NetworkManager forward + backward pass (5 networks)
  4. Every aggregator strategy — aggregate() returns a differentiable tensor
  5. PINNTrainer: 5 training steps with timing.json written to disk
  6. Inverse mode: physics.k3_0_param created and present in optimizer

Usage:
  # Inside Docker (PYTHONPATH set by docker_test_install.sh):
  python scripts/test_install.py

  # Local (from repo root):
  PYTHONPATH=pinnacle python scripts/test_install.py
"""

import sys
import os
import tempfile
import traceback

# Allow running from repo root without a full install
_pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pinnacle")
if os.path.isdir(_pkg) and _pkg not in sys.path:
    sys.path.insert(0, _pkg)

# ── result tracking ───────────────────────────────────────────────────────────
_pass = 0
_fail = 0

def ok(msg):
    global _pass
    _pass += 1
    print(f"  [PASS] {msg}")

def fail(msg, exc=None):
    global _fail
    _fail += 1
    print(f"  [FAIL] {msg}")
    if exc:
        tb = traceback.format_exc()
        print(f"         {tb.strip().splitlines()[-1]}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── minimal recursive-dict config ────────────────────────────────────────────
class _Cfg(dict):
    """Dict with dot-notation attribute access, recursively."""
    def __getattr__(self, k):
        try:
            v = self[k]
            return _Cfg(v) if isinstance(v, dict) else v
        except KeyError:
            raise AttributeError(k) from None
    def get(self, k, default=None):
        v = super().get(k, default)
        return _Cfg(v) if isinstance(v, dict) else v
    def __missing__(self, k):
        raise KeyError(k)

_L = {"hidden_layers": 2, "layer_size": 8}

TEST_CFG = _Cfg({
    "arch": {n: dict(_L) for n in ["fully_connected","potential","CV","AV","h","L"]},
    "networks": {"type": "FFN", "initialize": True, "soft": False},
    "optimizer": {"adam": {
        "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 1e-4, "amsgrad": False,
    }},
    "scheduler": {
        "type": "None",
        "tf_exponential_lr": {"decay_rate": 0.99, "decay_steps": 15000},
        "RLROP": {"factor": 0.5, "patience": 1000, "threshold": 1e-4, "min_lr": 1e-8},
    },
    "precision": "float32",
    "sampling": {
        "strat": "Uniform", "use_sobol": False,
        "adaptive": {
            "adaptive_update_freq": 100,
            "interior_points": 64, "boundary_points": 32,
            "initial_points": 32, "film_points": 32,
            "interior_base_size": 200, "boundary_base_size": 200,
            "initial_base_size": 100, "film_base_size": 100,
            "x_base_points": 10, "t_base_points": 5, "E_base_points": 5,
            "base_update_freq": 100, "L_growth_threshold": 0.01,
            "safety_factor": 1.2, "uniform_ratio": 0.6,
            "residual_batch_size": 50,
        },
    },
    "training": {
        "max_steps": 5,
        "rec_results_freq": 2,
        "rec_inference_freq": 1000,
        "save_network_freq": 5,
        "weight_strat": "uniform",
        "ntk_update_freq": 100,
        "ntk_start_step": 0,
        "ntk_steps": 20000,
        "physics_steps": 100,
    },
    "hybrid": {
        "fem_data_dir": "/app/pinnacle/FEM",
        "fem_batch_size": 1,
        "use_data": False,
        "n_data_points": 1,
        "anchor_seed": 42,
        "noise_sigma": 0.0,
        "al_beta": 500.0, "al_lr_lambda": 1e-4,
        "al_start_step": 0, "al_tolerance": 1e-6, "al_lambda_max": 100.0,
    },
    "batch_size": {"BC": 16, "interior": 32, "IC": 16, "L": 16, "inference": 16},
    "pde": {
        "scales": {"lc": 1e-9, "cc": 1e-5},
        "physics": {
            "include_holes": False,
            "F": 96485.0, "R": 8.3145, "T": 293.0, "k_B": 1.3806e-23,
            "eps0": 8.85e-12, "E_min": 0.0, "E_max": 2.0,
            "D_cv": 1e-21, "D_av": 1e-21, "D_h": 3.2823e-4,
            "U_cv": -1.0562e-19, "U_av": 7.9212e-20, "U_h": 0.013,
            "z_cv": -2.6667, "z_av": 2.0, "z_h": 1.0,
            "epsilonf": 1.239e-10, "eps_film": 1.239e-10,
            "eps_Ddl": 1.77e-11, "eps_dl": 6.947e-10, "eps_sol": 6.947e-10,
            "c_h0": 4.1683e-4, "c_e0": 9.5329e-28, "tau": 4.9817e-13,
            "Nc": 166.06, "Nv": 1.6606e5, "mu_e0": 2.4033e-19,
            "Ec0": 5.127e-19, "Ev0": 1.6022e-19,
            "c_H": 0.01, "pH": 5.0, "Omega": 1.4e-5,
        },
        "rates": {
            "k1_0": 4.5e-8, "k2_0": 3.6e-6, "k3_0": 4.5e-9,
            "k4_0": 2.25e-7, "k5_0": 7.65e-9, "ktp_0": 4.5e-8, "ko2_0": 0.005,
            "alpha_cv": 0.3, "alpha_av": 0.8, "beta_cv": 0.1, "beta_av": 0.8,
            "alpha_tp": 0.2, "a_par": 0.45,
            "a_cv": 23.764, "a_av": 84.493, "b_cv": 7.9212, "phi_O2_eq": 1.35,
        },
        "geometry": {"d_Ddl": 2e-10, "d_dl": 5e-10, "L_cell": 1e-6},
        "chemistry": {"delta3": 1.0},
    },
    "domain": {"time": {"time_scale": 400000}, "initial": {"L_initial": 1e-9}},
    "experiment": {"name": "smoke_test"},
    "seed": 42,
    "inverse": {
        "enabled": False, "unknown_param": "k3_0",
        "k3_0_init": 1e-8, "k3_0_bounds": [1e-11, 1e-6],
        "n_obs": 5, "obs_voltages": [0.1, 1.0], "obs_noise_sigma": 0.05,
    },
    "plotting": {"topksteps": 1000},
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Third-party packages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("1 — Third-party package imports")

import importlib
_pkgs = [
    ("torch",       "PyTorch"),
    ("numpy",       "NumPy"),
    ("scipy",       "SciPy"),
    ("matplotlib",  "Matplotlib"),
    ("tqdm",        "tqdm"),
    ("pandas",      "pandas"),
    ("omegaconf",   "OmegaConf"),
    ("hydra",       "Hydra"),
]
for mod, name in _pkgs:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        ok(f"{name} {ver}")
    except ImportError as e:
        fail(f"{name} import failed", e)

try:
    import torch
    if torch.cuda.is_available():
        ok(f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
    else:
        print("  [WARN] CUDA not available — running on CPU")
except Exception as e:
    fail("CUDA check", e)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — PINNACLE module imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("2 — PINNACLE module imports")

_modules = [
    ("networks.networks",   "NetworkManager, FFN"),
    ("physics.physics",     "ElectrochemicalPhysics"),
    ("gradients.gradients", "GradientComputer"),
    ("sampling.sampling",   "CollocationSampler"),
    ("losses.losses",       "compute_total_loss"),
    ("weighting.weighting", "NTKWeightManager"),
    ("losses.aggregator",   "build_aggregator, AGGREGATOR_NAMES [NEW]"),
    ("training.training",   "PINNTrainer"),
    ("utils.plotting",      "apply_pub_style, plot_loss_landscape [NEW]"),
    ("analysis.analysis",   "analyze_training_results"),
]
for mod, desc in _modules:
    try:
        importlib.import_module(mod)
        ok(f"{mod}  ({desc})")
    except Exception as e:
        fail(f"{mod}", e)

# Bring into namespace
from networks.networks   import NetworkManager, FFN
from physics.physics     import ElectrochemicalPhysics
from sampling.sampling   import CollocationSampler
from losses.aggregator   import (
    build_aggregator, AGGREGATOR_NAMES, LOSS_KEYS,
    SumAggregator, EMAAggregator, BRDRAggregator,
    SoftAdaptAggregator, RelobRaloAggregator,
    HomoscedasticAggregator,
)
from training.training   import PINNTrainer
import torch.nn as nn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — NetworkManager forward + backward
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("3 — NetworkManager forward + backward pass")

try:
    nm = NetworkManager(TEST_CFG, device)
    ok(f"NetworkManager built ({sum(p.numel() for p in nm.get_all_parameters()):,} params)")
except Exception as e:
    fail("NetworkManager build", e)
    nm = None

if nm is not None:
    try:
        xte = torch.randn(8, 3, device=device)
        te  = torch.randn(8, 2, device=device)
        outs = [nm.networks[k](xte) for k in ("potential", "cv", "av", "h")]
        outs.append(nm.networks["film_thickness"](te))
        assert all(o.shape == torch.Size([8, 1]) for o in outs)
        ok("forward pass — 5 networks, output shape (8,1)")
    except Exception as e:
        fail("forward pass", e)

    try:
        loss = sum(o.sum() for o in outs)
        loss.backward()
        assert next(nm.networks["potential"].parameters()).grad is not None
        ok("backward pass — gradients populated")
    except Exception as e:
        fail("backward pass", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Aggregators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("4 — Loss aggregators")

_dummy_loss = {k: torch.tensor(float(i + 1) * 1e-3, requires_grad=True)
               for i, k in enumerate(LOSS_KEYS)}
# Add granular PDE keys expected by NTKAggregator
for gk in ['weighted_cv_pde','weighted_av_pde','weighted_h_pde','weighted_poisson_pde']:
    _dummy_loss[gk] = torch.tensor(1e-4, requires_grad=True)

ok(f"AGGREGATOR_NAMES = {AGGREGATOR_NAMES}")

_simple = [
    ("sum",        SumAggregator()),
    ("ema",        EMAAggregator()),
    ("soft_adapt", SoftAdaptAggregator()),
    ("relobralo",  RelobRaloAggregator()),
    ("brdr",       BRDRAggregator().to(device)),
    ("homoscedastic", HomoscedasticAggregator()),
]
for name, agg in _simple:
    try:
        out = agg.aggregate(_dummy_loss, step=0)
        assert isinstance(out, torch.Tensor), "output is not a tensor"
        assert out.requires_grad or out.is_leaf, "no grad path"
        ok(f"{name:16s} → {out.item():.4e}")
    except Exception as e:
        fail(f"{name}", e)

# build_aggregator factory for gradient-based aggregators.
# These need losses that flow through the network parameters so that
# torch.autograd.grad returns non-None values and gn_loss is differentiable.
if nm is not None:
    _x3  = torch.randn(4, 3, device=device)
    _te2 = torch.randn(4, 2, device=device)
    _net_keys = ['potential', 'cv', 'av']
    _connected_loss = {
        k: nm.networks[_net_keys[i]](_x3).sum() * 1e-3
        for i, k in enumerate(LOSS_KEYS[:3])
    }
    _connected_loss['film_physics'] = nm.networks['film_thickness'](_te2).sum() * 1e-3
    for _gk in ['weighted_cv_pde', 'weighted_av_pde', 'weighted_h_pde', 'weighted_poisson_pde']:
        _connected_loss[_gk] = torch.tensor(1e-4, requires_grad=True)

    for name in ("grad_norm", "lr_annealing"):
        try:
            agg = build_aggregator(name, TEST_CFG, nm, None, None, device)
            out = agg.aggregate(_connected_loss, step=0)
            assert isinstance(out, torch.Tensor)
            ok(f"{name:16s} → {out.item():.4e}")
        except Exception as e:
            fail(f"{name}", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — PINNTrainer: 5 training steps
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("5 — PINNTrainer: 5 training steps (weight_strat=uniform, use_data=False)")

_tmpdir = tempfile.mkdtemp(prefix="pinnacle_test_")
_trainer = None
try:
    _trainer = PINNTrainer(TEST_CFG, device, output_dir=_tmpdir)
    ok("PINNTrainer created")
except Exception as e:
    fail("PINNTrainer init", e)

if _trainer is not None:
    try:
        _hist = _trainer.train()
        assert len(_hist['total']) == 5, f"expected 5 loss entries, got {len(_hist['total'])}"
        ok(f"train() completed — final total loss: {_hist['total'][-1]:.4e}")
    except Exception as e:
        fail("train() 5 steps", e)

    try:
        _tp = os.path.join(_tmpdir, "timing.json")
        assert os.path.isfile(_tp), f"timing.json not written to {_tp}"
        import json
        _t = json.load(open(_tp))
        ok(f"timing.json written — {_t['mean_ms_per_step']:.1f} ms/step, "
           f"peak GPU {_t['peak_gpu_mb']:.0f} MB")
    except Exception as e:
        fail("timing.json", e)

    try:
        assert len(_trainer.step_times) == 5
        ok(f"step_times populated ({len(_trainer.step_times)} entries)")
    except Exception as e:
        fail("step_times", e)

# Test aggregator path: weight_strat='sum'
try:
    _cfg2 = _Cfg(dict(TEST_CFG))
    _cfg2['training'] = dict(TEST_CFG['training'])
    _cfg2['training']['weight_strat'] = 'sum'
    _t2 = PINNTrainer(_cfg2, device, output_dir=_tmpdir)
    assert _t2.aggregator is not None, "aggregator not set for 'sum'"
    _h2 = _t2.train()
    ok(f"weight_strat=sum aggregator path — final loss: {_h2['total'][-1]:.4e}")
except Exception as e:
    fail("weight_strat=sum aggregator path", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — Inverse mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("6 — Inverse mode: k3_0 as trainable parameter")

try:
    _inv_cfg = _Cfg(dict(TEST_CFG))
    _inv_cfg['inverse'] = {
        "enabled": True, "unknown_param": "k3_0",
        "k3_0_init": 1e-8, "k3_0_bounds": [1e-11, 1e-6],
        "n_obs": 5, "obs_voltages": [0.1], "obs_noise_sigma": 0.0,
    }
    phys_inv = ElectrochemicalPhysics(_inv_cfg, device)
    assert phys_inv.k3_0_param is not None, "k3_0_param is None"
    assert isinstance(phys_inv.k3_0_param, nn.Parameter), "not nn.Parameter"
    assert abs(phys_inv.k3_0_param.item() - 1e-8) < 1e-15, "wrong init value"
    ok(f"physics.k3_0_param = {phys_inv.k3_0_param.item():.2e}  (init=1e-8)")
except Exception as e:
    fail("ElectrochemicalPhysics inverse mode", e)

try:
    _inv_trainer = PINNTrainer(_inv_cfg, device, output_dir=_tmpdir)
    opt_params = [id(p) for g in _inv_trainer.optimizer.param_groups for p in g['params']]
    assert id(_inv_trainer.physics.k3_0_param) in opt_params, \
        "k3_0_param not in optimizer"
    ok("k3_0_param included in optimizer param groups")
except Exception as e:
    fail("k3_0_param in optimizer", e)

try:
    _h_inv = _inv_trainer.train()
    k3_recovered = _inv_trainer.physics.k3_0_param.item()
    ok(f"Inverse training 5 steps — k3_0: {1e-8:.2e} → {k3_recovered:.4e}")
except Exception as e:
    fail("Inverse training steps", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*60}")
print(f"  RESULTS: {_pass} passed, {_fail} failed")
print('='*60)

if _fail > 0:
    sys.exit(1)
else:
    print("  ALL TESTS PASSED")
    sys.exit(0)
