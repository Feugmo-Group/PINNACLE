#!/usr/bin/env python3
"""Inverse-recovery figure: learnable parameter value vs training step, showing
k2_0 converging to its true value while D_cv does not. Reads loss_history from
the inverse-run checkpoints. Run in Docker (needs torch)."""
import glob, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def traj(pattern, key):
    d = sorted(glob.glob(pattern))[-1]
    ck = torch.load(os.path.join(d, "checkpoints/final_model.pt"),
                    map_location="cpu", weights_only=False)
    v = np.array(ck["loss_history"][key], dtype=float)
    step = np.arange(len(v))
    ok = np.isfinite(v)
    return step[ok], v[ok]

k2_s_x, k2_s = traj("outputs/experiments/e6_recover_k2_0/*", "k2_0")
k2_m_x, k2_m = traj("outputs/experiments/e6_recover_k2_0_multiV/*", "k2_0")
dcv_x, dcv = traj("outputs/experiments/e6_recover_D_cv/*", "D_cv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) k2_0
ax[0].plot(k2_s_x, k2_s, color="steelblue", lw=1.6, label="1 voltage")
ax[0].plot(k2_m_x, k2_m, color="darkorange", lw=1.6, label="3 voltages")
ax[0].axhline(3.6e-6, color="k", ls="--", lw=1.3, label="true value")
ax[0].axhline(3.6e-5, color="grey", ls=":", lw=1.2, label="initial guess")
ax[0].set_yscale("log"); ax[0].set_title(r"(a)  $k_2^0$ (boundary-stiff)")
ax[0].set_xlabel("Training step"); ax[0].set_ylabel(r"recovered $k_2^0$  [mol\,m$^{-2}$s$^{-1}$]")
ax[0].legend(fontsize=9); ax[0].grid(True, alpha=0.3)

# (b) D_cv
ax[1].plot(dcv_x, dcv, color="crimson", lw=1.6, label="recovered")
ax[1].axhline(1.0e-21, color="k", ls="--", lw=1.3, label="true value")
ax[1].axhline(5.0e-21, color="grey", ls=":", lw=1.2, label="initial guess")
ax[1].set_yscale("log"); ax[1].set_title(r"(b)  $D_\mathrm{CV}$ (interior-stiff)")
ax[1].set_xlabel("Training step"); ax[1].set_ylabel(r"recovered $D_\mathrm{CV}$  [m$^2$s$^{-1}$]")
ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("paper/APL_PDM-PINN_review/images/fig_inverse_recovery.pdf", dpi=300, bbox_inches="tight")
fig.savefig("/tmp/fig_inverse_recovery.png", dpi=130, bbox_inches="tight")
print("k2_0 1V: final", f"{k2_s[-1]:.2e}", " 3V:", f"{k2_m[-1]:.2e}", " D_cv:", f"{dcv[-1]:.2e}")
print("saved fig_inverse_recovery.pdf")
