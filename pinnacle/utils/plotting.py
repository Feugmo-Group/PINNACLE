# utils/plotting.py
"""
Publication-quality visualisation utilities for PINNACLE.

Public API
----------
apply_pub_style()
    Apply publication-quality matplotlib rcParams globally (fonts ≥ 10 pt).
    Call once at the start of any analysis script or notebook.

plot_loss_landscape(networks, compute_loss_fn, save_path, ...)
    2-D random-direction total loss landscape around the trained parameters.
    Saves a single 3-D surface of log10(total loss).

plot_loss_landscape_components(networks, compute_loss_components_fn, save_path, ...)
    2-D random-direction landscape for PINNACLE's 4 main loss categories
    (interior, boundary, initial, film_physics) + total.
    Layout: 2×3 figure (5 surface panels + 1 summary text panel).

plot_loss_landscape_pde(networks, compute_loss_components_fn, save_path, ...)
    Sub-component landscape for the 4 PDE residual terms (CV, AV, hole,
    Poisson) + their sum.  Layout: 1×5.

plot_loss_landscape_bc(networks, compute_loss_components_fn, save_path, ...)
    Sub-component landscape for the 7 boundary-condition terms (CV m/f,
    AV m/f, φ m/f, CV f/s, AV f/s, φ f/s, hole f/s) + their sum.
    Layout: 2×4.

All landscape functions use:
  • Filter-normalised orthogonal random directions (Li et al. 2018;
    Basir 2023).
  • log10 colour scale on 3-D surfaces with contour floor projection.
  • torch.nn.utils.{parameters_to_vector, vector_to_parameters} for
    clean multi-network parameter snapshots.

References
----------
Li, H., et al. (2018). "Visualizing the Loss Landscape of Neural Nets."
    NeurIPS. arXiv:1712.09913.
Basir, S. (2023). "Investigating and Mitigating Failure Modes in PINNs."
    Commun. Comput. Phys. 33, 1240–1269. arXiv:2209.09988.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Callable, Dict, Optional

__all__ = [
    "apply_pub_style",
    "plot_loss_landscape",
    "plot_loss_landscape_components",
    "plot_loss_landscape_pde",
    "plot_loss_landscape_bc",
]


# ── Publication style ─────────────────────────────────────────────────────────

def apply_pub_style() -> None:
    """
    Apply publication-quality matplotlib rcParams globally.

    Sets font sizes ≥ 10 pt (APL Machine Learning single-column requirement),
    removes top/right spines, and increases default line width.  Call once
    before generating any figure.
    """
    import matplotlib
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          12,
        "axes.titlesize":     12,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         150,
        "lines.linewidth":    1.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


# ── Internal helpers ──────────────────────────────────────────────────────────

def _all_params(networks):
    """Yield all parameters from a networks dict or NetworkManager."""
    nets = networks.networks.values() if hasattr(networks, 'networks') else networks.values()
    for net in nets:
        yield from net.parameters()


def _snapshot(networks: Dict[str, nn.Module]) -> torch.Tensor:
    """Return a flat clone of all network parameters."""
    return parameters_to_vector(_all_params(networks)).clone()


def _restore(networks: Dict[str, nn.Module], theta: torch.Tensor) -> None:
    """Restore all network parameters from a flat vector."""
    vector_to_parameters(theta, _all_params(networks))


def _orthogonal_directions(theta: torch.Tensor, range_scale: float):
    """
    Two orthogonal random directions scaled to range_scale * ‖θ‖.

    d1 is a random unit vector scaled to range_scale * ‖θ‖.
    d2 is orthogonalised to d1 (Gram-Schmidt) then scaled identically.
    Seed is fixed at 0 so landscapes are reproducible across calls.
    """
    norm = theta.norm().item() + 1e-12
    torch.manual_seed(0)
    d1 = torch.randn_like(theta)
    d2 = torch.randn_like(theta)
    d1 = (d1 / d1.norm()) * norm * range_scale
    d2 = d2 - (d2.dot(d1) / (d1.dot(d1) + 1e-12)) * d1
    d2 = (d2 / (d2.norm() + 1e-12)) * norm * range_scale
    return d1, d2


def _make_grids(nr_steps: int):
    """Return (alphas, betas, a_g, b_g) for the landscape scan."""
    alphas = np.linspace(-1.0, 1.0, nr_steps)
    betas  = np.linspace(-1.0, 1.0, nr_steps)
    a_g, b_g = np.meshgrid(alphas, betas, indexing="ij")
    return alphas, betas, a_g, b_g


def _surface_3d(ax, a_g, b_g, surf_data, cmap: str, label: str, fig):
    """
    Plot a single log10 3-D surface on *ax* with a colourbar.

    Returns the colourbar artist so the caller can style it further.
    """
    log_s = np.log10(np.maximum(surf_data, 1e-20))
    sp = ax.plot_surface(
        a_g, b_g, log_s,
        alpha=0.88, cmap=cmap,
        antialiased=True, edgecolor="none",
        linewidth=0, shade=True,
    )
    ax.contourf(a_g, b_g, log_s,
                zdir="z", offset=log_s.min() - 0.2, cmap=cmap, alpha=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(r"$\epsilon_1$", fontsize=8)
    ax.set_ylabel(r"$\epsilon_2$", fontsize=8)
    ax.set_zlabel(r"$\log_{10}(L)$", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    ax.tick_params(axis="z", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.view_init(elev=25, azim=-45)
    cbar = fig.colorbar(sp, ax=ax, shrink=0.45, aspect=10, pad=0.12)
    cbar.ax.tick_params(labelsize=6)
    return cbar


def _scan(networks, theta, d1, d2, alphas, betas, keys,
          compute_fn, verbose_prefix="landscape"):
    """
    Run the landscape grid scan for *keys* and return a dict of surfaces.

    *compute_fn* must return a dict keyed by *keys* for the current weights.
    """
    nr_steps = len(alphas)
    surfs = {k: np.zeros((nr_steps, nr_steps)) for k in keys}
    total = nr_steps * nr_steps
    log10 = max(1, total // 10)

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            a_d1 = alpha * d1
            for j, beta in enumerate(betas):
                _restore(networks, theta + a_d1 + beta * d2)
                comp = compute_fn()
                for k in keys:
                    surfs[k][i, j] = float(comp.get(k, 0.0))
                done = i * nr_steps + j + 1
                if done % log10 == 0:
                    tot = sum(surfs[k][i, j] for k in keys)
                    print(f"    {verbose_prefix} {100*done//total:3d}%  "
                          f"sum={tot:.3e}")
        _restore(networks, theta)
    return surfs


# ── Public functions ──────────────────────────────────────────────────────────

def plot_loss_landscape(
    networks: Dict[str, nn.Module],
    compute_loss_fn: Callable[[], float],
    save_path: str,
    range_scale: float = 0.5,
    nr_steps: int = 28,
    title: str = "PINNACLE Loss Landscape",
) -> None:
    """
    2-D random-direction total loss landscape around the trained model.

    Perturbs all network parameters along two orthogonal filter-normalised
    random directions and evaluates the total loss at each grid point.
    Saves a 3-D surface of log10(total_loss) with a contour floor.

    Parameters
    ----------
    networks : dict[str, nn.Module]
        Trained PINNACLE networks (potential, cv, av, h, film_thickness).
    compute_loss_fn : () → float
        No-argument closure returning the scalar total loss for the **current**
        network weights.  Collocation points should be fixed before calling
        this function to avoid resampling inside the landscape loop.
    save_path : str
        Output PNG path; parent directory is created if it does not exist.
    range_scale : float
        Perturbation radius as a fraction of ‖θ‖.  0.5 is a good default
        for a well-trained model; increase to 1.0 to see wider structure.
    nr_steps : int
        Grid points per axis (nr_steps² total loss evaluations).
    title : str
        Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    theta = _snapshot(networks)
    d1, d2 = _orthogonal_directions(theta, range_scale)
    alphas, betas, a_g, b_g = _make_grids(nr_steps)

    surfs = _scan(networks, theta, d1, d2, alphas, betas,
                  ["total"],
                  lambda: {"total": float(compute_loss_fn())},
                  verbose_prefix="total landscape")
    surface = surfs["total"]
    log_surf = np.log10(np.maximum(surface, 1e-20))

    fig = plt.figure(figsize=(7, 5), dpi=150)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\epsilon_1$", fontsize=9)
    ax.set_ylabel(r"$\epsilon_2$", fontsize=9)
    ax.set_zlabel(r"$\log_{10}$(Loss)", fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="z",    labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.view_init(elev=25, azim=-45)

    surf_p = ax.plot_surface(a_g, b_g, log_surf, cmap="inferno",
                             alpha=0.85, antialiased=True,
                             edgecolor="none", linewidth=0, shade=True)
    ax.contourf(a_g, b_g, log_surf,
                zdir="z", offset=log_surf.min() - 0.2,
                cmap="inferno", alpha=0.5)
    cbar = fig.colorbar(surf_p, ax=ax, shrink=0.5, aspect=11, pad=0.13)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"     {os.path.basename(save_path)}")


def plot_loss_landscape_components(
    networks: Dict[str, nn.Module],
    compute_loss_components_fn: Callable[[], Dict[str, float]],
    save_path: str,
    range_scale: float = 0.5,
    nr_steps: int = 24,
    title: str = "PINNACLE Loss Landscape — Main Components",
) -> None:
    """
    2-D random-direction landscape for PINNACLE's 4 main loss categories + total.

    Layout: 2×3 figure
        [Interior (PDE)]  [Boundary]       [Initial]
        [Film-growth ODE] [Total]          [Summary text]

    Directly addresses APL reviewer R2 comment 3: quantitative per-component
    loss landscape identifies which constraint dominates training difficulty.

    Parameters
    ----------
    networks : dict[str, nn.Module]
        Trained PINNACLE networks.
    compute_loss_components_fn : () → dict[str, float]
        No-argument closure returning a dict with at least the keys
        ``{'interior', 'boundary', 'initial', 'film_physics'}``
        for the **current** network weights.
    save_path : str
        Output PNG path.
    range_scale : float
        Perturbation radius as a fraction of ‖θ‖.
    nr_steps : int
        Grid points per axis.
    title : str
        Figure suptitle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    keys   = ["interior", "boundary", "initial", "film_physics"]
    cmaps  = {"interior": "viridis", "boundary": "inferno",
               "initial": "plasma",  "film_physics": "magma"}
    labels = {"interior": "Interior (PDE)", "boundary": "Boundary conditions",
               "initial": "Initial conditions", "film_physics": "Film-growth ODE"}

    theta = _snapshot(networks)
    d1, d2 = _orthogonal_directions(theta, range_scale)
    alphas, betas, a_g, b_g = _make_grids(nr_steps)

    surfs = _scan(networks, theta, d1, d2, alphas, betas, keys,
                  compute_loss_components_fn,
                  verbose_prefix="components landscape")
    total_surf = sum(surfs[k] for k in keys)

    fig = plt.figure(figsize=(18, 10), dpi=120)
    fig.suptitle(title, fontsize=13, y=1.01)

    for idx, k in enumerate(keys):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        _surface_3d(ax, a_g, b_g, surfs[k], cmaps[k], labels[k], fig)

    # 5th panel: total
    ax = fig.add_subplot(2, 3, 5, projection="3d")
    _surface_3d(ax, a_g, b_g, total_surf, "turbo", "Total loss", fig)

    # 6th panel: numeric summary
    ax_txt = fig.add_subplot(2, 3, 6)
    ax_txt.axis("off")
    lines = [r"$\log_{10}$ minimum per component:", ""]
    for k in keys:
        mn = np.log10(max(float(surfs[k].min()), 1e-20))
        lines.append(f"  {labels[k]:30s}: {mn:+.2f}")
    mn_tot = np.log10(max(float(total_surf.min()), 1e-20))
    lines.append("")
    lines.append(f"  {'Total':30s}: {mn_tot:+.2f}")
    ax_txt.text(0.04, 0.96, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=9, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"     {os.path.basename(save_path)}")


def plot_loss_landscape_pde(
    networks: Dict[str, nn.Module],
    compute_loss_components_fn: Callable[[], Dict[str, float]],
    save_path: str,
    range_scale: float = 0.5,
    nr_steps: int = 24,
    title: str = "PINNACLE Loss Landscape — PDE Sub-components",
) -> None:
    """
    2-D random-direction landscape for the 4 individual NP+Poisson residuals + total.

    Layout: 1×5 figure
        [CV NP]  [AV NP]  [Hole NP]  [Poisson]  [Sum]

    Parameters
    ----------
    networks : dict[str, nn.Module]
        Trained PINNACLE networks.
    compute_loss_components_fn : () → dict[str, float]
        No-argument closure returning a dict with at least the keys
        ``{'weighted_cv_pde', 'weighted_av_pde', 'weighted_h_pde',
            'weighted_poisson_pde'}`` for the **current** weights.
    save_path : str
        Output PNG path.
    range_scale : float
        Perturbation radius as a fraction of ‖θ‖.
    nr_steps : int
        Grid points per axis.
    title : str
        Figure suptitle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    keys   = ["weighted_cv_pde", "weighted_av_pde",
               "weighted_h_pde", "weighted_poisson_pde"]
    cmaps  = ["viridis", "plasma", "magma", "cividis"]
    labels = ["CV — Nernst-Planck", "AV — Nernst-Planck",
               "Hole — Nernst-Planck", "Poisson"]

    theta = _snapshot(networks)
    d1, d2 = _orthogonal_directions(theta, range_scale)
    alphas, betas, a_g, b_g = _make_grids(nr_steps)

    surfs = _scan(networks, theta, d1, d2, alphas, betas, keys,
                  compute_loss_components_fn,
                  verbose_prefix="PDE landscape")
    total_surf = sum(surfs[k] for k in keys)

    fig = plt.figure(figsize=(22, 5), dpi=120)
    fig.suptitle(title, fontsize=13)

    for idx, (k, cmap, lbl) in enumerate(zip(keys, cmaps, labels)):
        ax = fig.add_subplot(1, 5, idx + 1, projection="3d")
        _surface_3d(ax, a_g, b_g, surfs[k], cmap, lbl, fig)

    ax = fig.add_subplot(1, 5, 5, projection="3d")
    _surface_3d(ax, a_g, b_g, total_surf, "turbo", "Total PDE loss", fig)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"     {os.path.basename(save_path)}")


def plot_loss_landscape_bc(
    networks: Dict[str, nn.Module],
    compute_loss_components_fn: Callable[[], Dict[str, float]],
    save_path: str,
    range_scale: float = 0.5,
    nr_steps: int = 20,
    title: str = "PINNACLE Loss Landscape — Boundary-Condition Sub-components",
) -> None:
    """
    2-D random-direction landscape for PINNACLE's 7 BC terms + total.

    Layout: 2×4 figure
        [CV m/f]   [AV m/f]   [φ m/f]   [CV f/s]
        [AV f/s]   [φ f/s]    [hole f/s] [Total BC]

    The anion-vacancy film/solution BC is expected to dominate (Fig. 8 in the
    paper); this figure provides per-BC quantitative diagnostics to address
    APL reviewer R2 comment 3.

    Parameters
    ----------
    networks : dict[str, nn.Module]
        Trained PINNACLE networks.
    compute_loss_components_fn : () → dict[str, float]
        No-argument closure returning a dict with at least the keys
        ``{'weighted_cv_mf_bc', 'weighted_av_mf_bc', 'weighted_u_mf_bc',
            'weighted_cv_fs_bc', 'weighted_av_fs_bc', 'weighted_u_fs_bc',
            'weighted_h_fs_bc'}`` for the **current** weights.
    save_path : str
        Output PNG path.
    range_scale : float
        Perturbation radius as a fraction of ‖θ‖.
    nr_steps : int
        Grid points per axis.  Default is 20 (400 evaluations) — lower than
        the other functions because there are 8 panels to fill.
    title : str
        Figure suptitle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    keys   = ["weighted_cv_mf_bc", "weighted_av_mf_bc", "weighted_u_mf_bc",
               "weighted_cv_fs_bc", "weighted_av_fs_bc", "weighted_u_fs_bc",
               "weighted_h_fs_bc"]
    cmaps  = ["viridis", "plasma", "magma",
               "cividis", "inferno", "Blues", "Oranges"]
    labels = ["CV (m/f)", "AV (m/f)", r"$\varphi$ (m/f)",
               "CV (f/s)", "AV (f/s)", r"$\varphi$ (f/s)", r"$h^+$ (f/s)"]

    theta = _snapshot(networks)
    d1, d2 = _orthogonal_directions(theta, range_scale)
    alphas, betas, a_g, b_g = _make_grids(nr_steps)

    surfs = _scan(networks, theta, d1, d2, alphas, betas, keys,
                  compute_loss_components_fn,
                  verbose_prefix="BC landscape")
    total_surf = sum(surfs[k] for k in keys)

    fig = plt.figure(figsize=(24, 10), dpi=120)
    fig.suptitle(title, fontsize=13, y=1.01)

    for idx, (k, cmap, lbl) in enumerate(zip(keys, cmaps, labels)):
        ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
        _surface_3d(ax, a_g, b_g, surfs[k], cmap, lbl, fig)

    ax = fig.add_subplot(2, 4, 8, projection="3d")
    _surface_3d(ax, a_g, b_g, total_surf, "turbo", "Total BC loss", fig)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"     {os.path.basename(save_path)}")
