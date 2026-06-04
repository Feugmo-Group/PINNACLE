# losses/aggregator.py
"""
Unified loss aggregator for PINNACLE.

All aggregators share a common interface::

    aggregator = build_aggregator('ema', config, networks, physics, sampler, device)

    # In training loop — call compute_total_loss with uniform weights first:
    loss_dict = compute_total_loss(..., weights=UNIFORM_WEIGHTS, ntk_weights=None)

    # Then aggregate:
    total = aggregator.aggregate(loss_dict, step)
    total.backward()

The ``loss_dict`` expected by every aggregator is the dict returned by
``compute_total_loss`` when called with uniform weights.  Each aggregator
picks the keys it needs:

* Simple aggregators  (EMA, SoftAdapt, ReLoBRaLo, BRDR, GradNorm, …)
  work on the four coarse scalars::

      {'interior', 'boundary', 'initial', 'film_physics'}

* ``NTKAggregator`` works on the granular per-component scalars::

      {'weighted_cv_pde', 'weighted_av_pde', 'weighted_h_pde',
       'weighted_poisson_pde', 'boundary', 'initial', 'film_physics'}

  These are the *unweighted* component losses when compute_total_loss is
  called with uniform weights (the ``weighted_`` prefix reflects that the
  batch-size normalisation has already been applied inside compute_total_loss).

Available strategies
--------------------
``'sum'``           Fixed weighted sum (default baseline).
``'batch_size'``    1/N_batch normalisation per term.
``'ema'``           EMA scale balancing (ResNorm equivalent, no gradients).
``'soft_adapt'``    SoftAdapt (Heydari et al. 2019).
``'relobralo'``     ReLoBRaLo (Bischof & Kraus 2021).
``'brdr'``          Balanced Residual Decay Rate (Chen et al. 2025).
``'grad_norm'``     GradNorm (Chen et al. 2018) — gradient-based.
``'lr_annealing'``  LR annealing (Wang et al. 2020) — gradient-based.
``'homoscedastic'`` Homoscedastic uncertainty (Kendall & Gal 2018).
``'ntk'``           Existing PINNACLE NTK (Jacobian-based, granular weights).

References
----------
Bischof & Kraus (2021), arXiv:2110.09813.
Chen, Howard & Stinis (2025), J. Comput. Phys. 542, 114226.
Chen, Badrinarayanan & Andrew (2018), arXiv:1711.02132.
Heydari et al. (2019), arXiv:1912.12355.
Jacot, Gabriel & Hongler (2020), arXiv:1806.07572.
Kendall & Gal (2018), arXiv:1705.07115.
Wang, Teng & Perdikaris (2021), arXiv:2001.04536.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

# Loss components the coarse aggregators operate on.
LOSS_KEYS: List[str] = ['interior', 'boundary', 'initial', 'film_physics']

# Granular PDE keys used by NTKAggregator.
NTK_PDE_KEYS: List[str] = [
    'weighted_cv_pde', 'weighted_av_pde',
    'weighted_h_pde',  'weighted_poisson_pde',
]
NTK_COARSE_KEYS: List[str] = ['boundary', 'initial', 'film_physics']
# Short name → key used in NTKWeightManager output
NTK_WEIGHT_MAP: Dict[str, str] = {
    'weighted_cv_pde':      'cv_pde',
    'weighted_av_pde':      'av_pde',
    'weighted_h_pde':       'h_pde',
    'weighted_poisson_pde': 'poisson_pde',
    'boundary':             'boundary',
    'initial':              'initial',
    'film_physics':         'film_physics',
}

# Uniform initial weights (passed to compute_total_loss so every component
# enters with weight 1.0 before the aggregator applies dynamic weights).
UNIFORM_WEIGHTS: Dict[str, float] = {k: 1.0 for k in LOSS_KEYS}


# ── Base class ────────────────────────────────────────────────────────────────

class PINNAggregator(nn.Module):
    """
    Abstract base for PINNACLE loss aggregators.

    Subclasses implement ``aggregate(loss_dict, step) -> Tensor``.
    ``loss_dict`` is the dict returned by ``compute_total_loss`` with
    uniform weights; ``step`` is the current optimiser step (0-indexed).
    """

    @property
    def current_weights(self) -> Optional[Dict[str, float]]:
        """Return current effective per-key weights for logging, or None."""
        return None

    def aggregate(
        self, loss_dict: Dict[str, torch.Tensor], step: int
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _add_obs(total: torch.Tensor, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Add data terms unweighted so FEM anchors and inverse obs are always enforced.

        Checks both 'obs_loss' (inverse-problem path) and 'data_loss' (hybrid-training
        path). Skips non-finite values to avoid NaN propagation.
        """
        for key in ('obs_loss', 'data_loss'):
            val = loss_dict.get(key)
            if val is not None and isinstance(val, torch.Tensor) and val.isfinite():
                total = total + val
        return total


# ── Fixed weighted sum ────────────────────────────────────────────────────────

class SumAggregator(PINNAggregator):
    """
    Fixed weighted sum.

    ``total = w_interior * L_int + w_boundary * L_bc
              + w_initial * L_ic + w_film * L_film``

    This is the baseline; no dynamic rebalancing.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self._w = weights or {k: 1.0 for k in LOSS_KEYS}

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._w)

    def aggregate(self, loss_dict, step):
        return sum(self._w[k] * loss_dict[k] for k in LOSS_KEYS)


# ── Batch-size normalisation ──────────────────────────────────────────────────

class BatchSizeAggregator(PINNAggregator):
    """
    Scale each loss component by 1 / N_batch so that larger batches do not
    inflate a term's effective weight.
    """

    def __init__(self, batch_sizes: Dict[str, int]):
        super().__init__()
        self._w = {
            'interior':    1.0 / max(batch_sizes.get('interior', 1), 1),
            'boundary':    1.0 / max(batch_sizes.get('BC', 1), 1),
            'initial':     1.0 / max(batch_sizes.get('IC', 1), 1),
            'film_physics':1.0 / max(batch_sizes.get('L', 1), 1),
        }

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._w)

    def aggregate(self, loss_dict, step):
        return self._add_obs(sum(self._w[k] * loss_dict[k] for k in LOSS_KEYS if k in loss_dict), loss_dict)


# ── EMA scale balancing ───────────────────────────────────────────────────────

class EMAAggregator(PINNAggregator):
    """
    EMA scale balancing (gradient-free).

    Tracks the exponential moving average of each loss and rescales so
    every component contributes roughly the same order of magnitude as the
    interior (PDE) loss::

        w_k = clamp( ema_interior / ema_k, clamp_lo, clamp_hi )

    ``ema_alpha = 0.99`` is a good default; lower values adapt faster
    but are noisier.
    """

    def __init__(
        self,
        ema_alpha: float = 0.99,
        clamp_lo: float = 1e-3,
        clamp_hi: float = 1e3,
    ):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self._ema: Optional[Dict[str, float]] = None
        self._w: Dict[str, float] = {k: 1.0 for k in LOSS_KEYS}

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._w)

    def aggregate(self, loss_dict, step):
        curr = {k: float(loss_dict[k].detach()) + 1e-30 for k in LOSS_KEYS}

        if self._ema is None:
            self._ema = dict(curr)
        else:
            a = self.ema_alpha
            self._ema = {
                k: a * self._ema[k] + (1 - a) * curr[k]
                for k in LOSS_KEYS
            }

        ref = self._ema['interior']
        self._w = {
            k: max(min(ref / (self._ema[k] + 1e-30), self.clamp_hi), self.clamp_lo)
            for k in LOSS_KEYS
        }
        return self._add_obs(sum(self._w[k] * loss_dict[k] for k in LOSS_KEYS if k in loss_dict), loss_dict)


# ── SoftAdapt ─────────────────────────────────────────────────────────────────

class SoftAdaptAggregator(PINNAggregator):
    """
    SoftAdapt (Heydari et al. 2019, arXiv:1912.12355).

    Weights proportional to softmax of the rate of change of each loss::

        w_k = softmax( beta * dL_k / L_k )

    Terms that are improving fastest receive less attention.
    ``beta`` controls sharpness (higher = more focus on lagging terms).
    """

    def __init__(
        self,
        beta: float = 0.1,
        epsilon: float = 1e-7,
        update_freq: int = 1,
    ):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.update_freq = update_freq
        self._prev: Optional[Dict[str, float]] = None
        self._w: Dict[str, float] = {k: 1.0 / len(LOSS_KEYS) for k in LOSS_KEYS}

    @property
    def current_weights(self) -> Dict[str, float]:
        return {k: self._w[k] for k in LOSS_KEYS}

    def aggregate(self, loss_dict, step):
        curr = {k: float(loss_dict[k].detach()) for k in LOSS_KEYS}

        if self._prev is not None and step % self.update_freq == 0:
            rates = {
                k: (curr[k] - self._prev[k]) / (abs(self._prev[k]) + self.epsilon)
                for k in LOSS_KEYS
            }
            r_t = torch.tensor([rates[k] for k in LOSS_KEYS], dtype=torch.float32)
            w_t = torch.softmax(self.beta * r_t, dim=0).tolist()
            self._w = {k: w for k, w in zip(LOSS_KEYS, w_t)}

        self._prev = curr
        total_w = sum(self._w.values()) + 1e-30
        return self._add_obs(sum((self._w[k] / total_w) * loss_dict[k] for k in LOSS_KEYS if k in loss_dict), loss_dict)


# ── ReLoBRaLo ─────────────────────────────────────────────────────────────────

class RelobRaloAggregator(PINNAggregator):
    """
    Relative Loss Balancing Residual Algorithm (Bischof & Kraus 2021,
    arXiv:2110.09813).

    Tracks relative progress from initial loss values::

        ratio_k = L_k(0) / L_k(t)
        w_hat_k = ratio_k / mean(ratio_j)
        w_k(t)  = rho * w_k(t-1) + (1-rho) * w_hat_k

    ``rho`` close to 1 gives slow, stable adaptation.
    """

    def __init__(
        self,
        rho: float = 0.999,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        self.rho = rho
        self.epsilon = epsilon
        self._initial: Optional[Dict[str, float]] = None
        self._w: Dict[str, float] = {k: 1.0 for k in LOSS_KEYS}

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._w)

    def aggregate(self, loss_dict, step):
        curr = {k: float(loss_dict[k].detach()) + self.epsilon for k in LOSS_KEYS}

        if self._initial is None:
            self._initial = dict(curr)

        ratios = {k: self._initial[k] / curr[k] for k in LOSS_KEYS}
        mean_r = sum(ratios.values()) / len(ratios) + self.epsilon
        w_hat  = {k: ratios[k] / mean_r for k in LOSS_KEYS}
        rho = self.rho
        self._w = {
            k: rho * self._w[k] + (1 - rho) * w_hat[k]
            for k in LOSS_KEYS
        }
        return self._add_obs(sum(self._w[k] * loss_dict[k] for k in LOSS_KEYS if k in loss_dict), loss_dict)


# ── BRDR ──────────────────────────────────────────────────────────────────────

class BRDRAggregator(PINNAggregator):
    """
    Balanced Residual Decay Rate (Chen, Howard & Stinis 2025,
    J. Comput. Phys. 542, 114226).

    Uses the inverse relative decay rate normalised by a running EMA::

        irdr_k   = L_k / sqrt( EMA(L_k²) )
        w_hat_k  = irdr_k / mean(irdr)
        w_k(t)   = EMA_w(w_hat_k),   bias-corrected & normalised to mean = 1

    ``beta_c``: EMA decay for the squared loss tracker.
    ``beta_w``: EMA decay for the weight smoother.
    """

    def __init__(
        self,
        beta_c: float = 0.999,
        beta_w: float = 0.999,
        eps: float = 1e-14,
    ):
        super().__init__()
        self.beta_c = beta_c
        self.beta_w = beta_w
        self.eps = eps
        n = len(LOSS_KEYS)
        self.register_buffer("_l2_ema",   torch.zeros(n))
        self.register_buffer("_w_ema",    torch.ones(n))

    @property
    def current_weights(self) -> Dict[str, float]:
        return {k: float(self._w_ema[i]) for i, k in enumerate(LOSS_KEYS)}

    def aggregate(self, loss_dict, step):
        n   = step + 1
        vals = torch.stack([loss_dict[k] for k in LOSS_KEYS])

        if step == 0:
            with torch.no_grad():
                self._l2_ema.copy_(vals.detach() ** 2)
            return vals.sum()

        with torch.no_grad():
            self._l2_ema.mul_(self.beta_c).add_((1.0 - self.beta_c) * vals.detach() ** 2)
            l2_bc = self._l2_ema / (1.0 - self.beta_c ** n)
            irdr  = vals.detach() / (torch.sqrt(l2_bc) + self.eps)
            w_hat = irdr / (irdr.mean() + self.eps)
            self._w_ema.mul_(self.beta_w).add_((1.0 - self.beta_w) * w_hat)
            self._w_ema.clamp_(min=self.eps)
            self._w_ema.mul_(len(LOSS_KEYS) / (self._w_ema.sum() + self.eps))

        return self._add_obs((self._w_ema.detach() * vals).sum(), loss_dict)


# ── Homoscedastic uncertainty ─────────────────────────────────────────────────

class HomoscedasticAggregator(PINNAggregator):
    """
    Homoscedastic uncertainty weighting (Kendall & Gal 2018,
    arXiv:1705.07115).

    Learns a log-variance ``log_sigma_k`` for each loss term::

        total = sum_k  0.5 * exp(-2*log_sigma_k) * L_k + log_sigma_k

    The ``log_sigma`` parameters must be added to the main optimiser —
    use ``aggregator.parameters()`` when constructing the optimiser.
    """

    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(
            torch.zeros(len(LOSS_KEYS), dtype=torch.float32)
        )

    @property
    def current_weights(self) -> Dict[str, float]:
        w = 0.5 * torch.exp(-2.0 * self.log_sigma.detach())
        return {k: float(w[i]) for i, k in enumerate(LOSS_KEYS)}

    def aggregate(self, loss_dict, step):
        vals  = [loss_dict[k] for k in LOSS_KEYS]
        total = sum(
            0.5 * torch.exp(-2.0 * self.log_sigma[i]) * vals[i] + self.log_sigma[i]
            for i in range(len(LOSS_KEYS))
        )
        return self._add_obs(total, loss_dict)


# ── GradNorm ──────────────────────────────────────────────────────────────────

class GradNormAggregator(PINNAggregator):
    """
    Gradient normalisation (Chen et al. 2018, arXiv:1711.02132).

    Learns task weights whose gradient norms are equalised.
    ``alpha`` controls aggressiveness (0 = equal norms; higher = more
    penalty on fast-learning tasks).

    The ``task_weights`` are updated via a separate Adam step inside
    ``aggregate()``, orthogonal to the main network optimiser.

    .. warning::
        Requires two backward passes per step. Use ``update_freq > 1``
        to amortise the cost.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        alpha: float = 0.1,
        update_freq: int = 1,
    ):
        super().__init__()
        self.params = params
        self.alpha = alpha
        self.update_freq = update_freq
        n = len(LOSS_KEYS)
        self.task_weights = nn.Parameter(
            torch.ones(n, dtype=torch.float32), requires_grad=True
        )
        self._w_opt = torch.optim.Adam([self.task_weights], lr=1e-2)
        self._initial_losses: Optional[torch.Tensor] = None

    @property
    def current_weights(self) -> Dict[str, float]:
        w = torch.softmax(self.task_weights.detach(), dim=0) * len(LOSS_KEYS)
        return {k: float(w[i]) for i, k in enumerate(LOSS_KEYS)}

    def aggregate(self, loss_dict, step):
        vals = [loss_dict[k] for k in LOSS_KEYS]
        L    = torch.stack([v.detach().float() for v in vals])

        if self._initial_losses is None:
            self._initial_losses = L.clone() + 1e-30

        w = torch.softmax(self.task_weights, dim=0) * len(LOSS_KEYS)
        weighted = sum(w[i] * vals[i] for i in range(len(LOSS_KEYS)))

        if self.training and step % self.update_freq == 0:
            all_p = [p for p in self.params if p.requires_grad]
            G = []
            for i, v in enumerate(vals):
                g = torch.autograd.grad(
                    w[i] * v, all_p,
                    retain_graph=True, allow_unused=True, create_graph=True,
                )
                g_flat = torch.cat([
                    gi.reshape(-1) if gi is not None
                    else v.new_zeros(p.numel())
                    for gi, p in zip(g, all_p)
                ])
                G.append(g_flat.norm())
            G = torch.stack(G)
            G_bar  = G.mean().detach()
            r      = L.detach() / self._initial_losses
            target = (G_bar * (r / r.mean().clamp(min=1e-30)) ** self.alpha).detach()
            gn_loss = (G - target).abs().sum()
            self._w_opt.zero_grad()
            gn_loss.backward(inputs=[self.task_weights], retain_graph=True)
            self._w_opt.step()

        return self._add_obs(weighted, loss_dict)


# ── LR Annealing ──────────────────────────────────────────────────────────────

class LRAnnealingAggregator(PINNAggregator):
    """
    Learning-rate annealing (Wang et al. 2020, arXiv:2001.04536).

    Scales λ_k by the ratio of the PDE gradient mean to each term's gradient
    mean::

        λ_k ← EMA( mean|grad L_interior| / mean|grad L_k| )

    ``update_freq > 1`` amortises gradient computation costs.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        ema_alpha: float = 0.9,
        update_freq: int = 1,
    ):
        super().__init__()
        self.params = params
        self.ema_alpha = ema_alpha
        self.update_freq = update_freq
        self._lambdas: Dict[str, float] = {k: 1.0 for k in LOSS_KEYS}

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._lambdas)

    def _mean_abs_grad(self, loss: torch.Tensor) -> float:
        if loss.grad_fn is None:
            return 0.0
        all_p = [p for p in self.params if p.requires_grad]
        if not all_p:
            return 0.0
        grads = torch.autograd.grad(
            loss, all_p, retain_graph=True, allow_unused=True
        )
        total = sum(g.abs().mean().item() for g in grads if g is not None)
        count = sum(1 for g in grads if g is not None)
        return total / max(count, 1)

    def aggregate(self, loss_dict, step):
        vals = {k: loss_dict[k] for k in LOSS_KEYS}

        if self.training and step % self.update_freq == 0:
            ref = self._mean_abs_grad(vals['interior']) + 1e-30
            new_l = {}
            for k, v in vals.items():
                g = self._mean_abs_grad(v)
                new_l[k] = ref / (g + 1e-30) if g > 0.0 else self._lambdas[k]
            a = self.ema_alpha
            self._lambdas = {
                k: a * self._lambdas[k] + (1 - a) * new_l[k]
                for k in LOSS_KEYS
            }

        return self._add_obs(sum(self._lambdas[k] * vals[k] for k in LOSS_KEYS if k in loss_dict), loss_dict)


# ── NTK-L2 (gradient-norm of √loss, no Jacobian matrix) ─────────────────────

class NTKL2Aggregator(PINNAggregator):
    """
    NTK-L2 loss balancing (Wang, Teng & Perdikaris 2021, simpler variant).

    Weights each loss component by ``ntk_sum / ntk_i`` where::

        ntk_i = || ∂ sqrt(|L_i| + ε) / ∂θ ||₂

    No Jacobian matrix is built; no residual sampling or batch-size tuning
    is needed.  Gradient norms are computed directly from the scalar per-
    component losses returned by ``compute_total_loss``.

    Operates on the same granular keys as ``NTKAggregator`` by default
    (``NTK_PDE_KEYS + NTK_COARSE_KEYS``) so results are directly comparable.

    Parameters
    ----------
    networks : NetworkManager
        Provides ``get_all_parameters()`` for gradient computation.
    update_freq : int
        How often (in steps) to recompute NTK gradient norms (default 1000).
    start_step : int
        Step at which rebalancing begins; before this, uniform sum.
    keys : list[str] or None
        Loss keys to balance.  Defaults to
        ``NTK_PDE_KEYS + NTK_COARSE_KEYS``.
    save_name : str or None
        If set, NTK norms are appended to ``<save_name>.csv`` at each
        recomputation.  Requires ``pandas``.
    """

    def __init__(
        self,
        networks,
        update_freq: int = 1000,
        start_step: int = 0,
        keys: Optional[List[str]] = None,
        save_name: Optional[str] = None,
    ):
        super().__init__()
        self._networks = networks
        self.update_freq = update_freq
        self.start_step = start_step
        self._keys: List[str] = keys if keys is not None else (NTK_PDE_KEYS + NTK_COARSE_KEYS)
        self.save_name = save_name
        self._cache: Dict[str, float] = {}
        self._if_csv_head = True
        if save_name:
            import warnings
            warnings.warn(
                "NTKL2Aggregator: CUDA graphs are incompatible with save_name.",
                stacklevel=2,
            )

    def _compute_ntk_l2(
        self, loss_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute ||∂√(|L_i|+ε)/∂θ||₂ for each active key."""
        trainable = [p for p in self._networks.get_all_parameters() if p.requires_grad]
        norms: Dict[str, float] = {}
        for key in self._keys:
            if key not in loss_dict:
                continue
            loss_val = loss_dict[key]
            if loss_val.grad_fn is None:
                # Constant loss (e.g. disabled physics term) — assign near-zero norm
                # so it gets a large weight but doesn't crash autograd.
                norms[key] = float(1e-30)
                continue
            grads = torch.autograd.grad(
                torch.sqrt(torch.abs(loss_val) + 1e-30),
                trainable,
                retain_graph=True,
                allow_unused=True,
            )
            g2 = sum(g.detach().pow(2).sum() for g in grads if g is not None)
            norms[key] = float(torch.sqrt(g2 + 1e-30))
        return norms

    def _save_csv(self, step: int) -> None:
        try:
            import pandas as pd
        except ImportError:
            print("  NTKL2Aggregator: pandas not available — skipping CSV export.")
            return
        df = pd.DataFrame(
            {k: v for k, v in self._cache.items()}, index=[step]
        )
        df.to_csv(self.save_name + ".csv", mode="a", header=self._if_csv_head)
        self._if_csv_head = False

    @property
    def current_weights(self) -> Optional[Dict[str, float]]:
        if not self._cache:
            return None
        ntk_sum = sum(self._cache.values())
        return {k: ntk_sum / v for k, v in self._cache.items()}

    def aggregate(self, loss_dict: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        if step >= self.start_step and step % self.update_freq == 0 and step > 0:
            self._cache = self._compute_ntk_l2(loss_dict)
            if self.save_name:
                self._save_csv(step)

        # Before first update: plain sum over active keys
        if not self._cache:
            return self._add_obs(sum(loss_dict[k] for k in self._keys if k in loss_dict), loss_dict)

        ntk_sum = sum(self._cache.values())
        return self._add_obs(sum(
            (ntk_sum / self._cache.get(k, ntk_sum)) * loss_dict[k]
            for k in self._keys
            if k in loss_dict
        ), loss_dict)


# ── PINNACLE NTK (Jacobian-based, granular) ───────────────────────────────────

class NTKAggregator(PINNAggregator):
    """
    NTK-based adaptive loss balancing using the existing PINNACLE
    Jacobian-trace implementation (Wang, Teng & Perdikaris 2021).

    This aggregator wraps the existing ``NTKWeightManager`` and exposes the
    standard ``aggregate(loss_dict, step)`` interface.

    **How it works:**

    1. At every ``update_freq`` step, ``ntk_manager.compute_weights()`` is
       called.  This draws a mini-batch, computes the per-residual Jacobian
       matrix via ``compute_jacobian``, and returns 7 granular weights::

           {'cv_pde', 'av_pde', 'h_pde', 'poisson_pde',
            'boundary', 'initial', 'film_physics'}

    2. ``aggregate(loss_dict, step)`` applies these weights directly to
       the granular per-component losses returned by
       ``compute_total_loss(..., ntk_weights=None)``.  The four coarse
       boundary/initial/film losses are each weighted by a single scalar;
       the four PDE losses are individually reweighted::

           total = w_cv * L_cv  + w_av * L_av  + w_h * L_h
                 + w_pss * L_pss
                 + w_bc * L_bc  + w_ic * L_ic  + w_film * L_film

    The ``loss_dict`` argument must therefore contain the full granular keys
    ``'weighted_cv_pde'``, ``'weighted_av_pde'``, etc. (these are the batch-
    normalised component losses when ``compute_total_loss`` is called with
    ``ntk_weights=None``).

    Parameters
    ----------
    ntk_manager : NTKWeightManager
        Initialised PINNACLE NTKWeightManager.
    update_freq : int
        How often (in steps) to recompute NTK traces.  Defaults to the
        value set in ``config.training.ntk_update_freq`` (typically 100).
    start_step : int
        Step at which to begin NTK rebalancing (before this, all weights = 1).
    """

    def __init__(
        self,
        ntk_manager,                      # NTKWeightManager instance
        update_freq: int = 100,
        start_step:  int = 0,
    ):
        super().__init__()
        self._ntk = ntk_manager
        self.update_freq = update_freq
        self.start_step  = start_step
        self._cached: Optional[Dict[str, float]] = None

    @property
    def current_weights(self) -> Optional[Dict[str, float]]:
        if self._cached is None:
            return None
        # Map internal keys back to display keys
        return {NTK_WEIGHT_MAP.get(k, k): v
                for k, v in self._cached.items()}

    def aggregate(self, loss_dict: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        # Update NTK traces if needed
        if step >= self.start_step and step % self.update_freq == 0:
            self._cached = self._ntk.compute_weights()
            # Store for density plots in analysis
            if hasattr(self._ntk, 'ntk_weight_distributions') and self._cached:
                d = self._ntk.ntk_weight_distributions
                for short_key in d.keys():
                    full_key = short_key  # same in _cached
                    if full_key in self._cached:
                        v = self._cached[full_key]
                        d[short_key].append(
                            v.item() if isinstance(v, torch.Tensor) else float(v)
                        )

        if self._cached is None:
            # Before first update: uniform sum
            return sum(loss_dict[k] for k in LOSS_KEYS)

        # Apply granular NTK weights to per-component loss values
        total = sum(
            float(self._cached.get(NTK_WEIGHT_MAP[k], 1.0)) * loss_dict[k]
            for k in NTK_PDE_KEYS
        )
        total = total + sum(
            float(self._cached.get(k, 1.0)) * loss_dict[k]
            for k in NTK_COARSE_KEYS
        )
        return total


# ── Registry & factory ────────────────────────────────────────────────────────

AGGREGATOR_NAMES = [
    'sum', 'batch_size', 'ema', 'soft_adapt', 'relobralo',
    'brdr', 'grad_norm', 'lr_annealing', 'homoscedastic', 'ntk', 'ntk_l2',
]


def build_aggregator(
    name: str,
    config: Any,
    networks,                  # NetworkManager
    physics,                   # ElectrochemicalPhysics
    sampler,                   # CollocationSampler
    device: torch.device,
    **kwargs,
) -> PINNAggregator:
    """
    Factory function for PINNACLE loss aggregators.

    Parameters
    ----------
    name : str
        Aggregator strategy name.  One of: ``'sum'``, ``'batch_size'``,
        ``'ema'``, ``'soft_adapt'``, ``'relobralo'``, ``'brdr'``,
        ``'grad_norm'``, ``'lr_annealing'``, ``'homoscedastic'``, ``'ntk'``.
    config : DictConfig
        Hydra config object (for reading ``training.*``, ``batch_size.*``).
    networks : NetworkManager
        Trained PINNACLE networks; needed for gradient-based aggregators.
    physics : ElectrochemicalPhysics
        Physics module; passed to NTKWeightManager.
    sampler : CollocationSampler
        Collocation sampler; passed to NTKWeightManager.
    device : torch.device
        Computation device.
    **kwargs :
        Extra hyper-parameters forwarded to the aggregator constructor
        (e.g. ``ema_alpha=0.95``, ``beta=0.2``, ``update_freq=50``).

    Returns
    -------
    PINNAggregator
        An ``nn.Module``; call ``aggregator.aggregate(loss_dict, step)``
        inside the training loop to obtain the scalar total loss.

    Examples
    --------
    >>> agg = build_aggregator('ema', cfg, networks, physics, sampler, device)
    >>> total = agg.aggregate(loss_dict, step)
    >>> total.backward()
    """
    key = name.lower().strip()

    if key == 'sum':
        return SumAggregator(**kwargs)

    if key == 'batch_size':
        batch_cfg = config.get('batch_size', {})
        sizes = {
            'interior': int(batch_cfg.get('interior', 1)),
            'BC':        int(batch_cfg.get('BC', 1)),
            'IC':        int(batch_cfg.get('IC', 1)),
            'L':         int(batch_cfg.get('L', 1)),
        }
        return BatchSizeAggregator(sizes)

    if key == 'ema':
        return EMAAggregator(
            ema_alpha=float(kwargs.get('ema_alpha', 0.99)),
            clamp_lo=float(kwargs.get('clamp_lo', 1e-3)),
            clamp_hi=float(kwargs.get('clamp_hi', 1e3)),
        )

    if key == 'soft_adapt':
        return SoftAdaptAggregator(
            beta=float(kwargs.get('beta', 0.1)),
            update_freq=int(kwargs.get('update_freq', 1)),
        )

    if key == 'relobralo':
        return RelobRaloAggregator(
            rho=float(kwargs.get('rho', 0.999)),
        )

    if key == 'brdr':
        return BRDRAggregator(
            beta_c=float(kwargs.get('beta_c', 0.999)),
            beta_w=float(kwargs.get('beta_w', 0.999)),
        ).to(device)

    if key == 'grad_norm':
        params = list(networks.get_all_parameters())
        return GradNormAggregator(
            params=params,
            alpha=float(kwargs.get('alpha', 0.1)),
            update_freq=int(kwargs.get('update_freq', 1)),
        )

    if key == 'lr_annealing':
        params = list(networks.get_all_parameters())
        return LRAnnealingAggregator(
            params=params,
            ema_alpha=float(kwargs.get('ema_alpha', 0.9)),
            update_freq=int(kwargs.get('update_freq', 1)),
        )

    if key == 'homoscedastic':
        return HomoscedasticAggregator()

    if key in ('ntk', 'hybrid_ntk', 'hybrid_ntk_batch'):
        from weighting.weighting import NTKWeightManager, NTKConfig
        ntk_cfg = NTKConfig()
        ntk_mgr = NTKWeightManager(networks, physics, sampler, config)
        update_freq = int(
            config.training.get('ntk_update_freq',
            kwargs.get('update_freq', 100))
        )
        start_step = int(
            config.training.get('ntk_start_step',
            kwargs.get('start_step', 0))
        )
        return NTKAggregator(ntk_mgr, update_freq=update_freq, start_step=start_step)

    if key == 'ntk_l2':
        update_freq = int(
            config.training.get('ntk_update_freq',
            kwargs.get('update_freq', 1000))
        )
        start_step = int(
            config.training.get('ntk_start_step',
            kwargs.get('start_step', 0))
        )
        save_name = kwargs.get('save_name', None)
        return NTKL2Aggregator(
            networks,
            update_freq=update_freq,
            start_step=start_step,
            save_name=save_name,
        )

    raise ValueError(
        f"Unknown aggregator '{name}'.  Choose from: {AGGREGATOR_NAMES}"
    )
