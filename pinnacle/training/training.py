# training/training.py
"""
Training orchestration for PINNACLE with integrated NTK weighting.

Class-based approach for managing the complex state and lifecycle
of physics-informed neural network training with automatic loss balancing.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import os
import json
import time
import numpy as np

from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics
from sampling.sampling import CollocationSampler
from sampling.sampling import AdaptiveCollocationSampler
from losses.losses import compute_total_loss
from losses.losses import compute_total_loss_al
from losses.losses import compute_interior_loss,compute_boundary_loss,compute_initial_loss,compute_film_physics_loss
from weighting.weighting import (
    NTKWeightManager,
    setup_ntk_weighting,
    create_loss_weights
)
from weighting.weighting import ALConfig
from weighting.weighting import ALWeightManager
from losses.losses import _extract_constraint_violations_al
from losses.aggregator import (
    PINNAggregator,
    HomoscedasticAggregator,
    UNIFORM_WEIGHTS,
    build_aggregator,
    AGGREGATOR_NAMES,
)
torch.manual_seed(995) 

class PINNTrainer:
    """
    Physics-Informed Neural Network trainer for electrochemical systems with NTK weighting.

    **Training Process:**

    1. **Initialization Phase:**
       - Initialize networks, physics, and sampling
       - Setup optimizer and learning rate scheduler
       - Configure loss weighting strategy (uniform, batch_size, manual, or NTK)

    2. **Training Phase:**
       - Sample collocation points (interior, boundary, initial, film physics)
       - Compute physics-informed losses using governing equations
       - Update NTK weights periodically (if using NTK strategy)
       - Perform gradient descent optimization
       - Track progress and save checkpoints

    3. **Monitoring Phase:**
       - Log detailed loss breakdowns
       - Save best models automatically
       - Generate training curves and diagnostics

    This class manages all the complex state needed for PINN training while
    providing a clean, simple interface for research use with automatic loss balancing.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, output_dir: Optional[str] = None):
        """
        Initialize the PINN trainer with configurable loss weighting.

        Args:
            config: Complete configuration dictionary
            device: PyTorch device for computation
            output_dir: Directory for saving outputs and checkpoints
        """
        self.config = config
        self.device = device
        self.output_dir = output_dir or "outputs"
        self.weighting_strategy = self.config.training.weight_strat 
        self.al_config = ""
        self.use_al = False
        self.constraint_history = {}
        # Initialize core components
        print(" Initializing PINNACLE components...")
        self.networks = NetworkManager(config, device)
        self.physics = ElectrochemicalPhysics(config, device)

        if config.sampling.strat == "Adaptive":
            self.sampler = AdaptiveCollocationSampler(config, self.physics, device)
            self.use_adaptive = True
        else:
            self.sampler = CollocationSampler(config, self.physics, device)
            self.use_adaptive = False
        # Setup loss weighting strategy
        self._setup_loss_weighting()
        # Setup optimization
        if self.weighting_strategy == "AL":
            # Create optimizer with parameter groups (like the paper)
            network_params = self.networks.get_all_parameters()
            multiplier_params = self.al_manager.get_multiplier_parameters()
            optimizer_config = self.config['optimizer']['adam']
            self.optimizer = optim.Adam([
                {'params': network_params},  # θ with default lr
                {'params': multiplier_params, 'lr': self.al_config.lr_lambda}  # λ with different lr
            ], lr=optimizer_config["lr"],betas=optimizer_config['betas'],eps=optimizer_config['eps'])
            for param_group in self.optimizer.param_groups:
                if 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']
        else:
            # Standard optimizer
            self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_step = 0
        self.best_loss = float('inf')
        self.best_checkpoint_path = None

        # Setup directories
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Loss history tracking
        self.loss_history = {
            'total': [], 'interior': [], 'boundary': [], 'initial': [], 'film_physics': [],
            'weighted_cv_pde': [], 'weighted_av_pde': [], 'weighted_h_pde': [], 'weighted_poisson_pde': [],
            'weighted_cv_ic': [], 'weighted_av_ic': [], 'weighted_poisson_ic': [], 'weighted_h_ic': [],
            'weighted_L_ic': [],
            'weighted_cv_mf_bc': [], 'weighted_av_mf_bc': [], 'weighted_u_mf_bc': [],
            'weighted_cv_fs_bc': [], 'weighted_av_fs_bc': [], 'weighted_u_fs_bc': [], 'weighted_h_fs_bc': [],
            # Per-BC RMS residuals (paper revision E2). Empty until plumbed through
            # compute_total_loss / compute_total_loss_al; missing-step entries are NaN.
            'bc_cv_mf_rms': [], 'bc_av_mf_rms': [], 'bc_u_mf_rms': [],
            'bc_cv_fs_rms': [], 'bc_av_fs_rms': [], 'bc_u_fs_rms': [], 'bc_h_fs_rms': [],
            # Inverse-problem trajectories (paper revision E6). Populated only
            # when inverse.enabled; NaN-filled otherwise.
            'obs_loss': [], 'k2_0': [], 'k5_0': [], 'D_cv': [],
        }

        if self.use_al:
            self.total_multiplier_l2_history = []  # Track combined ||λ||₂ over time
        # Training configuration
        self.max_steps = config['training']['max_steps']
        self.print_freq = config['training']['rec_results_freq']
        self.save_freq = config['training']['save_network_freq']
        self.current_weighting_mode = ""
        self.ntk_steps = self.config.training.ntk_steps


        # Training statistics
        self.start_time = None
        self.total_params = sum(p.numel() for p in self.networks.get_all_parameters() if p.requires_grad)

        # Inverse-problem mode: cache observation set once so the inverse
        # experiment is deterministic across the run (E6).
        self.inverse_enabled = bool(getattr(self.physics, 'inverse_enabled', False))
        self.inverse_obs = None
        if self.inverse_enabled:
            self.inverse_obs = self.sampler.sample_inverse_observations()
            if self.inverse_obs is None:
                raise RuntimeError("inverse.enabled=true but FEM data unavailable; cannot sample observations.")
            # Cast cached observations to the model's dtype so they are
            # compatible with float64 network weights (E6 uses float64).
            _model_dtype = next(iter(self.networks.get_all_parameters())).dtype
            self.inverse_obs = {k: v.to(dtype=_model_dtype, device=self.device)
                                for k, v in self.inverse_obs.items()}
            n = int(self.inverse_obs['t'].shape[0])
            print(f" Inverse observations cached: N={n} points at voltages "
                  f"{sorted(set(self.inverse_obs['E'].view(-1).tolist()))}")

        print(f" Initialization complete!")
        print(f" Total parameters: {self.total_params:,}")
        print(f"  Loss weighting strategy: {self.weighting_strategy}")

    # Strategies handled by the legacy NTK/AL path.
    _LEGACY_STRATEGIES = {'ntk', 'hybrid_ntk', 'hybrid_ntk_batch', 'AL', 'None', 'batch_size', 'uniform'}

    def _setup_loss_weighting(self):
        """Setup the loss weighting strategy based on configuration."""
        self.aggregator: Optional[PINNAggregator] = None

        if self.weighting_strategy in self._LEGACY_STRATEGIES and self.weighting_strategy == 'ntk':
            # Setup NTK weight manager
            print(" Setting up NTK-based automatic loss weighting...")
            self.ntk_manager = setup_ntk_weighting(
                self.networks,
                self.physics,
                self.sampler,
                self.config
            )
            # Start with uniform weights until first NTK update
            self.loss_weights = create_loss_weights(self.config)
            self.ntk_weights = None  # Will be set when NTK weights are computed
        elif self.weighting_strategy == 'hybrid_ntk_batch':
            self.ntk_steps = self.config['training']['ntk_steps']
            self.current_weighting_mode = 'ntk'
            self.ntk_manager = setup_ntk_weighting(self.networks, self.physics, self.sampler, self.config)
            self.loss_weights = create_loss_weights(self.config)
        elif self.weighting_strategy == "AL":
            print(" Setting up Augmented Lagrangian loss weighting...")
            # Setup AL manager
            self.al_config = ALConfig(
            beta=self.config.training.get('al_beta', 100.0),
            lr_lambda=self.config.training.get('al_lr_lambda', 1e-3),
            start_step=self.config.training.get('al_start_step', 0),
            constraint_tolerance=self.config.training.get('al_tolerance', 1e-6),
            lambda_max=self.config.training.get('al_lambda_max', 100.0)
            )
            self.al_manager = ALWeightManager(self.sampler, self.al_config, self.device)
            self.al_manager._initialize_multipliers()
            self.use_al = True

            self.loss_weights = {
            'interior': 1.0,  # PDE terms remain weighted normally
            'boundary': 1.0,  # Boundary terms become constraints  
            'initial': 1.0,   # Initial terms become constraints
            'film_physics': 1.0  # Film terms become constraints
        }
            self.ntk_manager = None
            self.ntk_weights = None
        
            # AL-specific tracking
            self.al_metrics_history = {
                'penalty_term': [], 'lagrangian_term': [], 'constraint_satisfaction': []
            }
        elif self.weighting_strategy in self._LEGACY_STRATEGIES:
            # uniform / batch_size / None / other legacy static strategies
            self.loss_weights = create_loss_weights(self.config)
            self.ntk_manager = None
            self.ntk_weights = None
        else:
            # ── Unified aggregator path ───────────────────────────────────
            if self.weighting_strategy not in AGGREGATOR_NAMES:
                raise ValueError(
                    f"Unknown weight_strat '{self.weighting_strategy}'. "
                    f"Choose from legacy {sorted(self._LEGACY_STRATEGIES)} "
                    f"or aggregator {AGGREGATOR_NAMES}."
                )
            print(f"  Building '{self.weighting_strategy}' loss aggregator...")
            self.aggregator = build_aggregator(
                self.weighting_strategy,
                self.config,
                self.networks,
                self.physics,
                self.sampler,
                self.device,
            )
            self.loss_weights = UNIFORM_WEIGHTS
            self.ntk_manager = None
            self.ntk_weights = None

        print(f" Initial loss weights: {self.loss_weights}")


    def save_ntk_weights(self, updated_weights):
        """Store updated weights in list for plotting"""
        def _f(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v)
        for key in ("cv_pde", "av_pde", "h_pde", "poisson_pde", "boundary", "initial", "film_physics"):
            self.ntk_manager.ntk_weight_distributions[key].append(_f(updated_weights[key]))

        return
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create and configure the optimizer.

        In inverse-problem mode, learnable physics parameters (k3_0, D_cv)
        are placed in a separate param group with weight_decay=0 and a
        (typically smaller) learning rate. AdamW's decoupled L2 would
        otherwise pull these parameters toward zero, biasing the recovery.
        """
        optimizer_config = self.config['optimizer']['adam']
        network_params = list(self.networks.get_all_parameters())
        # HomoscedasticAggregator has learnable log_sigma that must be optimised.
        if isinstance(getattr(self, 'aggregator', None), HomoscedasticAggregator):
            network_params = network_params + list(self.aggregator.parameters())

        param_groups = [{
            'params': network_params,
            'weight_decay': optimizer_config['weight_decay'],
        }]

        # Append a separate group for the inverse params if enabled.
        inverse_cfg = self.config.get('inverse', {}) if hasattr(self.config, 'get') else {}
        if inverse_cfg.get('enabled', False) and getattr(self.physics, 'inverse_params', None) is not None:
            inv_params = list(self.physics.inverse_params.parameters())
            param_groups.append({
                'params': inv_params,
                'weight_decay': 0.0,
                'lr': float(inverse_cfg.get('lr_params', optimizer_config['lr'] * 0.1)),
            })
            print(f" Inverse mode: {len(inv_params)} learnable physics parameter(s) "
                  f"with weight_decay=0, lr={param_groups[-1]['lr']:.2e}")

        optimizer = optim.AdamW(
            param_groups,
            lr=optimizer_config['lr'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
        )

        # Set initial_lr for scheduler compatibility
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create and configure the learning rate scheduler."""
        scheduler_config = self.config['scheduler']

        if scheduler_config['type'] == "RLROP":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_config['RLROP']['factor'],
                patience=scheduler_config['RLROP']['patience'],
                threshold=scheduler_config['RLROP']['threshold'],
                min_lr=scheduler_config['RLROP']['min_lr'],
            )
        elif scheduler_config['type'] == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config['tf_exponential_lr']['decay_rate'],
                last_epoch=scheduler_config['tf_exponential_lr']['decay_steps']
            )
        elif scheduler_config['type'] == "None":
            return optim.lr_scheduler.ConstantLR(
                self.optimizer,1.0,self.config.training.max_steps,self.config.training.max_steps
            )

    def sample_training_points(self) -> Tuple[torch.Tensor, ...]:
        """
        Sample all collocation points needed for one training step.

        Returns:
            Tuple of all sampled points for loss computation
        """
        _dtype = next(iter(self.networks.get_all_parameters())).dtype

        def _cast(t):
            return t.to(dtype=_dtype) if t is not None else None

        # Sample different types of points

        if self.config.sampling.strat == "Uniform":
            x_interior, t_interior, E_interior = self.sampler.sample_interior_points(self.networks)
            x_boundary, t_boundary, E_boundary = self.sampler.sample_boundary_points(self.networks)
            x_initial, t_initial, E_initial = self.sampler.sample_initial_points(self.networks)
            t_film, E_film = self.sampler.sample_film_physics_points()
            # NB: do NOT pass fem_batch_size here — that silently overrides
            # hybrid.n_data_points (E4 data-efficiency sweep). Let sample_fem_data
            # resolve the anchor count from n_data_points (0 => pure physics).
            fem_data = self.sampler.sample_fem_data() if self.config.hybrid.use_data else None

            return (_cast(x_interior), _cast(t_interior), _cast(E_interior),
                _cast(x_boundary), _cast(t_boundary), _cast(E_boundary),
                _cast(x_initial), _cast(t_initial), _cast(E_initial),
                _cast(t_film), _cast(E_film),
                {k: _cast(v) for k, v in fem_data.items()} if fem_data is not None else None)

        elif self.config.sampling.strat == "Adaptive":
            adaptive_interior = self.sampler.get_interior_points()
            adaptive_boundary = self.sampler.get_boundary_points()
            adaptive_initial = self.sampler.get_initial_points()
            adaptive_film = self.sampler.get_film_points()

            return (_cast(adaptive_interior[0]), _cast(adaptive_interior[1]), _cast(adaptive_interior[2]),
                    _cast(adaptive_boundary[0]), _cast(adaptive_boundary[1]), _cast(adaptive_boundary[2]),
                    _cast(adaptive_initial[0]), _cast(adaptive_initial[1]), _cast(adaptive_initial[2]),
                    _cast(adaptive_film[0]), _cast(adaptive_film[1]))

    def _update_loss_weights(self):
        """Update loss weights using the configured strategy."""
        if self.weighting_strategy == 'ntk' or (self.weighting_strategy == "hybrid_ntk_batch" and self.current_step < self.ntk_steps) and self.ntk_manager is not None:
            # Update NTK weights if needed
            updated_weights = self.ntk_manager.update_weights(self.current_step)
            if updated_weights is not None:
                # Store NTK weights for passing to compute_total_loss
                self.ntk_weights = updated_weights
                self.save_ntk_weights(updated_weights)
        elif self.weighting_strategy == "hybrid_ntk_batch" and self.current_step>= self.ntk_steps:
            if self.current_step == self.ntk_steps:
                print(f" SWITCHING: NTK → Batch Size Weighting at step {self.current_step}")
                self.current_weighting_mode = 'batch_size'
                self.ntk_weights = None  # Clear NTK weights
                # Set batch size weights
                batch_config = self.config.get('batch_size', {})
                self.loss_weights = {
                    'interior': 1.0 / batch_config.get('interior', 1),
                    'boundary': 1.0 / batch_config.get('BC', 1), 
                    'initial': 1.0 / batch_config.get('IC', 1),
                    'film_physics': 1.0 / batch_config.get('L', 1)
                }
                print(f" New batch size weights: {self.loss_weights}")
                return 
        else:
            # Clear NTK weights if not using NTK strategy
            self.ntk_weights = None

    def compute_losses(self) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses for current step with dynamic weighting.

        Returns:
            Dictionary of all loss components
        """
        # Update weights if using dynamic strategy
        self._update_loss_weights()

        # Sample training points
        (x_interior, t_interior, E_interior,
         x_boundary, t_boundary, E_boundary,
         x_initial, t_initial, E_initial,
         t_film, E_film,fem_data) = self.sample_training_points()
        

        if self.use_al:
            loss_dict, _, constraint_violations = compute_total_loss_al(
            x_interior, t_interior, E_interior,
            x_boundary, t_boundary, E_boundary,
            x_initial, t_initial, E_initial,
            t_film, E_film,
            self.networks, self.physics,
            self.al_manager
        )
            # Store AL metrics for plotting
            if 'penalty' in loss_dict:
                self.al_metrics_history['penalty_term'].append(loss_dict['penalty'].item())
            if 'lagrangian' in loss_dict:
                self.al_metrics_history['lagrangian_term'].append(loss_dict['lagrangian'].item())

            for name, violation in constraint_violations.items():
                if name not in self.constraint_history:
                    self.constraint_history[name] = []
                self.constraint_history[name].append(violation.item())

        # Compute all losses with current weights
        elif self.ntk_weights is not None:
            # Use NTK weights (granular component weighting)
            loss_dict = compute_total_loss(
                x_interior, t_interior, E_interior,
                x_boundary, t_boundary, E_boundary,
                x_initial, t_initial, E_initial,
                t_film, E_film,fem_data,
                self.networks, self.physics,
                weights=None,  # Don't use standard weights
                ntk_weights=self.ntk_weights,Hybrid=(self.config.hybrid.use_data and fem_data is not None)  # data term only when anchors exist (N>0)
            )
        else:
            # Use standard weights (uniform, batch_size, manual)
            loss_dict = compute_total_loss(
                x_interior, t_interior, E_interior,
                x_boundary, t_boundary, E_boundary,
                x_initial, t_initial, E_initial,
                t_film, E_film, fem_data,
                self.networks, self.physics,
                weights=self.loss_weights,
                ntk_weights=None,Hybrid=(self.config.hybrid.use_data and fem_data is not None)  # data term only when anchors exist (N>0)
            )

        # Inverse-problem observation loss (paper revision E6).
        # MSE between the film-thickness network prediction at the observation
        # locations and the FEM-derived ground-truth L. Gradient flows through
        # the network weights AND through k3_0/D_cv, which are the unknowns.
        if self.inverse_enabled and self.inverse_obs is not None:
            t_obs = self.inverse_obs['t']
            E_obs = self.inverse_obs['E']
            L_obs = self.inverse_obs['L']
            # Predict in dimensionless coordinates; the L network expects (t, E).
            L_pred = self.networks['film_thickness'](torch.cat([t_obs, E_obs], dim=1))
            obs_loss = torch.mean((L_pred - L_obs) ** 2)
            loss_dict['obs_loss'] = obs_loss
            loss_dict['total'] = loss_dict['total'] + obs_loss

        # Unified aggregator path: replace total with aggregator-weighted sum.
        if self.aggregator is not None:
            loss_dict['total'] = self.aggregator.aggregate(loss_dict, self.current_step)

        return loss_dict
    
    def _collect_current_points_and_residuals(self, networks):
        """Collect current collocation points and compute their residuals"""
        
        # Sample current points (works for both uniform and adaptive)
        if self.config.sampling.strat == "Uniform":
            x_interior, t_interior, E_interior = self.sampler.sample_interior_points(networks)
            x_boundary, t_boundary, E_boundary = self.sampler.sample_boundary_points(networks)
            x_initial, t_initial, E_initial = self.sampler.sample_initial_points(networks)
            t_film, E_film = self.sampler.sample_film_physics_points()
        else:  # Adaptive
            x_interior, t_interior, E_interior = self.sampler.get_interior_points()
            x_boundary, t_boundary, E_boundary = self.sampler.get_boundary_points()
            x_initial, t_initial, E_initial = self.sampler.get_initial_points()
            t_film, E_film = self.sampler.get_film_points()
        
        # Compute residuals directly using existing loss functions
        _,_, interior_residuals = compute_interior_loss(
            x_interior, t_interior, E_interior, networks, self.physics, return_residuals=True)
        
        _,_, _, boundary_residuals = compute_boundary_loss(
            x_boundary, t_boundary, E_boundary, networks, self.physics,return_residuals=True)
        
        _, _,_,initial_residuals = compute_initial_loss(
            x_initial, t_initial, E_initial, networks, self.physics, return_residuals=True)
        
        _, film_residuals = compute_film_physics_loss(
            t_film, E_film, networks, self.physics, return_residuals=True)
        

        # Package for plotting
        points_dict = {
            'interior': torch.cat([x_interior, t_interior, E_interior], dim=1),
            'boundary': torch.cat([x_boundary, t_boundary, E_boundary], dim=1),
            'initial': torch.cat([x_initial, t_initial, E_initial], dim=1),
            'film': torch.cat([t_film, E_film], dim=1)
        }
        
        residuals_dict = {
            **interior_residuals,  # cv_pde, av_pde, h_pde, poisson_pde
            'boundary': boundary_residuals,
            'initial': initial_residuals,
            'film_physics': film_residuals
        }
        
        return points_dict, residuals_dict

    def training_step(self) -> Dict[str, float]:
        """
        Perform one complete training step with automatic weight balancing.

        Returns:
            Dictionary of loss values (as floats)
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Compute losses (includes weight updates)
        loss_dict = self.compute_losses()

        # Backward pass
        total_loss = loss_dict['total']
        # No BV clamp is applied in the physics (removed — max=10 saturated the physical
        # voltage range and broke voltage-dependent kinetics). The 1e15 threshold here
        # guards Adam against non-physical init spikes without altering the physics.
        _component_spike = any(
            hasattr(loss_dict.get(k), 'item') and float(loss_dict[k].item()) > 1e15
            for k in ('film_physics', 'interior', 'boundary', 'initial')
        )
        _skip = not total_loss.isfinite() or _component_spike
        if _skip:
            self.optimizer.zero_grad()
            return {k: float(v.item()) if hasattr(v, 'item') else float(v)
                    for k, v in loss_dict.items()}
        total_loss.backward()

        # Stage 2 of inverse training: networks are frozen; only k3_0/D_cv are updated.
        # Zero out all network gradients so only the inverse param group gets a step.
        if self.inverse_enabled:
            inverse_cfg = self.config.get('inverse', {}) if hasattr(self.config, 'get') else {}
            stage2_start = int(inverse_cfg.get('stage2_start_step', float('inf')))
            if self.current_step >= stage2_start:
                if not getattr(self, '_stage2_announced', False):
                    print(f"\n[Step {self.current_step}] Inverse Stage 2: networks frozen, optimising k3_0/D_cv only")
                    self._stage2_announced = True
                for p in self.networks.get_all_parameters():
                    if p.grad is not None:
                        p.grad.zero_()

        torch.nn.utils.clip_grad_norm_(self.networks.get_all_parameters(), max_norm=0.1)

        # Clip the inverse-parameter gradients separately (log-space). The raw
        # gradient on log_k2_0 can reach O(1e6); without a guard a single
        # non-finite gradient (when the PDE solve momentarily diverges) poisons
        # the AdamW moment estimates and NaNs the recovered parameter for the
        # rest of training. Skip the inverse update entirely if its gradient is
        # not finite, so a transient network spike can't corrupt the parameter.
        if self.inverse_enabled and self.physics.inverse_params is not None:
            inv_ps = list(self.physics.inverse_params.parameters())
            inv_norm = torch.nn.utils.clip_grad_norm_(inv_ps, max_norm=1.0)
            if not torch.isfinite(inv_norm):
                for p in inv_ps:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.use_al:
            # Flip gradients for multipliers (ascent)
            torch.nn.utils.clip_grad_norm_(self.al_manager.get_multiplier_parameters(), max_norm=0.1)
            for param in self.al_manager.get_multiplier_parameters():
                if param.grad is not None:
                    param.grad *= -1

        # Optimizer step
        self.optimizer.step()

        # Update adaptive sampling periodically
        if (self.use_adaptive and 
            self.current_step % self.config.sampling.adaptive.adaptive_update_freq == 0):
            self.sampler.update_adaptive_sampling(self.current_step, self.networks)

        # Scheduler step
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss)
            else:
                self.scheduler.step()

        # Convert tensors to floats for logging
        loss_dict_float = {k: v.item() if torch.is_tensor(v) else v
                        for k, v in loss_dict.items()}

        # Snapshot inverse-problem parameters (paper revision E6).
        # Recorded every step so the recovery trajectory and ensemble
        # spread (E6-F) can be reconstructed without rerunning.
        if self.inverse_enabled and self.physics.inverse_params is not None:
            unknown = self.physics.inverse_params.unknown
            if 'k2_0' in unknown:
                loss_dict_float['k2_0'] = float(self.physics.inverse_params.k2_0.detach().cpu().item())
            if 'k5_0' in unknown:
                loss_dict_float['k5_0'] = float(self.physics.inverse_params.k5_0.detach().cpu().item())
            if 'D_cv' in unknown:
                loss_dict_float['D_cv'] = float(self.physics.inverse_params.D_cv.detach().cpu().item())

        # Update training state
        self.current_step += 1

        return loss_dict_float

    def update_loss_history(self, loss_dict: Dict[str, float]) -> None:
        """Update the loss history tracking.

        Missing keys are filled with NaN rather than 0.0 so the analysis
        layer can distinguish "not measured this step" from "perfectly
        converged". Plotting code must mask NaN before log-scale plots.
        """
        for key in self.loss_history.keys():
            if key in loss_dict:
                self.loss_history[key].append(loss_dict[key])
            else:
                self.loss_history[key].append(float('nan'))


    def print_progress(self, loss_dict: Dict[str, float]) -> None:
        """Print detailed training progress including weight information."""
        if self.current_step % self.print_freq == 0 or self.current_step==0:
            print(f"\n=== Step {self.current_step} ===")
            if self.use_al:
                # AL-specific reporting
                print(f"  Interior PDE: {loss_dict['interior']:.2e}")
                if 'penalty' in loss_dict:
                    print(f"  Penalty (β‖C‖²): {loss_dict['penalty']:.2e}")
                if 'lagrangian' in loss_dict:
                    print(f"  Lagrangian (⟨λ,C⟩): {loss_dict['lagrangian']:.2e}")
                for name, param in self.al_manager.lambda_params.items():
                    constraint_name = name.replace('lambda_', '')
                    if constraint_name in self.al_manager.lambda_distributions:
                        # Save mean absolute value or max
                        self.al_manager.lambda_distributions[constraint_name].append(
                            torch.mean(torch.abs(param)).item()
                        )
                # Show max multiplier magnitude
                if hasattr(self, 'al_manager') and self.al_manager.is_initialized:
                    stats = self.al_manager.log_multiplier_stats()
                    max_lambda = max([v for k, v in stats.items() if k.endswith('_max')])
                    print(f"  Max |λ|: {max_lambda:.2e}")
                return  
            
            if self.weighting_strategy == 'hybrid_ntk_batch':
                if self.current_step < self.ntk_steps:
                    phase = f"NTK Phase ({self.current_step}/{self.ntk_steps})"
                else:
                    batch_step = self.current_step - self.ntk_steps
                    phase = f"Batch Phase ({batch_step}/{self.max_steps-self.ntk_steps})"
                    print(f"\n=== Step {self.current_step} - {phase} ===")
            else:
                print(f"\n=== Step {self.current_step} ===")
            print(f"Total Loss: {loss_dict['total']:.6f}")
            if not self.use_al:
                print(f"Interior: {loss_dict['interior']:.6f} | Boundary: {loss_dict['boundary']:.6f} | "
                    f"Initial: {loss_dict['initial']:.6f} | Film Physics: {loss_dict['film_physics']:.6f}")

            # Show current weights
            if self.weighting_strategy == 'ntk' or self.weighting_strategy == "hybrid_ntk_batch" and self.ntk_weights is not None:
                print(f"NTK weights: CV={self.ntk_weights.get('cv_pde', 1.0):.3f}, "
                      f"AV={self.ntk_weights.get('av_pde', 1.0):.3f}, "
                      f"H={self.ntk_weights.get('h_pde', 1.0):.3f}, "
                      f"Poisson={self.ntk_weights.get('poisson_pde'):.3f}, "
                      f"BC={self.ntk_weights.get('boundary'):.3f}, "
                      f"IC={self.ntk_weights.get('initial'):.3f}, "
                      f"Film={self.ntk_weights.get('film_physics'):.10f}")
            elif self.aggregator is not None:
                w = self.aggregator.current_weights or {}
                print(f"Aggregator ({self.weighting_strategy}) weights: "
                      f"Interior={w.get('interior', 1.0):.4f}, "
                      f"Boundary={w.get('boundary', 1.0):.4f}, "
                      f"Initial={w.get('initial', 1.0):.4f}, "
                      f"Film={w.get('film_physics', 1.0):.4f}")
            elif self.weighting_strategy != 'uniform':
                print(f"Standard weights: Interior={self.loss_weights['interior']:.3f}, "
                      f"Boundary={self.loss_weights['boundary']:.3f}, "
                      f"Initial={self.loss_weights['initial']:.3f}, "
                      f"Film={self.loss_weights['film_physics']:.3f}")

            # PDE breakdown
            if 'weighted_cv_pde' in loss_dict:
                print("\nPDE Residuals:")
                print(f"  CV: {loss_dict['weighted_cv_pde']:.6f} | AV: {loss_dict['weighted_av_pde']:.6f}")
                print(f"  Hole: {loss_dict['weighted_h_pde']:.6f} | Poisson: {loss_dict['weighted_poisson_pde']:.6f}")

            if "data_loss" in loss_dict:
                print(f"\nData Loss: {loss_dict['data_loss']:.6f}")

            # Boundary breakdown
            if 'weighted_cv_mf_bc' in loss_dict:
                print("\nBoundary Conditions:")
                print(
                    f"  m/f - CV: {loss_dict['weighted_cv_mf_bc']:.6f} | AV: {loss_dict['weighted_av_mf_bc']:.6f} | U: {loss_dict['weighted_u_mf_bc']:.6f}")
                print(
                    f"  f/s - CV: {loss_dict['weighted_cv_fs_bc']:.6f} | AV: {loss_dict['weighted_av_fs_bc']:.6f} | U: {loss_dict['weighted_u_fs_bc']:.6f} | H: {loss_dict['weighted_h_fs_bc']:.6f}")

            # Initial conditions breakdown
            if 'weighted_cv_ic' in loss_dict:
                print("\nInitial Conditions:")
                print(
                    f"  CV: {loss_dict['weighted_cv_ic']:.6f} | AV: {loss_dict['weighted_av_ic']:.6f} | H: {loss_dict['weighted_h_ic']:.6f}")
                print(f"  Poisson: {loss_dict['weighted_poisson_ic']:.6f} | L: {loss_dict['weighted_L_ic']:.6f}")
            
        if self.current_step % self.config.plotting.topksteps == 0:
            points_dict, residuals_dict = self._collect_current_points_and_residuals(self.networks)
            from analysis.analysis import plot_top_k_worst
            import os
            save_path = os.path.join(os.getcwd(), f"worst_{self.config.experiment.name}", f"worst_points_step_{self.current_step:06d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plot_top_k_worst(points_dict,residuals_dict,networks=self.networks,physics=self.physics,save_path=save_path)

    def get_al_training_stats(self) -> Dict[str, Any]:
        """Get AL-specific training statistics"""
        if not self.use_al:
            return {}
        
        stats = {
            'al_config': {
                'beta': self.al_manager.config.beta,
                'lr_lambda': self.al_manager.config.lr_lambda,
                'lambda_max': self.al_manager.config.lambda_max
            },
            'multiplier_stats': self.al_manager.log_multiplier_stats(),
            'total_multipliers': sum(p.numel() for p in self.al_manager.get_multiplier_parameters()),
            'constraint_names': self.al_manager.constraint_names
        }
        
        if self.al_metrics_history['penalty_term']:
            stats['final_penalty'] = self.al_metrics_history['penalty_term'][-1]
            stats['final_lagrangian'] = self.al_metrics_history['lagrangian_term'][-1]
        
        return stats

    def save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save current training state to checkpoint.

        Args:
            checkpoint_name: Name for the checkpoint file
        """
        checkpoint = {
            'networks': self.networks.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'step': self.current_step,
            'loss': self.loss_history['total'][-1] if self.loss_history['total'] else float('inf'),
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'loss_weights': self.loss_weights,
            'weighting_strategy': self.weighting_strategy,
            'ntk_weights': self.ntk_weights  # Save current NTK weights
        }

        # Save NTK manager state if using NTK
        if self.ntk_manager is not None:
            checkpoint['ntk_current_weights'] = self.ntk_manager.get_current_weights()
            checkpoint['ntk_batch_sizes'] = self.ntk_manager.optimal_batch_sizes

        # Add hybrid data point info if using hybrid training
        if (self.config.hybrid.use_data and hasattr(self.sampler, 'last_fem_data')
                and self.sampler.last_fem_data is not None):
            checkpoint['hybrid_data_point'] = {
                't': self.sampler.last_fem_data['t'].cpu().numpy() if torch.is_tensor(self.sampler.last_fem_data['t']) else self.sampler.last_fem_data['t'],
                'E': self.sampler.last_fem_data['E'].cpu().numpy() if torch.is_tensor(self.sampler.last_fem_data['E']) else self.sampler.last_fem_data['E'],
                'L': self.sampler.last_fem_data['L'].cpu().numpy() if torch.is_tensor(self.sampler.last_fem_data['L']) else self.sampler.last_fem_data['L'],
            }

        save_path = os.path.join(self.checkpoints_dir, f"{checkpoint_name}.pt")
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.networks.load_state_dict(checkpoint['networks'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler and checkpoint.get('scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        # Restore training state so the loop resumes from where it stopped
        if 'step' in checkpoint:
            self.current_step = int(checkpoint['step'])
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        if 'ntk_current_weights' in checkpoint and self.ntk_manager is not None:
            self.ntk_manager.current_weights = checkpoint['ntk_current_weights']
        if 'ntk_batch_sizes' in checkpoint and self.ntk_manager is not None:
            self.ntk_manager.optimal_batch_sizes = checkpoint['ntk_batch_sizes']

        print(f" Resumed from checkpoint: step={self.current_step}, best_loss={self.best_loss:.6g}")


    def handle_checkpointing(self, current_loss: float) -> None:
        """Handle automatic checkpointing logic."""
        # Save best model
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_checkpoint_path = os.path.join(self.checkpoints_dir, "best_model")
            self.save_checkpoint("best_model")

        # Periodic checkpoints
        if self.current_step % self.save_freq == 0 and self.current_step > 0:
            self.save_checkpoint(f"model_step_{self.current_step}")


    def track_l2_lambda(self):
        """Compute combined L2 norm of ALL multipliers"""
        total_l2_norm = 0.0
        for param in self.al_manager.get_multiplier_parameters():
            total_l2_norm += torch.norm(param.detach()).cpu().item() ** 2

            total_l2_norm = total_l2_norm ** 0.5  # Take square root for L2 norm
            self.total_multiplier_l2_history.append(total_l2_norm)


    def train(self, manual_loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
        """
        Run the complete training process with automatic loss balancing.

        **Training Algorithm with NTK:**

        For each training step:
        1. Sample collocation points from domain
        2. Update NTK weights periodically (if using NTK strategy)
        3. Compute physics-informed loss function with current weights
        4. Perform gradient descent optimization
        5. Update learning rate schedule
        6. Track progress and save checkpoints

        Args:
            manual_loss_weights: Optional manual override for loss weights

        Returns:
            Complete loss history for analysis
        """
        # Resume from checkpoint if specified
        resume_path = self.config.get('training', {}).get('resume_checkpoint', None)
        if resume_path:
            import os as _os
            if _os.path.isfile(resume_path):
                print(f" Loading resume checkpoint: {resume_path}")
                self.load_checkpoint(resume_path)
            else:
                print(f" WARNING: resume_checkpoint path not found: {resume_path} — starting fresh")

        remaining_steps = self.max_steps - self.current_step
        if remaining_steps <= 0:
            print(f" Already completed {self.current_step}/{self.max_steps} steps — nothing to do.")
            return self.loss_history

        print(f" Starting PINNACLE training for {self.max_steps} steps...")
        if self.current_step > 0:
            print(f"  Resuming from step {self.current_step} ({remaining_steps} steps remaining)")
        print(f"  Using {self.weighting_strategy} loss weighting strategy")

        # Start timing
        self.start_time = time.time()

        # Per-step instrumentation buffers (Phase 0.1 — answers R2.2 on NTK cost).
        # Use CUDA events on GPU for accurate kernel-timing; perf_counter on CPU.
        use_cuda = self.device.type == 'cuda'
        if use_cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats(self.device)
        step_ms: List[float] = []
        peak_mem_mb: List[float] = []

        # Main training loop — start from current_step to support resume
        for step in tqdm(range(self.current_step, self.max_steps), desc="Training Progress",
                         initial=self.current_step, total=self.max_steps):
            if use_cuda:
                start_ev.record()
            else:
                t0 = time.perf_counter()

            # Perform training step (includes automatic weight updates)
            loss_dict = self.training_step()

            if use_cuda:
                end_ev.record()
                torch.cuda.synchronize(self.device)
                step_ms.append(float(start_ev.elapsed_time(end_ev)))
                peak_mem_mb.append(float(torch.cuda.max_memory_allocated(self.device) / 1024 ** 2))
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                step_ms.append((time.perf_counter() - t0) * 1000.0)
                peak_mem_mb.append(0.0)

            # Update tracking
            self.update_loss_history(loss_dict)

            if self.use_al and self.current_step % 10 == 0:  # Track every 10 steps to reduce overhead
                self.track_l2_lambda()

            # Print progress
            self.print_progress(loss_dict)

            # Handle checkpointing
            current_loss = loss_dict['total']
            self.handle_checkpointing(current_loss)

        # Persist per-step timing for the ablation table (E1 / Table II).
        self._write_timing_summary(step_ms, peak_mem_mb)

        # Training completed
        self._finish_training()

        return self.loss_history

    def _write_timing_summary(self, step_ms: List[float], peak_mem_mb: List[float]) -> None:
        """Dump per-step timing summary to timing.json for the ablation table."""
        if not step_ms:
            return
        # Drop the first step from statistics — first-step CUDA init and lazy
        # graph compilation make it a multi-second outlier on most setups.
        ms = np.asarray(step_ms[1:] if len(step_ms) > 1 else step_ms, dtype=float)
        mem = np.asarray(peak_mem_mb[1:] if len(peak_mem_mb) > 1 else peak_mem_mb, dtype=float)
        summary = {
            'strategy': self.weighting_strategy,
            'n_steps_total': len(step_ms),
            'n_steps_used_for_stats': int(ms.size),
            'mean_ms_per_step': float(np.mean(ms)),
            'median_ms_per_step': float(np.median(ms)),
            'p95_ms_per_step': float(np.percentile(ms, 95)),
            'first_step_ms': float(step_ms[0]),
            'peak_mem_mb_max': float(np.max(mem)) if mem.size else 0.0,
            'peak_mem_mb_mean': float(np.mean(mem)) if mem.size else 0.0,
            'device_type': self.device.type,
        }
        out_path = os.path.join(self.output_dir, 'timing.json')
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"⏱  Timing summary written to {out_path}")

    def _finish_training(self) -> None:
        """Complete training and print summary."""
        elapsed_time = time.time() - self.start_time
        final_loss = self.loss_history['total'][-1] if self.loss_history['total'] else float('inf')

        print(f"\n Training completed!")
        print(f" Final loss: {final_loss:.6f}")
        print(f" Best loss: {self.best_loss:.6f}")
        print(f"⏱  Training time: {elapsed_time / 60:.1f} minutes")
        print(f" Training speed: {self.max_steps / (elapsed_time / 60):.1f} steps/minute")

        if self.use_al:
            print(f" AL Method: β={self.al_manager.config.beta}")
            print(f"  Final constraint satisfaction: [show constraint metrics]")
        elif self.weighting_strategy == 'ntk' and self.ntk_weights is not None:
            print(f"  Final NTK weights: {self.ntk_weights}")
        else:
            print(f"  Final weights: {self.loss_weights}")

        # Save final checkpoint
        self.save_checkpoint("final_model")

        # Write losses.txt — tab-separated, one row per recorded step
        _LOSS_COLS = [
            'total', 'interior', 'boundary', 'initial', 'film_physics',
            'weighted_cv_pde', 'weighted_av_pde', 'weighted_h_pde', 'weighted_poisson_pde',
            'weighted_cv_ic', 'weighted_av_ic', 'weighted_poisson_ic', 'weighted_h_ic', 'weighted_L_ic',
            'weighted_cv_mf_bc', 'weighted_av_mf_bc', 'weighted_u_mf_bc',
            'weighted_cv_fs_bc', 'weighted_av_fs_bc', 'weighted_u_fs_bc', 'weighted_h_fs_bc',
        ]
        losses_path = os.path.join(self.output_dir, "losses.txt")
        n_rows = len(self.loss_history['total'])
        with open(losses_path, 'w') as _f:
            _f.write('step\t' + '\t'.join(_LOSS_COLS) + '\n')
            for _i in range(n_rows):
                _row = [str(_i + 1)]
                for _c in _LOSS_COLS:
                    _vals = self.loss_history.get(_c, [])
                    _row.append(f'{_vals[_i]:.6e}' if _i < len(_vals) else 'nan')
                _f.write('\t'.join(_row) + '\n')

        # Load best model for inference
        if self.best_checkpoint_path:
            print(f"\n Loading best checkpoint for inference...")
            self.load_checkpoint(f"{self.best_checkpoint_path}.pt")
            print(f" Using best model (loss: {self.best_loss:.6f})")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the current model.

        Returns:
            Dictionary of evaluation metrics
        """
        self.networks.eval()

        with torch.no_grad():
            loss_dict = self.compute_losses()
            loss_dict_float = {k: v.item() if torch.is_tensor(v) else v
                               for k, v in loss_dict.items()}

        self.networks.train()
        return loss_dict_float

    def set_loss_weights(self, weights: Dict[str, float]) -> None:
        """Set loss weights manually (overrides automatic weighting)."""
        self.loss_weights = weights
        print(f" Updated loss weights: {weights}")

    def get_ntk_diagnostics(self) -> Dict[str, Any]:
        """Get NTK weighting diagnostics if using NTK strategy."""
        if self.ntk_manager is None:
            return {"strategy": self.weighting_strategy, "ntk_active": False}

        return {
            "strategy": self.weighting_strategy,
            "ntk_active": True,
            "current_weights": self.ntk_manager.get_current_weights(),
            "optimal_batch_sizes": self.ntk_manager.optimal_batch_sizes,
            "last_update_step": self.ntk_manager.last_update_step,
            "update_frequency": self.ntk_manager.config.training.ntk_update_freq
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.loss_history['total']:
            return {"status": "No training completed"}

        stats = {
            "current_step": self.current_step,
            "total_steps": self.max_steps,
            "final_loss": self.loss_history['total'][-1],
            "best_loss": self.best_loss,
            "total_parameters": self.total_params,
            "training_time_minutes": (time.time() - self.start_time) / 60 if self.start_time else None,
            "loss_history_length": len(self.loss_history['total']),
            "weighting_strategy": self.weighting_strategy,
            "current_weights": self.loss_weights
        }

        # Add NTK diagnostics if available
        stats.update(self.get_ntk_diagnostics())

        # Add current NTK weights if using NTK
        if self.ntk_weights is not None:
            stats["current_ntk_weights"] = self.ntk_weights

        return stats