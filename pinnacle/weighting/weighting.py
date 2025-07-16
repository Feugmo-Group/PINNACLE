# weighting/weighting.py
"""
Neural Tangent Kernel (NTK) and loss weighting strategies for modular PINNACLE.

Clean functional approach that leverages existing loss computations from losses.py
by using the return_residuals flag to get exact residuals without duplication.

**Neural Tangent Kernel Theory:**

The NTK provides automatic loss balancing by analyzing training dynamics:

.. math::
    K(x, x') = \\nabla_\\theta f(x; \\theta) \\cdot \\nabla_\\theta f(x'; \\theta)

**Reference:**
Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating
gradient flow pathologies in physics-informed neural networks.
"""

import torch
from typing import Dict, List, Tuple, Optional, Callable, Any, NamedTuple
from dataclasses import dataclass
import torch.nn as nn
from sampling.sampling import CollocationSampler
torch.manual_seed(995) 

@dataclass
class NTKConfig:
    """Configuration for NTK-based weighting"""
    update_frequency: int = 100  # How often to update weights
    start_step: int = 0  # When to start NTK weighting
    trace_estimation_samples: int = 256  # Samples for trace estimation

def compute_jacobian(
        output: torch.Tensor,
        parameters: List[torch.nn.Parameter],
        device: torch.device
) -> torch.Tensor:
        """
    Compute Jacobian matrix using fast batched gradient computation.

    Args:
        outputs: Network outputs [batch_size] or [batch_size, 1]
        parameters: List of network parameters
        device: PyTorch device

    Returns:
        Jacobian matrix 
    """
        output = output.reshape(-1)
        grads = torch.autograd.grad(
            output,
            list(parameters),
            (torch.eye(output.shape[0]).to(device),),
            is_grads_batched=True, retain_graph=True,allow_unused=True
        )
        valid_grads = [grad.flatten().reshape(len(output), -1) 
                   for grad in grads if grad is not None]
        
        return torch.cat(valid_grads, 1)

def get_ntk(jac:torch.Tensor
            ,compute="trace") -> torch.Tensor:
    """Get the NTK matrix of jac """

    if compute == 'full':
        return torch.einsum('Na,Ma->NM', jac, jac)
    elif compute == 'diag':
        return torch.einsum('Na,Na->N', jac, jac)
    elif compute == 'trace':
        return torch.einsum('Na,Na->', jac, jac)
    else:
        raise ValueError('compute must be one of "full",'
                            + '"diag", or "trace"')

def compute_minimum_batch_size(jacobian):
    """Compute minimum batch size for 0.2 approximation error"""
    ntk_diag = get_ntk(jacobian, compute='diag')
    
    # Population statistics
    mu_X = torch.mean(ntk_diag)
    sigma_X = torch.std(ntk_diag)
    
    # Handle near-zero mean case
    if mu_X.abs() < 1e-8:
        # Use relative variation instead when mean is tiny
        if sigma_X < 1e-8:
            v_X = 1.0  # Uniform case
        else:
            # Use median as reference instead of mean
            median_X = torch.median(ntk_diag)
            v_X = sigma_X / (median_X.abs() + 1e-8)
    else:
        # Normal coefficient of variation
        v_X = sigma_X / mu_X.abs()
    
    # Clamp to reasonable bounds
    v_X = torch.clamp(v_X, min=0.1, max=5.0)
    
    min_batch_size = int(25 * (v_X ** 2))
    min_batch_size = max(min_batch_size, 32)
    min_batch_size = min(min_batch_size, len(jacobian) // 4)
    
    return min_batch_size
    
"""
class BRDRWeightManager:
    def __init__(
            self,
            networks,  # NetworkManager instance
            physics,  # ElectrochemicalPhysics instance
            sampler,  # CollocationSampler instance
            config: Optional[NTKConfig] = None
    ):
    
    #Initialize variables for storage
    #We need to compute all the residuals and store them
    #Calculate effective smoothing factors
    #Calculate the IRDR
    # Apply weight update algoritihim and return dict in form that compute_total_loss accept
    #Backpropogation happens here
    # Update scale factor    
    #"Correct the gradients??"
    #Optimizer step
""" 
class NTKWeightManager:
    """
    Neural Tangent Kernel-based automatic loss weighting for modular PINNACLE.

    This class leverages the modified loss functions in losses.py that can return
    exact residuals using the return_residuals flag, eliminating computation duplication.

    **Core Algorithm:**

    1. **Extract residuals** using existing loss functions with return_residuals=True
    2. **Compute Jacobians** for each loss component
    3. **Calculate NTK traces** to measure training difficulty
    4. **Balance weights** so all losses train at similar rates
    5. **Update periodically** during training

    **Weight Computation:**

    .. math::
        w_i = \\frac{1/\\bar{\\lambda}_i}{\\sum_j 1/\\bar{\\lambda}_j} \\cdot N

    where λ̄_i is the mean NTK eigenvalue for loss component i.
    """

    def __init__(
            self,
            networks,  # NetworkManager instance
            physics,  # ElectrochemicalPhysics instance
            sampler,  # CollocationSampler instance
            config: Optional[NTKConfig] = None
    ):
        """
        Initialize NTK weight manager for modular PINNACLE.

        Args:
            networks: NetworkManager instance
            physics: ElectrochemicalPhysics instance
            sampler: CollocationSampler instance
            config: NTK configuration
        """
        self.networks = networks
        self.physics = physics
        self.sampler = sampler
        self.config = config 
        self.device = physics.device

        # Storage for computed batch sizes (optimization)
        self.optimal_batch_sizes = {}

        # Current weights
        self.current_weights = {}

        # Update tracking
        self.last_update_step = -1

        self.ntk_weight_distributions = {
            'cv_pde': [],
            'av_pde': [], 
            'h_pde': [],
            'poisson_pde': [],
            'boundary': [],
            'initial': [],
            'film_physics': []
        }

    def compute_ntk_trace(
            self,
            loss_residuals: torch.Tensor,
            loss_name: str
    ) -> Tuple[float, int]:
        """
        Compute NTK-based weight for a single loss component.

        Args:
            loss_residuals: Residual tensor for this loss [batch_size]
            loss_name: Name of loss component

        Returns:
            Tuple of (ntk_trace, effective_batch_size)
        """
        # Determine batch size (one-time calculation)
        if loss_name not in self.optimal_batch_sizes:
            indices = torch.randperm(len(loss_residuals),device=self.device)[:256]
            residual_sampled = loss_residuals[indices]
            jacobian_sampled = compute_jacobian(residual_sampled,self.networks.get_all_parameters(),self.device)
            self.optimal_batch_sizes[loss_name] = compute_minimum_batch_size(jacobian_sampled)
            print(f"Computed batch size for {loss_name}: {self.optimal_batch_sizes[loss_name]}")

            trace = get_ntk(jacobian_sampled, compute='trace')
            
            return trace, len(jacobian_sampled) 

        #Use computed optimal batch size every other time
        else:
            # Random sampling
            batch_size = self.optimal_batch_sizes[loss_name]
            indices = torch.randperm(len(loss_residuals),device=self.device)[:batch_size]
            residual_sampled = loss_residuals[indices]
            jacobian_sampled = compute_jacobian(residual_sampled,self.networks.get_all_parameters(),self.device)

            # Compute NTK trace
            trace = get_ntk(jacobian_sampled, compute='trace')
            
            return trace, len(jacobian_sampled)
        
    def extract_all_residuals(self) -> Dict[str, torch.Tensor]:
        """
        Extract all residuals using the modified loss functions with return_residuals=True.

        This leverages the exact same computation logic as training without duplication.
        """
        # Import the modified loss functions
        from losses.losses import (
            compute_interior_loss,
            compute_boundary_loss,
            compute_initial_loss,
            compute_film_physics_loss
        )

        # Sample training points (same as training)
        x_interior, t_interior, E_interior = self.sampler.sample_interior_points(self.networks)
        x_boundary, t_boundary, E_boundary = self.sampler.sample_boundary_points(self.networks)
        x_initial, t_initial, E_initial = self.sampler.sample_initial_points(self.networks)
        t_film, E_film = self.sampler.sample_film_physics_points()

        # Extract residuals using existing loss functions
        _, _, interior_residuals = compute_interior_loss(
            x_interior, t_interior, E_interior,
            self.networks, self.physics,
            return_residuals=True
        )

        _, _, boundary_residuals = compute_boundary_loss(
            x_boundary, t_boundary, E_boundary,
            self.networks, self.physics,
            return_residuals=True
        )

        _, _, initial_residuals = compute_initial_loss(
            x_initial, t_initial, E_initial,
            self.networks, self.physics,
            return_residuals=True
        )

        _, film_residuals = compute_film_physics_loss(
            t_film, E_film,
            self.networks, self.physics,
            return_residuals=True
        )

        # Combine all residuals
        all_residuals = {
            **interior_residuals,  # cv_pde, av_pde, h_pde, poisson_pde
            'boundary': boundary_residuals,
            'initial': initial_residuals,
            'film_physics': film_residuals
        }

        return all_residuals

    def compute_weights(self) -> Dict[str, float]:
        """
        Compute NTK weights using exact residuals from existing loss computations.

        Returns:
            Dictionary of normalized weights
        """

        # Extract all residuals using modified loss functions
        all_residuals = self.extract_all_residuals()

        ntk_traces = {}
        batch_sizes = {}
        
        # Compute NTK trace for each component
        for component_name, residual in all_residuals.items():
            if len(residual) > 0:
                ntk_trace, effective_batch_size = self.compute_ntk_trace(residual, component_name)
                ntk_traces[component_name] = ntk_trace
                batch_sizes[component_name] = effective_batch_size


        # Compute normalized weights
        weights = self._normalize_ntk_weights(ntk_traces, batch_sizes)

        self.current_weights = weights

        return weights

    def _normalize_ntk_weights(
            self,
            ntk_traces: Dict[str, float],
            batch_sizes: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Normalize NTK traces to get balanced weights.

        Args:
            ntk_traces: Dictionary of NTK traces
            batch_sizes: Dictionary of batch sizes

        Returns:
            Dictionary of normalized weights
        """
        # Compute mean traces (trace per sample)
        mean_traces = {}
        for name in ntk_traces:
            mean_traces[name] = ntk_traces[name] / batch_sizes[name]

        # Compute raw weights
        raw_weights = {}
        for name, mean_trace in mean_traces.items():
            if mean_trace > 1e-12:  # Avoid division by zero
                sum_all_mean_traces = sum(mean_traces[n] for n,_ in mean_traces.items())
                raw_weights[name] = 1.0 / mean_trace * sum_all_mean_traces
            else:
                raw_weights[name] = 1.0

        # Normalize weights
        total_raw_weight = sum(raw_weights.values())
        normalization = len(raw_weights)/total_raw_weight
        
        normalized_weights = {
            name: raw_weights[name] * normalization
            for name, weight in raw_weights.items()
        }

        return normalized_weights

    def update_weights(self, current_step: int) -> Optional[Dict[str, float]]:
        """
        Update weights if needed at current training step.

        Args:
            current_step: Current training step

        Returns:
            Updated weights if computed, None otherwise
        """
        if not (current_step >= self.config.training.ntk_start_step and
                current_step % self.config.training.ntk_update_freq == 0):
            return None

        self.last_update_step = current_step
        weights = self.compute_weights()
        return weights

    def get_current_weights(self) -> Dict[str, float]:
            """Get the current weights (or uniform weights if none computed)."""
            return self.current_weights if self.current_weights else {}
    


@dataclass
class ALConfig:
    """Configuration for Augmented Lagrangian method"""
    beta: float = 100.0                    # Penalty parameter β
    lr_lambda: float = 1e-3                # Learning rate for multipliers
    start_step: int = 0                    # When to start AL weighting
    constraint_tolerance: float = 1e-6     # Target constraint satisfaction
    lambda_max: float = 100.0              # Clipping bound for multipliers
    update_frequency: int = 1              # How often to update multipliers


class ALWeightManager:
    """
    Manages learnable Lagrange multipliers for AL-PINNs.
    
    Mathematical Framework:
    - Maintains λ ∈ ℝ^m for m constraint equations
    - Updates: λ ← λ + η_λ * C(θ)
    - Computes: β‖C‖² + ⟨λ, C⟩
    """
    
    def __init__(self, sampler, config: ALConfig, device: torch.device):
        self.sampler = sampler
        self.config = config
        self.device = device
        
        # Will be initialized on first call
        self.lambda_params = None
        self.constraint_names = []
        self.is_initialized = False
        
        # Tracking for analysis
        self.constraint_history = {}
        self.multiplier_history = {}


    def _initialize_multipliers(self) -> Dict[str, nn.Parameter]:
        """
        Initialize learnable multipliers based on actual sampling dimensions.
        
        Args:
            sample_dict: Dictionary containing sampled points for sizing
            
        Returns:
            Dictionary of learnable λ parameters
        """
        self.lambda_params = {}

        # Access batch sizes directly from sampler
        batch_config = self.sampler.batch_sizes
        
        # Boundary constraint multipliers (7 types)
        n_boundary = batch_config['BC']  # e.g., 100 points
        boundary_types = ['cv_mf_bc', 'av_mf_bc', 'u_mf_bc', 
                         'cv_fs_bc', 'av_fs_bc', 'u_fs_bc', 'h_fs_bc']
        
        for bc_type in boundary_types:
            self.lambda_params[f'lambda_{bc_type}'] = nn.Parameter(
                torch.zeros(n_boundary, device=self.device)
            )
            self.constraint_names.append(bc_type)
        
        # Initial condition multipliers (4 types)
        n_initial = batch_config['IC']  # e.g., 50 points
        initial_types = ['cv_ic', 'av_ic', 'h_ic', 'poisson_ic']
        
        for ic_type in initial_types:
            self.lambda_params[f'lambda_{ic_type}'] = nn.Parameter(
                torch.zeros(n_initial, device=self.device)
            )
            self.constraint_names.append(ic_type)
        
        # Film thickness multipliers (1 type)
        n_film = batch_config['L']  # e.g., 25 points
        self.lambda_params['lambda_L_ic'] = nn.Parameter(
            torch.zeros(n_film, device=self.device)
        )
        self.constraint_names.append('L_ic')
        
        # Log initialization
        total_params = sum(p.numel() for p in self.lambda_params.values())
        print(f"🔗 AL-PINNs Multiplier Initialization:")
        print(f"  Boundary: {len(boundary_types)} types × {n_boundary} points = {len(boundary_types) * n_boundary}")
        print(f"  Initial: {len(initial_types)} types × {n_initial} points = {len(initial_types) * n_initial}")
        print(f"  Film: 1 type × {n_film} points = {n_film}")
        print(f"  📊 Total multiplier parameters: {total_params}")
        
        return 
    
    def get_multiplier_parameters(self) -> List[nn.Parameter]:
        """Get all multiplier parameters for optimizer"""
        if self.lambda_params is None:
            return []
        return list(self.lambda_params.values())
    

    def update_multipliers(self, constraint_violations: Dict[str, torch.Tensor]):
        """
        Update Lagrange multipliers: λ ← λ + η_λ * C(θ)
        
        Args:
            constraint_violations: Dict of constraint violations by type
        """
        if not self.is_initialized:
            return
            
        with torch.no_grad():
            for constraint_name, violation in constraint_violations.items():
                lambda_key = f'lambda_{constraint_name}'
                
                if lambda_key in self.lambda_params:
                    # Standard AL update: λ ← λ + η_λ * C
                    self.lambda_params[lambda_key] += self.config.lr_lambda * violation.detach()
                    
                    # Clip to prevent unbounded growth
                    self.lambda_params[lambda_key].clamp_(-self.config.lambda_max, 
                                                        self.config.lambda_max)
                    
    def get_constraint_satisfaction_metrics(self, constraint_violations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute constraint satisfaction metrics for monitoring"""
        metrics = {}
        
        for name, violation in constraint_violations.items():
            metrics[f'max_{name}'] = torch.max(torch.abs(violation)).item()
            metrics[f'mean_{name}'] = torch.mean(torch.abs(violation)).item()
            metrics[f'std_{name}'] = torch.std(violation).item()
        
        return metrics
    
    def log_multiplier_stats(self) -> Dict[str, float]:
        """Log statistics about current multipliers"""
        if not self.is_initialized:
            return {}
            
        stats = {}
        for name, param in self.lambda_params.items():
            stats[f'{name}_mean'] = torch.mean(param).item()
            stats[f'{name}_std'] = torch.std(param).item()
            stats[f'{name}_max'] = torch.max(torch.abs(param)).item()
        
        return stats
        
# Convenience functions for easy integration with training.py
def setup_ntk_weighting(
        networks,
        physics,
        sampler,
        config: Optional[Dict[str, Any]] = None
) -> NTKWeightManager:
    """
    Convenience function to setup NTK weighting for modular PINNACLE.

    Args:
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        sampler: CollocationSampler instance
        config: Optional configuration dictionary

    Returns:
        Configured NTKWeightManager
    """
    return NTKWeightManager(networks, physics, sampler, config)


def create_loss_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Create loss weights based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of loss weights
    """
    weight_strategy = config.training.weight_strat
    if weight_strategy == 'None':
        return {
            'interior': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'film_physics': 1.0
        }
    
    elif weight_strategy == 'batch_size':
        batch_config = config.get('batch_size', {})
        return {
            'interior': 1.0 / batch_config.get('interior'),
            'boundary': 1.0 / batch_config.get('BC'),
            'initial': 1.0 / batch_config.get('IC'),
            'film_physics': 1.0 / batch_config.get('L')
        }
    elif weight_strategy == 'ntk' or weight_strategy == "hybrid_ntk":
        # NTK weights will be computed dynamically during training
        return {
            'interior': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'film_physics': 1.0
        }