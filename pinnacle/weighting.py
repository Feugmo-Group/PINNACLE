# pinnacle/weighting.py
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
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Any, NamedTuple
from dataclasses import dataclass



class JacobianResults(NamedTuple):
    """Container for Jacobian computation results"""
    jacobian: torch.Tensor  # Jacobian matrix [batch_size, num_parameters]
    batch_size: int  # Effective batch size used
    parameter_count: int  # Total number of parameters


@dataclass
class NTKConfig:
    """Configuration for NTK-based weighting"""
    update_frequency: int = 100  # How often to update weights
    start_step: int = 1000  # When to start NTK weighting
    trace_estimation_samples: int = 256  # Samples for trace estimation #TODO: What is this doing!


def compute_jacobian(
        outputs: torch.Tensor,
        parameters: List[torch.nn.Parameter],
        device: torch.device
) -> JacobianResults:
    """
    Compute Jacobian matrix using fast batched gradient computation.

    Args:
        outputs: Network outputs [batch_size] or [batch_size, 1]
        parameters: List of network parameters
        device: PyTorch device

    Returns:
        JacobianResults with jacobian matrix and metadata
    """
    # Ensure outputs are 1D
    outputs = outputs.reshape(-1)
    batch_size = outputs.shape[0]

    if batch_size == 0:
        raise ValueError("Empty outputs tensor provided")

    try:
        # Fast batched gradient computation
        grads = torch.autograd.grad(
            outputs,
            parameters,
            grad_outputs=torch.eye(batch_size, device=device),
            is_grads_batched=True,
            retain_graph=True,
            allow_unused=True
        )

        # Process gradients
        valid_grads = []
        for grad in grads:
            if grad is not None:
                grad_flat = grad.flatten(start_dim=1)
                valid_grads.append(grad_flat)

        if not valid_grads:
            raise ValueError("No valid gradients computed")

        # Concatenate all parameter gradients
        jacobian = torch.cat(valid_grads, dim=1)  # [batch_size, total_parameters]

        return JacobianResults(
            jacobian=jacobian,
            batch_size=batch_size,
            parameter_count=jacobian.shape[1]
        )

    except RuntimeError as e:
        if "is_grads_batched" in str(e):
            warnings.warn("Batched gradients not supported")
        else:
            raise e

def compute_ntk_trace(jacobian_result: JacobianResults) -> torch.Tensor:
    """
    Compute trace of Neural Tangent Kernel matrix.

    Args:
        jacobian_result: Result from compute_jacobian

    Returns:
        Scalar tensor with NTK trace
    """
    jacobian = jacobian_result.jacobian
    # Trace = sum of diagonal elements = sum of squared norms
    trace = torch.sum(jacobian * jacobian)
    return trace


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
        self.config = config or NTKConfig()
        self.device = physics.device

        # Storage for computed batch sizes (optimization)
        self.optimal_batch_sizes = {}

        # Current weights
        self.current_weights = {}

        # Update tracking
        self.last_update_step = -1

    def should_update_weights(self, current_step: int) -> bool:
        """Check if weights should be updated at current step."""
        return (current_step >= self.config.start_step and
                current_step % self.config.update_frequency == 0 and
                current_step != self.last_update_step)

    def compute_loss_weight(
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
        # Determine optimal batch size for this loss
        #TODO: Use nexPinnacle statistics for calculating optimal batch_size
        if loss_name not in self.optimal_batch_sizes:
            batch_size = min(len(loss_residuals), self.config.trace_estimation_samples)
            self.optimal_batch_sizes[loss_name] = batch_size
        else:
            batch_size = self.optimal_batch_sizes[loss_name]

        # Sample residuals if needed
        if len(loss_residuals) > batch_size:
            indices = torch.randperm(len(loss_residuals), device=self.device)[:batch_size]
            sampled_residuals = loss_residuals[indices]
        else:
            sampled_residuals = loss_residuals

        # Compute Jacobian using functional approach
        jacobian_result = compute_jacobian(sampled_residuals, self.networks.get_all_parameters(), self.device)

        # Compute NTK trace using functional approach
        ntk_trace = compute_ntk_trace(jacobian_result)

        return ntk_trace.item(), jacobian_result.batch_size

    def extract_all_residuals(self) -> Dict[str, torch.Tensor]:
        """
        Extract all residuals using the modified loss functions with return_residuals=True.

        This leverages the exact same computation logic as training without duplication.
        """
        # Import the modified loss functions
        from .losses import (
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

        try:
            # Extract all residuals using modified loss functions
            all_residuals = self.extract_all_residuals()

            ntk_traces = {}
            batch_sizes = {}

            # Compute NTK trace for each component
            for component_name, residual in all_residuals.items():
                if len(residual) > 0:
                    ntk_trace, effective_batch_size = self.compute_loss_weight(residual, component_name)
                    ntk_traces[component_name] = ntk_trace
                    batch_sizes[component_name] = effective_batch_size
                    print(f"  {component_name}: NTK trace = {ntk_trace:.2e}, batch = {effective_batch_size}")
                else:
                    warnings.warn(f"Empty residuals for {component_name}, skipping")

        except Exception as e:
            warnings.warn(f"Failed to compute NTK weights: {e}")
            # Fallback to uniform weights
            return {
                'cv_pde': 1.0, 'av_pde': 1.0, 'h_pde': 1.0, 'poisson_pde': 1.0,
                'boundary': 1.0, 'initial': 1.0, 'film_physics': 1.0
            }

        if not ntk_traces:
            warnings.warn("No valid NTK traces computed, using uniform weights")
            return {
                'cv_pde': 1.0, 'av_pde': 1.0, 'h_pde': 1.0, 'poisson_pde': 1.0,
                'boundary': 1.0, 'initial': 1.0, 'film_physics': 1.0
            }

        # Compute normalized weights
        weights = self._normalize_ntk_weights(ntk_traces, batch_sizes)

        print(f"📊 Updated NTK weights: {weights}")
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

        # Compute raw weights (inverse of mean traces)
        raw_weights = {}
        for name, mean_trace in mean_traces.items():
            if mean_trace > 1e-12:  # Avoid division by zero
                raw_weights[name] = 1.0 / mean_trace
            else:
                raw_weights[name] = 1.0

        # Normalize weights
        total_weight = sum(raw_weights.values())
        num_losses = len(raw_weights)

        if total_weight > 1e-12:
            normalized_weights = {
                name: (weight / total_weight) * num_losses
                for name, weight in raw_weights.items()
            }
        else:
            # Fallback to uniform weights
            normalized_weights = {name: 1.0 for name in raw_weights.keys()}

        return normalized_weights

    def update_weights(self, current_step: int) -> Optional[Dict[str, float]]:
        """
        Update weights if needed at current training step.

        Args:
            current_step: Current training step

        Returns:
            Updated weights if computed, None otherwise
        """
        if not self.should_update_weights(current_step):
            return None

        self.last_update_step = current_step
        weights = self.compute_weights()
        return weights

    def get_current_weights(self) -> Dict[str, float]:
            """Get the current weights (or uniform weights if none computed)."""
            return self.current_weights if self.current_weights else {}

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
    ntk_config = NTKConfig()

    if config and 'training' in config and 'weight_strat' in config['training']:
        if config['training']['weight_strat'] == "ntk":
            training_cfg = config['training']
            ntk_config.update_frequency = training_cfg.get('ntk_update_freq', 100)
            ntk_config.start_step = training_cfg.get('ntk_start_step', 0)
            ntk_config.trace_estimation_samples = training_cfg.get('ntk_samples', 256)

    return NTKWeightManager(networks, physics, sampler, ntk_config)


def create_loss_weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Create loss weights based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of loss weights
    """
    weight_strategy = config.get('training', {}).get('weight_strat', 'uniform')

    if weight_strategy == 'uniform':
        return {
            'interior': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'film_physics': 1.0
        }
    elif weight_strategy == 'batch_size':
        batch_config = config.get('batch_size', {})
        return {
            'interior': 1.0 / batch_config.get('interior', 1),
            'boundary': 1.0 / batch_config.get('BC', 1),
            'initial': 1.0 / batch_config.get('IC', 1),
            'film_physics': 1.0 / batch_config.get('L', 1)
        }
    elif weight_strategy == 'ntk':
        # NTK weights will be computed dynamically during training
        return {
            'interior': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'film_physics': 1.0
        }
    else:
        # Manual weights (should be specified in config)
        manual_weights = config.get('training', {}).get('manual_weights', {})
        default_weights = {
            'interior': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'film_physics': 1.0
        }
        default_weights.update(manual_weights)
        return default_weights