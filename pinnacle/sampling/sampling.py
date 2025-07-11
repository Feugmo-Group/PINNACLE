# sampling/sampling.py
"""
Simple collocation point sampling for PINNACLE.

This module provides the basic sampling functions needed for PINN training.
"""
import torch
from typing import Dict, Any, Tuple
import torch.nn as nn

torch.manual_seed(995)


class AdaptiveCollocationSampler:
    """Adaptive Collocation sampler

    Generates denser set of random points near areas with high residuals, or steep gradients.

    Based on:
    PF-PINNs: Physics-informed neural networks for solving coupled Allen-Cahn and Cahn-Hilliard phase field equations  Nanxi Chen a, Sergio Lucarini b,c, Rujin Ma a, Airong Chen a, Chuanjie Cui d, ,∗
    """

    def __init__(self, config: Dict[str, Any], physics, device: torch.device):
        """
        Initialize the collocation sampler.

        Args:
            config: Configuration dictionary
            physics: ElectrochemicalPhysics instance
            device: PyTorch device
        """
        self.config = config
        self.physics = physics
        self.device = device

        # Store batch sizes for easy access
        self.batch_sizes = config["batch_size"]

        # Base set configuration
        self.base_set_size = config.sampling.adaptive.base_set_size  # e.g., 60000
        self.active_set_size = config.sampling.adaptive.active_set_size  # e.g., 8000
        self.x_base_points = config.sampling.adaptive.x_base_points  # e.g., 100
        self.t_base_points = config.sampling.adaptive.t_base_points  # e.g., 100
        self.E_base_points = config.sampling.adaptive.E_base_points  # e.g., 50

        # Fixed grids (never change)
        self.t_base_grid = torch.linspace(0, 1, self.t_base_points, device=device)
        E_min = physics.geometry.E_min / physics.scales.phic
        E_max = physics.geometry.E_max / physics.scales.phic
        self.E_base_grid = torch.linspace(
            E_min, E_max, self.E_base_points, device=device
        )

        # Dynamic components
        self.current_L_max = None
        self.base_set = None
        self.active_set = None

        # Update tracking
        self.last_base_update_step = -1
        self.base_update_frequency = config.sampling.adaptive.base_update_freq  # e.g., 1000
        self.last_L_max = 0.0
        self.L_growth_threshold = config.sampling.adaptive.L_growth_threshold  # e.g., 0.2 (20%)

        # Batching for efficiency
        self.residual_batch_size = config.sampling.adaptive.residual_batch_size  # e.g., 1000

    def estimate_current_domain_extent(self, networks: Dict[str, nn.Module]) -> float:
        """Estimate maximum film thickness L(t,E) across domain


        Args:
            networks: Dictionary of segregated networks for problem

        Returns:
            Float value of maximum L extend plus a safety factor. Allowing for
            safe capture of full domain

        """

        # Create representative (t,E) sample grid
        n_sample = 50  # Manageable size for quick estimation
        t_sample = torch.linspace(0, 1, n_sample, device=self.device)
        E_sample = torch.linspace(
            self.E_base_grid.min(), self.E_base_grid.max(), n_sample, device=self.device
        )

        # Create meshgrid and predict L(t,E)
        T_mesh, E_mesh = torch.meshgrid(t_sample, E_sample, indexing="ij")
        L_inputs = torch.stack([T_mesh.flatten(), E_mesh.flatten()], dim=1)

        with torch.no_grad():
            L_predictions = networks["film_thickness"](L_inputs)

        # Get maximum with safety margin
        L_max_current = torch.max(L_predictions).item()
        safety_factor = self.config.sampling.adaptive.safety_factor  # e.g., 1.2

        return L_max_current * safety_factor

    def should_update_base_set(self, current_step, networks):
        """Determine if base set needs regeneration"

        Args:
            current_step: The current step of training
            networks: Dictionary of segregated networks for problem
        Returns:
            Boolean value, should we update the base set of points?

        """

        # Check step frequency
        steps_since_update = current_step - self.last_base_update_step
        if steps_since_update < self.base_update_frequency:
            return False

        # Check domain growth
        current_L_max = self.estimate_current_domain_extent(networks)
        if self.last_L_max == 0:
            return True  # First time

        growth_ratio = (current_L_max - self.last_L_max) / self.last_L_max
        if growth_ratio > self.L_growth_threshold:
            return True

        return False

    def regenerate_base_set(self, networks: Dict[str, nn.Module]):
        """Generate new base set with updated spatial domain

        Args:
            networks: Dictionary of networks
        Returns:
            None

        """

        print(f"Regenerating adaptive base set...")

        # Step 1: Get current domain extent
        self.current_L_max = self.estimate_current_domain_extent(networks)
        print(f"  Current L_max: {self.current_L_max:.6f}")

        # Step 2: Create new spatial grid
        x_base_new = torch.linspace(
            0, self.current_L_max, self.x_base_points, device=self.device
        )

        # Step 3: Generate full meshgrid
        X_mesh, T_mesh, E_mesh = torch.meshgrid(
            x_base_new,
            self.t_base_grid,
            self.E_base_grid,  # t and E do not change in extent
            indexing="ij",
        )

        # Step 4: Filter valid points
        self.base_set = self._filter_valid_points(X_mesh, T_mesh, E_mesh, networks)

        # Step 5: Update tracking
        self.last_L_max = self.current_L_max

        print(f"  Generated {len(self.base_set)} valid base points")

    def _filter_valid_points(self, X_mesh, T_mesh, E_mesh, networks):
        """Remove points where x > L(t,E)

        Args:
            X_mesh: Component of Torch Meshgrid
            T_mesh: Component of Torch Meshgrid
            E_mesh: Component of Torch Meshgrid
            netowrks; Dict of segereated networks
        Returns:
            tensor of all valid points

        """

        # Flatten meshgrid
        x_flat = X_mesh.flatten()
        t_flat = T_mesh.flatten()
        E_flat = E_mesh.flatten()

        print(f"  Filtering {len(x_flat)} candidate points...")

        # Predict L(t,E) for all points (batched for memory)
        valid_points = []

        for i in range(0, len(x_flat), self.residual_batch_size):
            end_idx = min(i + self.residual_batch_size, len(x_flat))

            # Batch data
            x_batch = x_flat[i:end_idx]
            t_batch = t_flat[i:end_idx]
            E_batch = E_flat[i:end_idx]

            # Predict L(t,E) for this batch
            L_inputs_batch = torch.stack([t_batch, E_batch], dim=1)
            with torch.no_grad():
                L_pred_batch = networks["film_thickness"](L_inputs_batch).squeeze()

            # Keep valid points: x ≤ L(t,E)
            valid_mask = x_batch <= L_pred_batch

            if valid_mask.any():
                valid_batch_points = torch.stack(
                    [x_batch[valid_mask], t_batch[valid_mask], E_batch[valid_mask]],
                    dim=1,
                )
                valid_points.append(valid_batch_points)

        # Concatenate all valid points
        return torch.cat(valid_points, dim=0)

    def compute_residual_indicators(self, networks, physics):
        """Compute PDE residuals across entire base set
        
        Args:
            networks: Dictionary of Segregated neural networks
            phyiscs: ElectochemicalPhyiscs Object
        Returns:
            final_residuals: Dictionary of tensors of residuals at each point in base_set grid
        
        """

        if self.base_set is None:
            raise ValueError("Base set not initialized")

        print(f"Computing residuals on {len(self.base_set)} base points...")

        all_residuals = {
            'cv_pde': [],
            'av_pde': [], 
            'h_pde': [],
            'poisson_pde': []
        }

        # Process base set in batches
        for i in range(0, len(self.base_set), self.residual_batch_size):
            end_idx = min(i + self.residual_batch_size, len(self.base_set))
            
            # Extract batch
            batch_points = self.base_set[i:end_idx]
            x_batch = batch_points[:, 0:1].requires_grad_(True)
            t_batch = batch_points[:, 1:2].requires_grad_(True) 
            E_batch = batch_points[:, 2:3]
            
            # Compute PDE residuals for this batch
            cv_res, av_res, h_res, poisson_res = physics.compute_pde_residuals(
                x_batch, t_batch, E_batch, networks
            )
            
            # Store residuals (detach to save memory)
            all_residuals['cv_pde'].append(cv_res.detach())
            all_residuals['av_pde'].append(av_res.detach())
            all_residuals['h_pde'].append(h_res.detach())
            all_residuals['poisson_pde'].append(poisson_res.detach())

        # Concatenate all batches
        final_residuals = {}
        for key in all_residuals:
            final_residuals[key] = torch.cat(all_residuals[key], dim=0)

        return final_residuals
    
    def combine_residuals(self, residuals: torch.Tensor):
        """Combine residuals from multiple PDEs into single indicator
        
        Args:
            residuals: Dictionary of tensors of residuals at each point in base_set grid

        returns:
            torch.Tensor of summed residuals at each collocation point
        
        """
        
        combined = (torch.abs(residuals['cv_pde']) + 
                    torch.abs(residuals['av_pde']) +
                    torch.abs(residuals['h_pde']) + 
                    torch.abs(residuals['poisson_pde']))

        return combined
    
    def select_adaptive_points(self, residuals):
        """Select top-k points based on residual magnitudes"""
        
        # Combine all PDE residuals
        combined_residual = self.combine_residuals(residuals)
        
        print(f"  Residual range: [{combined_residual.min():.2e}, {combined_residual.max():.2e}]")
        
        # Select top-k highest residual points
        if len(combined_residual) <= self.active_set_size:
            # Use all points if base set is small
            selected_indices = torch.arange(len(combined_residual), device=self.device)
        else:
            # Select top-k points
            _, selected_indices = torch.topk(
                combined_residual, 
                k=self.active_set_size, 
                largest=True
            )
        
        # Extract corresponding points
        self.active_set = self.base_set[selected_indices].clone()
        
        print(f"  Selected {len(self.active_set)} adaptive points")
        print(f"  Selection residual range: "
            f"[{combined_residual[selected_indices].min():.2e}, "
            f"{combined_residual[selected_indices].max():.2e}]")
        
        return self.active_set

class CollocationSampler:
    """
    Simple collocation point sampler for electrochemical PINNs.

    Generates the different types of training points needed for physics-informed
    neural network training.
    """

    def __init__(self, config: Dict[str, Any], physics, device: torch.device):
        """
        Initialize the collocation sampler.

        Args:
            config: Configuration dictionary
            physics: ElectrochemicalPhysics instance
            device: PyTorch device
        """
        self.config = config
        self.physics = physics
        self.device = device

        # Store batch sizes for easy access
        self.batch_sizes = config["batch_size"]

    def sample_interior_points(
        self, networks
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample interior collocation points for PDE residuals.

        Args:
            networks: NetworkManager instance

        Returns:
            Tuple of (x, t, E) tensors with requires_grad=True for x and t
        """
        batch_size = self.batch_sizes["interior"]

        # Sample time and applied potential
        t = torch.rand(batch_size, 1, device=self.device, requires_grad=True)
        single_E = (
            torch.rand(1, 1, device=self.device)
            * (self.physics.geometry.E_max - self.physics.geometry.E_min)
            + self.physics.geometry.E_min
        )
        E = single_E.expand(batch_size, 1)

        # Get film thickness prediction
        L_pred = networks["film_thickness"](torch.cat([t, E], dim=1))

        # Sample spatial coordinates within [0, L(t,E)]
        x = torch.rand(batch_size, 1, device=self.device, requires_grad=True) * L_pred

        return x, t, E

    def sample_boundary_points(
        self, networks
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample boundary collocation points for boundary conditions.

        Args:
            networks: NetworkManager instance

        Returns:
            Tuple of (x, t, E) tensors for boundary points
        """
        batch_size = 2 * self.batch_sizes["BC"]

        # Sample time and applied potential
        t = torch.rand(batch_size, 1, device=self.device, requires_grad=True)
        single_E = (
            torch.rand(1, 1, device=self.device)
            * (self.physics.geometry.E_max - self.physics.geometry.E_min)
            + self.physics.geometry.E_min
        )
        E = single_E.expand(batch_size, 1)

        # Predict L for f/s boundary
        L_inputs = torch.cat([t, E], dim=1)
        L_pred = networks["film_thickness"](L_inputs)

        half_batch = batch_size // 2

        # Metal/film interface points (x = 0)
        x_mf = torch.zeros(half_batch, 1, device=self.device, requires_grad=True)
        t_mf = t[:half_batch]
        E_mf = E[:half_batch]

        # Film/solution interface points (x = L)
        x_fs = L_pred[half_batch:]
        t_fs = t[half_batch:]
        E_fs = E[half_batch:]

        # Combine boundary points
        x_boundary = torch.cat([x_mf, x_fs], dim=0)
        t_boundary = torch.cat([t_mf, t_fs], dim=0)
        E_boundary = torch.cat([E_mf, E_fs], dim=0)

        return x_boundary, t_boundary, E_boundary

    def sample_initial_points(
        self, networks
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample initial condition points at t = 0.

        Args:
            networks: NetworkManager instance

        Returns:
            Tuple of (x, t, E) tensors for initial condition points
        """
        batch_size = self.batch_sizes["IC"]

        # Initial time (t = 0)
        t = torch.zeros(batch_size, 1, device=self.device, requires_grad=True)
        single_E = (
            torch.rand(1, 1, device=self.device)
            * (self.physics.geometry.E_max - self.physics.geometry.E_min)
            + self.physics.geometry.E_min
        )
        E = single_E.expand(batch_size, 1)

        # Get initial film thickness
        L_initial_pred = networks["film_thickness"](torch.cat([t, E], dim=1))

        # Sample spatial coordinates
        x = (
            torch.rand(batch_size, 1, device=self.device, requires_grad=True)
            * L_initial_pred
        )

        return x, t, E

    def sample_film_physics_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points for film growth physics constraint.

        Returns:
            Tuple of (t, E) tensors for film physics constraint
        """
        batch_size = self.batch_sizes["L"]

        # Sample time and applied potential
        t = torch.rand(batch_size, 1, device=self.device, requires_grad=True)
        single_E = (
            torch.rand(1, 1, device=self.device)
            * (self.physics.geometry.E_max - self.physics.geometry.E_min)
            + self.physics.geometry.E_min
        )
        E = single_E.expand(batch_size, 1)

        return t, E
