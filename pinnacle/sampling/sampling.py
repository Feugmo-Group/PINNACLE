# sampling/sampling.py
"""
Simple collocation point sampling for PINNACLE.

This module provides the basic sampling functions needed for PINN training.
"""
import torch
from typing import Dict, Any, Tuple
from physics.physics import ElectrochemicalPhysics
from losses.losses import compute_boundary_loss
from losses.losses import compute_initial_loss
from losses.losses import compute_film_physics_loss
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

        # Adaptive set sizes for each component
        self.adaptive_sizes = {
            'interior': config.sampling.adaptive.interior_points,      # e.g., 8000
            'boundary': config.sampling.adaptive.boundary_points,      # e.g., 2000  
            'initial': config.sampling.adaptive.initial_points,        # e.g., 1500
            'film_physics': config.sampling.adaptive.film_points       # e.g., 1000
        }
        
        # Base set sizes (larger pools to select from)
        self.base_set_sizes = {
            'interior': config.sampling.adaptive.interior_base_size,   # e.g., 60000
            'boundary': config.sampling.adaptive.boundary_base_size,   # e.g., 20000
            'initial': config.sampling.adaptive.initial_base_size,     # e.g., 15000
            'film_physics': config.sampling.adaptive.film_base_size    # e.g., 10000
        }
        
        # Fixed grids (time and potential)
        self.t_base_points = config.sampling.adaptive.t_base_points
        self.E_base_points = config.sampling.adaptive.E_base_points
        self.t_base_grid = torch.linspace(0, 1, config.sampling.adaptive.t_base_points, device=device)
        E_min = physics.geometry.E_min / physics.scales.phic
        E_max = physics.geometry.E_max / physics.scales.phic
        self.E_base_grid = torch.linspace(E_min, E_max, config.sampling.adaptive.E_base_points, device=device)
        
        # Current adaptive sets
        self.adaptive_sets = {
            'interior': None,
            'boundary': None, 
            'initial': None,
            'film_physics': None
        }

        # Base sets for each component
        self.base_sets = {
            'interior': None,
            'boundary': None,
            'initial': None, 
            'film_physics': None
        }   
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

    def should_update_base_set(self, current_step:int, networks:Dict[str,nn.Module]):
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

    def regenerate_all_base_sets(self, networks:Dict[str,nn.Module]):
        """Generate base sets for all loss components
        Args:
            networks: Dictionary of neural networks
        Returns:
            None
        
        """
        
        print(f"Regenerating all adaptive base sets...")
        
        # Update domain extent
        self.current_L_max = self.estimate_current_domain_extent(networks)
        print(f"  Current L_max: {self.current_L_max:.6f}")
        
        # Generate base sets for each component
        self.base_sets['interior'] = self._generate_interior_base_set(networks)
        self.base_sets['boundary'] = self._generate_boundary_base_set(networks)
        self.base_sets['initial'] = self._generate_initial_base_set(networks)
        self.base_sets['film_physics'] = self._generate_film_base_set()
        
        # Update tracking
        self.last_L_max = self.current_L_max
        
        total_points = sum(len(base_set) for base_set in self.base_sets.values())
        print(f"  Generated {total_points} total base points across all components")

    def _generate_interior_base_set(self, networks:Dict[str,nn.Module]):
        """Generate base set for interior PDE residuals
        Args: 
            networks: Dictionary of neural networks
        Returns: 
            torch.Tensor of valid interior points
        
        """
        
        # Create spatial grid based on current domain
        x_base = torch.linspace(0, self.current_L_max, 
                            self.config.sampling.adaptive.x_base_points, device=self.device)
        
        # Create meshgrid for interior domain
        X_mesh, T_mesh, E_mesh = torch.meshgrid(
            x_base, self.t_base_grid, self.E_base_grid, indexing='ij'
        )
        
        # Filter valid points: x <= L(t,E)
        return self._filter_valid_points(X_mesh, T_mesh, E_mesh, networks)

    def _generate_boundary_base_set(self, networks:Dict[str,nn.Module]):
        """Generate base set for boundary condition residuals
        Args:
            networks: Dictionary of neural networks
        Returns:
            torch.Tensor of boundary points
        
        """
        # Sample time and potential
        n_sample = self.base_set_sizes['boundary']
        t_sample = torch.rand(n_sample, device=self.device)
        E_sample = torch.rand(n_sample, device=self.device) * (
            self.E_base_grid.max() - self.E_base_grid.min()
        ) + self.E_base_grid.min()
        
        # Get film thickness predictions
        L_inputs = torch.stack([t_sample, E_sample], dim=1)
        with torch.no_grad():
            L_pred = networks['film_thickness'](L_inputs).squeeze()
        
        # Create boundary points: half at x=0 (m/f), half at x=L(t,E) (f/s)
        half_size = n_sample // 2
        
        # Metal/film interface (x = 0)
        x_mf = torch.zeros(half_size, device=self.device)
        t_mf = t_sample[:half_size]
        E_mf = E_sample[:half_size]
        
        # Film/solution interface (x = L(t,E))
        x_fs = L_pred[half_size:]
        t_fs = t_sample[half_size:]
        E_fs = E_sample[half_size:]
        
        # Combine boundary points
        x_boundary = torch.cat([x_mf, x_fs])
        t_boundary = torch.cat([t_mf, t_fs])
        E_boundary = torch.cat([E_mf, E_fs])
        
        return torch.stack([x_boundary, t_boundary, E_boundary], dim=1)

    def _generate_initial_base_set(self, networks:Dict[str,nn.Module]):
        """Generate base set for initial condition residuals
        Args:
            networks: Dictionary of neural networks
        Returns:
            Torch.Tensor of initial points
        """
        
        n_sample = self.base_set_sizes['initial']
        
        # Time is always zero for initial conditions
        t_sample = torch.zeros(n_sample, device=self.device)
        
        # Sample potential
        E_sample = torch.rand(n_sample, device=self.device) * (
            self.E_base_grid.max() - self.E_base_grid.min()
        ) + self.E_base_grid.min()
        
        # Get initial film thickness
        L_inputs = torch.stack([t_sample, E_sample], dim=1)
        with torch.no_grad():
            L_initial = networks['film_thickness'](L_inputs).squeeze()
        
        # Sample spatial coordinates within initial film
        x_sample = torch.rand(n_sample, device=self.device) * L_initial
        
        return torch.stack([x_sample, t_sample, E_sample], dim=1)

    def _generate_film_base_set(self):
        """Generate base set for film physics residuals
        Args:
            None
        Returns:
            torch.Tensor of film_sample points
        """
        
        n_sample = self.base_set_sizes['film_physics']
        
        # Sample time and potential (no spatial dimension for film thickness)
        t_sample = torch.rand(n_sample, device=self.device)
        E_sample = torch.rand(n_sample, device=self.device) * (
            self.E_base_grid.max() - self.E_base_grid.min()
        ) + self.E_base_grid.min()
        
        return torch.stack([t_sample, E_sample], dim=1)

    def _filter_valid_points(self, X_mesh:torch.Tensor, T_mesh:torch.Tensor, E_mesh:torch.Tensor, networks:Dict[str,nn.Module]):
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
    
    
    def compute_all_residuals(self, networks:Dict[str,nn.Module], physics:ElectrochemicalPhysics):
        """Compute residuals for all loss components
        Args:
            networks: Dictionary of neural networks
            physics: ElectrochemicalPhysics Object
        Returns:
            all_residuals: Dictionary of residuals for each category
        """
        
        print(f"Computing residuals on all base sets...")
        
        all_residuals = {}
        
        # Interior PDE residuals
        all_residuals.update(self._compute_interior_residuals(networks, physics))
        
        # Boundary condition residuals
        all_residuals['boundary'] = self._compute_boundary_residuals(networks, physics)
        
        # Initial condition residuals  
        all_residuals['initial'] = self._compute_initial_residuals(networks, physics)
        
        # Film physics residuals
        all_residuals['film_physics'] = self._compute_film_residuals(networks, physics)
        
        return all_residuals

    def _compute_interior_residuals(self, networks:Dict[str,nn.Module], physics:ElectrochemicalPhysics):
        """Compute PDE residuals on interior base set
        Args:
            networks: Dictionary of neural networks
            physics: Elcectrochemical Physics Object
        Returns: 
            Dictionary of interior residuals
        """
        base_set = self.base_sets['interior']
        batch_size = self.config.sampling.adaptive.residual_batch_size
        
        residuals = {'cv_pde': [], 'av_pde': [], 'h_pde': [], 'poisson_pde': []}
        
        # Process in batches
        for i in range(0, len(base_set), batch_size):
            end_idx = min(i + batch_size, len(base_set))
            
            # Extract batch
            batch_points = base_set[i:end_idx]
            x_batch = batch_points[:, 0:1].requires_grad_(True)
            t_batch = batch_points[:, 1:2].requires_grad_(True)
            E_batch = batch_points[:, 2:3]
            
            # Compute PDE residuals
            cv_res, av_res, h_res, poisson_res = physics.compute_pde_residuals(
                x_batch, t_batch, E_batch, networks
            )
            
            # Store residuals
            residuals['cv_pde'].append(cv_res.detach())
            residuals['av_pde'].append(av_res.detach())
            residuals['h_pde'].append(h_res.detach())
            residuals['poisson_pde'].append(poisson_res.detach())
        
        # Concatenate all batches
        return {key: torch.cat(values, dim=0) for key, values in residuals.items()}
    
    def _compute_boundary_residuals(self, networks:Dict[str,nn.Module], physics:ElectrochemicalPhysics):
        """Compute boundary condition residuals
        Args:
            networks: Dictionary of neural networks
            physics: Elcectrochemical Physics Object
        Returns:
            torch.Tensor all boundary residuals at every collocation point
        """
        
        base_set = self.base_sets['boundary']
        batch_size = self.config.sampling.adaptive.residual_batch_size
        
        all_residuals = []
        
        # Process in batches using existing boundary loss computation
        for i in range(0, len(base_set), batch_size):
            end_idx = min(i + batch_size, len(base_set))
            
            batch_points = base_set[i:end_idx]
            x_batch = batch_points[:, 0:1].requires_grad_(True)
            t_batch = batch_points[:, 1:2].requires_grad_(True)
            E_batch = batch_points[:, 2:3]
            
            # Use existing boundary loss function with return_residuals=True
            _, _, boundary_residuals = compute_boundary_loss(
                x_batch, t_batch, E_batch, networks, physics, return_residuals=True
            )
            
            all_residuals.append(boundary_residuals.detach())
        
        return torch.cat(all_residuals, dim=0)
    

    def _compute_initial_residuals(self, networks:Dict[str,nn.Module], physics:ElectrochemicalPhysics):
        """Compute initial condition residuals
         Args:
            networks: Dictionary of neural networks
            physics: Elcectrochemical Physics Object
        Returns:
            all initial residuals at every collocation point
        """
        
        base_set = self.base_sets['initial']
        batch_size = self.config.sampling.adaptive.residual_batch_size
        
        all_residuals = []
        
        # Process in batches
        for i in range(0, len(base_set), batch_size):
            end_idx = min(i + batch_size, len(base_set))
            
            batch_points = base_set[i:end_idx]
            x_batch = batch_points[:, 0:1].requires_grad_(True)
            t_batch = batch_points[:, 1:2].requires_grad_(True)
            E_batch = batch_points[:, 2:3]
            
            # Use existing initial loss function with return_residuals=True
            _, _, initial_residuals = compute_initial_loss(
                x_batch, t_batch, E_batch, networks, physics, return_residuals=True
            )
            
            all_residuals.append(initial_residuals.detach())
        
        return torch.cat(all_residuals, dim=0)
    

    def _compute_film_residuals(self, networks:Dict[str,nn.Module], physics:ElectrochemicalPhysics):
        """Compute film physics residuals
        Args:
            networks: Dictionary of neural networks
            physics: Elcectrochemical Physics Object
        Returns:
            all initial residuals at every collocation point
        
        """
        base_set = self.base_sets['film_physics']
        batch_size = self.config.sampling.adaptive.residual_batch_size
        
        all_residuals = []
        
        # Process in batches
        for i in range(0, len(base_set), batch_size):
            end_idx = min(i + batch_size, len(base_set))
            
            batch_points = base_set[i:end_idx]
            t_batch = batch_points[:, 0:1].requires_grad_(True)
            E_batch = batch_points[:, 1:2]
            
            # Use existing film loss function with return_residuals=True
            _, film_residuals = compute_film_physics_loss(
                t_batch, E_batch, networks, physics, return_residuals=True
            )
            
            all_residuals.append(film_residuals.detach())
        
        return torch.cat(all_residuals, dim=0)
    
    def select_all_adaptive_points(self, residuals:Dict[str,torch.Tensor]) -> None:
        """Select top-k points for all loss components
        Args:
            residuals: dictionary of torch.Tensor containing residuals for each loss at every point
        Returns:
            None
        """
        
        # Interior points (combine all PDE residuals)
        interior_combined = (torch.abs(residuals['cv_pde']) + 
                            torch.abs(residuals['av_pde']) +
                            torch.abs(residuals['h_pde']) + 
                            torch.abs(residuals['poisson_pde']))
        
        interior_indices = self._select_top_k_indices(
            interior_combined, self.adaptive_sizes['interior']
        )
        self.adaptive_sets['interior'] = self.base_sets['interior'][interior_indices].clone()
        
        # Boundary points
        boundary_combined = torch.abs(residuals['boundary'])
        boundary_indices = self._select_top_k_indices(
            boundary_combined, self.adaptive_sizes['boundary']
        )
        self.adaptive_sets['boundary'] = self.base_sets['boundary'][boundary_indices].clone()
        
        # Initial points
        initial_combined = torch.abs(residuals['initial'])
        initial_indices = self._select_top_k_indices(
            initial_combined, self.adaptive_sizes['initial']
        )
        self.adaptive_sets['initial'] = self.base_sets['initial'][initial_indices].clone()
        
        # Film physics points
        film_combined = torch.abs(residuals['film_physics'])
        film_indices = self._select_top_k_indices(
            film_combined, self.adaptive_sizes['film_physics']
        )
        self.adaptive_sets['film_physics'] = self.base_sets['film_physics'][film_indices].clone()
        
        print(f"Selected adaptive points:")
        for component, points in self.adaptive_sets.items():
            print(f"  {component}: {len(points)} points")

    def _select_top_k_indices(self, residuals:torch.Tensor, k:int)-> list[int]:
        """Select top-k indices based on residual magnitudes
        Args:
            residuals: torch.Tensor of residuals
            k: # of top k points to select
        Returns:
            indices: List of indices to be sampled 
        """
        
        if len(residuals) <= k:
            return torch.arange(len(residuals), device=self.device)
        else:
            _, indices = torch.topk(residuals, k=k, largest=True)
            return indices
        
    def update_adaptive_sampling(self, current_step, networks):
        """Main update method called from PINNTrainer"""
        
        if (current_step % self.config.sampling.adaptive.adaptive_update_freq != 0):
            return
            
        print(f"\n=== Complete Adaptive Sampling Update (Step {current_step}) ===")
        
        # Check if base sets need regeneration
        if self.should_update_base_set(current_step, networks):
            self.regenerate_all_base_sets(networks)
            self.last_update_step = current_step
        
        # Ensure base sets exist
        if any(base_set is None for base_set in self.base_sets.values()):
            print("  Initializing all base sets for first time...")
            self.regenerate_all_base_sets(networks)
            self.last_update_step = current_step
        
        # Compute residuals for all components
        residuals = self.compute_all_residuals(networks, self.physics)
        
        # Select adaptive points for all components
        self.select_all_adaptive_points(residuals)
        
        print("=== Complete Adaptive Sampling Complete ===\n")


    def get_interior_points(self):
        """Get adaptive interior points (replaces regular sampling)"""
        if self.adaptive_sets['interior'] is None:
            return None
        points = self.adaptive_sets['interior'].clone()
        return (points[:, 0:1].requires_grad_(True),  # x
                points[:, 1:2].requires_grad_(True),  # t  
                points[:, 2:3])                       # E

    def get_boundary_points(self):
        """Get adaptive boundary points (replaces regular sampling)"""
        if self.adaptive_sets['boundary'] is None:
            return None
        points = self.adaptive_sets['boundary'].clone()
        return (points[:, 0:1].requires_grad_(True),  # x
                points[:, 1:2].requires_grad_(True),  # t
                points[:, 2:3])                       # E

    def get_initial_points(self):
        """Get adaptive initial points (replaces regular sampling)"""
        if self.adaptive_sets['initial'] is None:
            return None
        points = self.adaptive_sets['initial'].clone()
        return (points[:, 0:1].requires_grad_(True),  # x
                points[:, 1:2].requires_grad_(True),  # t
                points[:, 2:3])                       # E

    def get_film_points(self):
        """Get adaptive film points (replaces regular sampling)"""
        if self.adaptive_sets['film_physics'] is None:
            return None
        points = self.adaptive_sets['film_physics'].clone()
        return (points[:, 0:1].requires_grad_(True),  # t
                points[:, 1:2])                       # E

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
