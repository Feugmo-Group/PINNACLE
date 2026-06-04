# sampling/sampling.py
"""
Simple collocation point sampling for PINNACLE.

This module provides the basic sampling functions needed for PINN training.
"""
import torch
from typing import Dict, Any, Tuple, Optional
from physics.physics import ElectrochemicalPhysics
from losses.losses import compute_boundary_residuals_for_adaptive
from losses.losses import compute_initial_residuals_for_adaptive
from losses.losses import compute_film_physics_loss
import torch.nn as nn
import numpy as np
from pathlib import Path
import pandas as pd

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
        #Hybrid sampling ratios
        self.uniform_ratio = config.sampling.adaptive.get('uniform_ratio', 0.6)  # Default 60% uniform
        self.adaptive_ratio = 1.0 - self.uniform_ratio

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
        print(f"  Interior base set size: {len(self.base_sets['interior'])}")
        print(f"  Boundary base set size: {len(self.base_sets['boundary'])}")
        print(f"initial base set size: {len(self.base_sets['initial'])}")
        print(f"film_phyiscs base set size: {len(self.base_sets['film_physics'])}")
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
        #TODO: Should hafe one E-value for give (x,t) pair and need to fix this, is it even fixable? 
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
        filtered_tensor = torch.cat(valid_points, dim=0)
        print(f"  Valid interior points after filtering: {len(filtered_tensor)} / {len(x_flat)}")
        return filtered_tensor
    
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
            boundary_residuals = compute_boundary_residuals_for_adaptive(
                x_batch, t_batch, E_batch, networks, physics
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
            initial_residuals = compute_initial_residuals_for_adaptive(
                x_batch, t_batch, E_batch, networks, physics
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
        # Interior points
        interior_combined = (torch.abs(residuals['cv_pde']) + 
                            torch.abs(residuals['av_pde']) +
                            torch.abs(residuals['h_pde']) + 
                            torch.abs(residuals['poisson_pde'])).flatten()
        
        if len(interior_combined) != len(self.base_sets['interior']):
            print(f"ERROR: Interior mismatch - residuals:{len(interior_combined)} vs base:{len(self.base_sets['interior'])}")
            return
        
        interior_indices = self._select_hybrid_points(interior_combined, self.adaptive_sizes['interior'])
        if len(interior_indices) > 0 and torch.max(interior_indices).item() >= len(self.base_sets['interior']):
            print(f"ERROR: Interior index {torch.max(interior_indices).item()} >= {len(self.base_sets['interior'])}")
            return
        self.adaptive_sets['interior'] = self.base_sets['interior'][interior_indices].clone()
        
        # Boundary points
        boundary_combined = torch.abs(residuals['boundary']).flatten()
        if len(boundary_combined) != len(self.base_sets['boundary']):
            print(f"ERROR: Boundary mismatch - residuals:{len(boundary_combined)} vs base:{len(self.base_sets['boundary'])}")
            return
        boundary_indices = self._select_hybrid_points(boundary_combined, self.adaptive_sizes['boundary'])
        if len(boundary_indices) > 0 and torch.max(boundary_indices).item() >= len(self.base_sets['boundary']):
            print(f"ERROR: Boundary index out of bounds")
            return
        self.adaptive_sets['boundary'] = self.base_sets['boundary'][boundary_indices].clone()
        
        # Initial points  
        initial_combined = torch.abs(residuals['initial']).flatten()
        if len(initial_combined) != len(self.base_sets['initial']):
            print(f"ERROR: Initial mismatch - residuals:{len(initial_combined)} vs base:{len(self.base_sets['initial'])}")
            return
        initial_indices = self._select_hybrid_points(initial_combined, self.adaptive_sizes['initial'])
        if len(initial_indices) > 0 and torch.max(initial_indices).item() >= len(self.base_sets['initial']):
            print(f"ERROR: Initial index out of bounds")
            return
        self.adaptive_sets['initial'] = self.base_sets['initial'][initial_indices].clone()
        
        # Film physics points
        film_combined = torch.abs(residuals['film_physics']).flatten()
        if len(film_combined) != len(self.base_sets['film_physics']):
            print(f"ERROR: Film mismatch - residuals:{len(film_combined)} vs base:{len(self.base_sets['film_physics'])}")
            return
        film_indices = self._select_hybrid_points(film_combined, self.adaptive_sizes['film_physics'])
        if len(film_indices) > 0 and torch.max(film_indices).item() >= len(self.base_sets['film_physics']):
            print(f"ERROR: Film index out of bounds")
            return
        self.adaptive_sets['film_physics'] = self.base_sets['film_physics'][film_indices].clone()
        
        print(f"Adaptive selection OK")
        
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
        
    def _select_hybrid_points(self, residuals:torch.Tensor, target_count:int) -> list[int]:
        """
        Select hybrid mix of uniform and adaptive points
        
        Args:
            residuals: Residual magnitudes for all base points
            target_count: Total number of points to select
            
        Returns:
            indices: Selected point indices
        """
        total_available = len(residuals)

        if total_available <= target_count:
            # If we don't have enough points, return all
            return torch.arange(total_available, device=self.device)
        
        #Calculate uniform to adaptive split
        n_uniform = int(target_count * self.uniform_ratio)
        n_adaptive = target_count - n_uniform

        uniform_indices = torch.randperm(total_available, device=self.device)[:n_uniform]

        if n_adaptive > 0:
            _, adaptive_indices = torch.topk(residuals, k=n_adaptive, largest=True)
            
            # Combine indices
            combined_indices = torch.cat([uniform_indices, adaptive_indices])
        else:
            combined_indices = uniform_indices
        
        return combined_indices


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

def _resolve_dtype(precision: str) -> torch.dtype:
    return torch.float64 if precision == "float64" else torch.float32


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
        self.time_scale = physics.domain.time_scale
        self.tc = physics.scales.tc
        # Store batch sizes for easy access
        self.batch_sizes = config["batch_size"]

        # dtype (respects precision config)
        self.dtype = _resolve_dtype(config.get("precision", "float32"))

        # Sobol quasi-random engines — one per sampling context so sequences are independent
        self.use_sobol = config.sampling.get("use_sobol", False)
        if self.use_sobol:
            self._sobol_interior = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
            self._sobol_boundary = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            self._sobol_initial  = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            self._sobol_film     = torch.quasirandom.SobolEngine(dimension=2, scramble=True)

    # ── low-level sampling helpers ────────────────────────────────────────────

    def _draw(self, engine, n: int, d: int) -> torch.Tensor:
        """Draw n unit samples in [0, 1)^d using Sobol engine or pseudo-random."""
        if engine is not None:
            return engine.draw(n).to(device=self.device, dtype=self.dtype)
        return torch.rand(n, d, device=self.device, dtype=self.dtype)

    def _scale_E(self, E_unit: torch.Tensor) -> torch.Tensor:
        """Scale unit [0,1) E samples to the dimensional E range (volts).
        Matches the original sampling convention used throughout training."""
        E_min = self.physics.geometry.E_min
        E_max = self.physics.geometry.E_max
        return E_min + E_unit * (E_max - E_min)

    # ── public sampling methods ───────────────────────────────────────────────

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
        t_scale = self.time_scale / self.tc

        engine = self._sobol_interior if self.use_sobol else None
        raw = self._draw(engine, batch_size, 3)  # (N, 3): [t_unit, E_unit, x_unit]

        t = (raw[:, 0:1] * t_scale).detach().requires_grad_(True)
        E = self._scale_E(raw[:, 1:2]).detach()

        # Get film thickness prediction
        L_pred = networks["film_thickness"](torch.cat([t, E], dim=1))

        # Sample spatial coordinates within [0, L(t,E)]
        x = (raw[:, 2:3].detach() * L_pred.detach()).requires_grad_(True)

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

        t_scale = self.time_scale / self.tc

        engine = self._sobol_boundary if self.use_sobol else None
        raw = self._draw(engine, batch_size, 2)  # (2N, 2): [t_unit, E_unit]

        t = (raw[:, 0:1] * t_scale).detach().requires_grad_(True)
        E = self._scale_E(raw[:, 1:2]).detach()

        # Predict L for f/s boundary
        L_pred = networks["film_thickness"](torch.cat([t, E], dim=1))

        half = batch_size // 2

        # Metal/film interface points (x = 0)
        x_mf = torch.zeros(half, 1, device=self.device, dtype=self.dtype).requires_grad_(True)
        # Film/solution interface points (x = L)
        x_fs = L_pred[half:].detach().requires_grad_(True)

        x_boundary = torch.cat([x_mf, x_fs], dim=0)
        t_boundary = torch.cat([t[:half], t[half:]], dim=0)
        E_boundary = torch.cat([E[:half], E[half:]], dim=0)

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

        engine = self._sobol_initial if self.use_sobol else None
        raw = self._draw(engine, batch_size, 2)  # (N, 2): [E_unit, x_unit]

        # Initial time (t = 0)
        t = torch.zeros(batch_size, 1, device=self.device, dtype=self.dtype).requires_grad_(True)
        E = self._scale_E(raw[:, 0:1]).detach()

        # Get initial film thickness
        L_initial_pred = networks["film_thickness"](torch.cat([t, E], dim=1))

        x = (raw[:, 1:2].detach() * L_initial_pred.detach()).requires_grad_(True)

        return x, t, E

    def sample_film_physics_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points for film growth physics constraint.

        Returns:
            Tuple of (t, E) tensors for film physics constraint
        """
        batch_size = self.batch_sizes["L"]
        t_scale = self.time_scale / self.tc

        engine = self._sobol_film if self.use_sobol else None
        raw = self._draw(engine, batch_size, 2)  # (N, 2): [t_unit, E_unit]

        t = (raw[:, 0:1] * t_scale).detach().requires_grad_(True)
        E = self._scale_E(raw[:, 1:2]).detach()

        return t, E

    def _load_all_fem_data(self):
        """Load all FEM txt files and combine"""
        all_data = {'t': [], 'E': [], 'L': []}

        fem_dir = Path(self.config.hybrid.fem_data_dir)
        if not fem_dir.is_dir():
            print(f"Warning: FEM data directory '{fem_dir}' not found — disabling hybrid FEM training.")
            self.config.hybrid.use_data = False
            return None
        # Sorted glob: filesystem iteration order is OS-/inode-dependent, so an
        # unsorted glob silently changes which row torch.randperm(N, seed)
        # picks. Sorting alphabetically makes the FEM concat order
        # reproducible across machines and remounts (paper revision E3).
        txt_files = sorted(fem_dir.glob("*.txt"))
        if not txt_files:
            print(f"Warning: No .txt files found in '{fem_dir}' — disabling hybrid FEM training.")
            self.config.hybrid.use_data = False
            return None

        for txt_file in txt_files:
            # Extract voltage from filename like "0.1 V.txt"
            voltage = float(txt_file.stem.split()[0])
            
            # Load tab-separated data
            df = pd.read_csv(txt_file, sep='\t')
            
            t_vals = df['Time/s'].values
            E_vals = np.full(len(t_vals), voltage) # Constant voltage per file
            L_vals = df['Filmthickness/m'].values

            valid_mask = ~(np.isnan(t_vals) | np.isnan(E_vals) | np.isnan(L_vals))
        
            if np.any(~valid_mask):
                n_invalid = np.sum(~valid_mask)
                print(f"Warning: Filtered {n_invalid} NaN values from {txt_file.name}")

            t_vals = t_vals[valid_mask]
            E_vals = E_vals[valid_mask] 
            L_vals = L_vals[valid_mask]
            
            all_data['t'].append(t_vals)
            all_data['E'].append(E_vals)
            all_data['L'].append(L_vals)
        
        # Concatenate and non-dimensionalize
        t_all = np.concatenate(all_data['t']) / self.physics.scales.tc
        E_all = np.concatenate(all_data['E']) / self.physics.scales.phic  
        L_all = np.concatenate(all_data['L']) / self.physics.scales.lc
        
        return {
            't': torch.tensor(t_all, dtype=torch.get_default_dtype(), device=self.device),
            'E': torch.tensor(E_all, dtype=torch.get_default_dtype(), device=self.device),
            'L': torch.tensor(L_all, dtype=torch.get_default_dtype(), device=self.device)
        }

    # Default anchor used to reproduce the published predictions_overview.
    # NB: this is the *empirical* anchor that reproduces the original figures
    # — (t = 76 000 s, E = 0.4 V, L = 3.64 nm) — and differs from the value
    # historically cited in the manuscript text. The discrepancy was traced
    # during the paper revision: see Sec. IV.D ("Anchor sensitivity").
    _DEFAULT_ANCHOR_T_PHYS = 76_000.0  # seconds
    _DEFAULT_ANCHOR_E_PHYS = 0.4       # volts

    def sample_fem_data(self, n_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample anchor points from pre-loaded FEM data.

        Anchor-selection mode is controlled by ``hybrid.anchor_mode``:

        - ``"default"`` (default): snap to the empirical anchor that
          reproduces the published figures, (t = 76 000 s, E = 0.4 V,
          L = 3.64 nm). ``"paper"`` is accepted as a deprecated alias.
          ``anchor_seed`` is ignored.
        - ``"seed"``: random selection seeded by ``hybrid.anchor_seed``;
          used by E3-A (30-seed sweep) and E4 (data-count sweep).
        - ``"position"``: nearest-FEM-point lookup to
          (``hybrid.anchor_t``, ``hybrid.anchor_E``); used by E3-B
          (systematic position sweep).

        Other controls:

        - ``hybrid.n_data_points``: how many anchors to draw (E4).
        - ``hybrid.noise_sigma``: Gaussian multiplicative noise on L (E5).

        Returns ``None`` if the FEM data directory is missing.
        """
        if not hasattr(self, 'fem_data'):
            self.fem_data = self._load_all_fem_data()
        if self.fem_data is None:
            return None

        hybrid = getattr(self.config, 'hybrid', {}) if hasattr(self.config, 'hybrid') else self.config.get('hybrid', {})
        # Anchor count: explicit n_samples wins, then n_data_points, then fem_batch_size.
        if n_samples is None:
            n_samples = int(getattr(hybrid, 'n_data_points', None) or hybrid.get('n_data_points')
                            or hybrid.get('fem_batch_size', 1))

        total_points = len(self.fem_data['t'])
        anchor_mode = str(hybrid.get('anchor_mode', 'default')).lower()

        if anchor_mode in ('default', 'paper'):
            # Snap to the empirical anchor that reproduces the published
            # figures (t = 76 000 s, E = 0.4 V).
            t_target_nd = self._DEFAULT_ANCHOR_T_PHYS / float(self.physics.scales.tc)
            E_target_nd = self._DEFAULT_ANCHOR_E_PHYS / float(self.physics.scales.phic)
            t_diff = (self.fem_data['t'] - t_target_nd).abs()
            E_diff = (self.fem_data['E'] - E_target_nd).abs()
            t_scale = (self.fem_data['t'].max() - self.fem_data['t'].min()).clamp(min=1e-12)
            E_scale = (self.fem_data['E'].max() - self.fem_data['E'].min()).clamp(min=1e-12)
            score = t_diff / t_scale + E_diff / E_scale
            indices = torch.argsort(score)[:n_samples]
        elif anchor_mode == 'position':
            # Systematic E3-B: choose the FEM points closest to (anchor_t, anchor_E).
            anchor_t = hybrid.get('anchor_t', None)
            anchor_E = hybrid.get('anchor_E', None)
            if anchor_t is None or anchor_E is None:
                raise ValueError(
                    "anchor_mode='position' requires hybrid.anchor_t and hybrid.anchor_E to be set."
                )
            t_target_nd = float(anchor_t) / float(self.physics.scales.tc)
            E_target_nd = float(anchor_E) / float(self.physics.scales.phic)
            t_diff = (self.fem_data['t'] - t_target_nd).abs()
            E_diff = (self.fem_data['E'] - E_target_nd).abs()
            t_scale = (self.fem_data['t'].max() - self.fem_data['t'].min()).clamp(min=1e-12)
            E_scale = (self.fem_data['E'].max() - self.fem_data['E'].min()).clamp(min=1e-12)
            score = t_diff / t_scale + E_diff / E_scale
            indices = torch.argsort(score)[:n_samples]
        elif anchor_mode == 'seed':
            # E3-A / E4 random sampling, seeded for reproducibility.
            seed = int(hybrid.get('anchor_seed', 0))
            g = torch.Generator(device=self.device).manual_seed(seed)
            indices = torch.randperm(total_points, generator=g, device=self.device)[:n_samples]
        elif anchor_mode == 'random':
            # Original PINNACLE behaviour: draw n_samples random FEM points
            # each call (no fixed seed). Provides dense coverage of the full
            # (t, E) domain over 50k steps and reproduces the published errors.
            indices = torch.randperm(total_points, device=self.device)[:n_samples]
        else:
            raise ValueError(
                f"Unknown hybrid.anchor_mode={anchor_mode!r}. "
                "Use 'default' (= 'paper' alias), 'random', 'seed', or 'position'."
            )

        t_sel = self.fem_data['t'][indices]
        E_sel = self.fem_data['E'][indices]
        L_sel = self.fem_data['L'][indices]

        # E5 noise: multiplicative Gaussian on L, seeded for reproducibility.
        sigma = float(hybrid.get('noise_sigma', 0.0))
        if sigma > 0.0:
            seed = int(hybrid.get('anchor_seed', 0))
            g = torch.Generator(device=self.device).manual_seed(seed + 1)  # decorrelate from anchor draw
            noise = torch.randn(L_sel.shape, generator=g, device=self.device) * sigma
            L_sel = L_sel * (1.0 + noise)

        fem_data = {"t": t_sel, "E": E_sel, "L": L_sel}
        self.last_fem_data = fem_data
        return fem_data

    def sample_inverse_observations(self) -> Optional[Dict[str, torch.Tensor]]:
        """Sample fixed observation set for the inverse-problem experiment.

        Reads ``config.inverse``:
          - ``obs_voltages``        : list of voltages at which to draw L(t)
          - ``n_obs``               : number of observations per voltage
          - ``obs_noise_sigma``     : Gaussian multiplicative noise on L

        The observations are drawn deterministically (evenly spaced in time
        per voltage) so the inverse experiment is reproducible across runs.
        Noise is seeded from ``config.seed`` if present, else from a fixed
        seed. Returns ``None`` if the FEM data directory is missing.

        Returns
        -------
        Optional[Dict[str, torch.Tensor]]
            Dictionary with keys ``t``, ``E``, ``L`` (each shape ``[N]``),
            or ``None`` if FEM data is unavailable.
        """
        if not hasattr(self, 'fem_data'):
            self.fem_data = self._load_all_fem_data()
        if self.fem_data is None:
            return None

        inv_cfg = getattr(self.config, 'inverse', {}) if hasattr(self.config, 'inverse') else self.config.get('inverse', {})
        obs_voltages = list(inv_cfg.get('obs_voltages', [0.1]))
        n_obs = int(inv_cfg.get('n_obs', 5))
        sigma = float(inv_cfg.get('obs_noise_sigma', 0.0))

        # FEM data is concatenated across voltages; select rows whose E is
        # within a small tolerance of each requested voltage.
        t_full = self.fem_data['t']
        E_full = self.fem_data['E']
        L_full = self.fem_data['L']

        # obs_voltages are configured in physical volts; FEM data is stored
        # in non-dimensional units (E_full = E_phys / phic). Convert before
        # matching.
        phic = float(self.physics.scales.phic)
        t_obs_list, E_obs_list, L_obs_list = [], [], []
        for v in obs_voltages:
            v_nd = float(v) / phic
            mask = (E_full - v_nd).abs() < 1e-3
            n_match = int(mask.sum().item())
            if n_match == 0:
                # If the exact voltage is not in the FEM table, snap to the closest.
                v_actual = float(E_full[(E_full - v_nd).abs().argmin()].item())
                mask = (E_full - v_actual).abs() < 1e-3
                n_match = int(mask.sum().item())
            idx_v = mask.nonzero(as_tuple=False).squeeze(-1)
            t_v = t_full[idx_v]
            # Evenly spaced selection across this voltage's time domain.
            sort_idx = torch.argsort(t_v)
            idx_v_sorted = idx_v[sort_idx]
            if n_obs >= n_match:
                pick = idx_v_sorted
            else:
                step = max(1, n_match // n_obs)
                pick = idx_v_sorted[::step][:n_obs]
            t_obs_list.append(t_full[pick])
            E_obs_list.append(E_full[pick])
            L_obs_list.append(L_full[pick])

        t_obs = torch.cat(t_obs_list).view(-1, 1)
        E_obs = torch.cat(E_obs_list).view(-1, 1)
        L_obs = torch.cat(L_obs_list).view(-1, 1)

        if sigma > 0.0:
            # Seed independently from the anchor draw so noise realisations
            # are reproducible but uncorrelated with E3/E4/E5 anchor noise.
            seed = int(inv_cfg.get('seed', 0))
            g = torch.Generator(device=self.device).manual_seed(seed + 7919)
            noise = torch.randn(L_obs.shape, generator=g, device=self.device) * sigma
            L_obs = L_obs * (1.0 + noise)

        return {"t": t_obs, "E": E_obs, "L": L_obs}

        