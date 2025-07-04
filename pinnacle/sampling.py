# pinnacle/sampling.py
"""
Simple collocation point sampling for PINNACLE.

This module provides the basic sampling functions needed for PINN training.
"""
import torch
from typing import Dict, Any, Tuple


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
        self.batch_sizes = config['batch_size']

    def sample_interior_points(self, networks) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample interior collocation points for PDE residuals.

        Args:
            networks: NetworkManager instance

        Returns:
            Tuple of (x, t, E) tensors with requires_grad=True for x and t
        """
        batch_size = self.batch_sizes['interior']

        # Sample time and applied potential
        t = torch.rand(batch_size, 1, device=self.device, requires_grad=True)
        single_E = torch.rand(1, 1, device=self.device) * (
                    self.physics.geometry.E_max - self.physics.geometry.E_min) + self.physics.geometry.E_min
        E = single_E.expand(batch_size, 1)

        # Get film thickness prediction
        L_pred = networks['film_thickness'](torch.cat([t, E], dim=1))

        # Sample spatial coordinates within [0, L(t,E)]
        x = torch.rand(batch_size, 1, device=self.device, requires_grad=True) * L_pred

        return x, t, E

    def sample_boundary_points(self, networks) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample boundary collocation points for boundary conditions.

        Args:
            networks: NetworkManager instance

        Returns:
            Tuple of (x, t, E) tensors for boundary points
        """
        batch_size = self.batch_sizes['BC']

        # Sample time and applied potential
        t = torch.rand(batch_size, 1, device=self.device, requires_grad=True)
        single_E = torch.rand(1, 1, device=self.device) * (
                    self.physics.geometry.E_max - self.physics.geometry.E_min) + self.physics.geometry.E_min
        E = single_E.expand(batch_size, 1)

        # Predict L for f/s boundary
        L_inputs = torch.cat([t, E], dim=1)
        L_pred = networks['film_thickness'](L_inputs)

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

    def sample_initial_points(self, networks) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample initial condition points at t = 0.

        Args:
            networks: NetworkManager instance

        Returns:
            Tuple of (x, t, E) tensors for initial condition points
        """
        batch_size = self.batch_sizes['IC']

        # Initial time (t = 0)
        t = torch.zeros(batch_size, 1, device=self.device, requires_grad=True)
        single_E = torch.rand(1, 1, device=self.device) * (
                    self.physics.geometry.E_max - self.physics.geometry.E_min) + self.physics.geometry.E_min
        E = single_E.expand(batch_size, 1)

        # Get initial film thickness
        L_initial_pred = networks['film_thickness'](torch.cat([t, E], dim=1))

        # Sample spatial coordinates
        x = torch.rand(batch_size, 1, device=self.device, requires_grad=True) * L_initial_pred

        return x, t, E

    def sample_film_physics_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points for film growth physics constraint.

        Returns:
            Tuple of (t, E) tensors for film physics constraint
        """
        batch_size = self.batch_sizes['L']

        # Sample time and applied potential
        t = torch.rand(batch_size, 1, device=self.device, requires_grad=True)
        single_E = torch.rand(1, 1, device=self.device) * (
                    self.physics.geometry.E_max - self.physics.geometry.E_min) + self.physics.geometry.E_min
        E = single_E.expand(batch_size, 1)

        return t, E