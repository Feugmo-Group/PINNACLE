# gradients/gradients.py
"""
Gradient computation utilities for Physics-Informed Neural Networks.

This module provides efficient and reusable gradient computation tools
that can be used across different PINN applications.
"""

import torch
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
import warnings


class GradientResults(NamedTuple):
    """
    Container for gradient computation results.

    Organizes all computed derivatives in a structured way for easy access.
    """
    # Network predictions
    phi: torch.Tensor  # Potential φ
    c_cv: torch.Tensor  # Cation vacancy concentration
    c_av: torch.Tensor  # Anion vacancy concentration
    c_h: torch.Tensor  # Hole concentration

    # Time derivatives
    c_cv_t: torch.Tensor  # ∂c_cv/∂t
    c_av_t: torch.Tensor  # ∂c_av/∂t
    c_h_t: torch.Tensor  # ∂c_h/∂t

    # First spatial derivatives
    phi_x: torch.Tensor  # ∂φ/∂x
    c_cv_x: torch.Tensor  # ∂c_cv/∂x
    c_av_x: torch.Tensor  # ∂c_av/∂x
    c_h_x: torch.Tensor  # ∂c_h/∂x

    # Second spatial derivatives
    phi_xx: torch.Tensor  # ∂²φ/∂x²
    c_cv_xx: torch.Tensor  # ∂²c_cv/∂x²
    c_av_xx: torch.Tensor  # ∂²c_av/∂x²
    c_h_xx: torch.Tensor  # ∂²c_h/∂x²


@dataclass
class GradientConfig:
    """Configuration for gradient computations"""
    create_graph: bool = True  # For higher-order derivatives
    retain_graph: bool = True  # Keep computation graph
    validate_inputs: bool = True  # Check input tensor properties


class GradientComputer:
    """
    Efficient gradient computation for Physics-Informed Neural Networks.

    This class provides methods to compute gradients needed for PDE residuals
    in a clean, reusable way that works with any network architecture.
    """

    def __init__(self, config: Optional[GradientConfig] = None, device: Optional[torch.device] = None):
        """
        Initialize gradient computer.

        Args:
            config: Configuration for gradient computations
            device: PyTorch device for computations
        """
        self.config = config or GradientConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_derivative(
            self,
            output: torch.Tensor,
            input_var: torch.Tensor,
            order: int = 1,
            create_graph: Optional[bool] = None,
            retain_graph: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Compute derivative of output with respect to input_var.

        Args:
            output: Network output tensor
            input_var: Input variable to differentiate with respect to
            order: Order of derivative (1 or 2)
            create_graph: Override config setting for create_graph
            retain_graph: Override config setting for retain_graph

        Returns:
            Derivative tensor
        """

        create_graph = create_graph if create_graph is not None else self.config.create_graph
        retain_graph = retain_graph if retain_graph is not None else self.config.retain_graph

        if order == 1:
            return self._first_derivative(output, input_var, create_graph, retain_graph)
        elif order == 2:
            # Compute first derivative, then take derivative again
            first_deriv = self._first_derivative(output, input_var, create_graph=True, retain_graph=True)
            return self._first_derivative(first_deriv, input_var, create_graph, retain_graph)
        else:
            raise ValueError(f"Derivative order {order} not supported. Use 1 or 2.")

    def _first_derivative(
            self,
            output: torch.Tensor,
            input_var: torch.Tensor,
            create_graph: bool,
            retain_graph: bool
    ) -> torch.Tensor:
        """Compute first-order derivative using autograd"""
        grad_outputs = torch.ones_like(output)

        try:
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=input_var,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]

            return gradients

        except RuntimeError as e:
            if "does not require grad" in str(e):
                raise ValueError(
                    f"Input tensor must have requires_grad=True. Got requires_grad={input_var.requires_grad}")
            else:
                raise e

    def compute_electrochemistry_gradients(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            E: torch.Tensor,
            networks: Dict[str, torch.nn.Module]
    ) -> GradientResults:
        """
        Compute all gradients needed for electrochemistry PINNs.

        Args:
            x: Spatial coordinates (requires_grad=True)
            t: Time coordinates (requires_grad=True)
            E: Applied potential
            networks: Dictionary of networks or NetworkManager instance

        Returns:
            GradientResults with all computed derivatives
        """
        # Forward pass through networks
        inputs_3d = torch.cat([x, t, E], dim=1)

        # Get network predictions
        phi = networks['potential'](inputs_3d)
        c_cv_raw = networks['cv'](inputs_3d)
        c_av_raw = networks['av'](inputs_3d)
        c_h_raw = networks['h'](inputs_3d)


        # Networks predict concentrations directly
        c_cv = c_cv_raw
        c_av = c_av_raw
        c_h = c_h_raw


        # Direct derivatives
        c_cv_t = self.compute_derivative(c_cv, t)
        c_av_t = self.compute_derivative(c_av, t)
        c_h_t = self.compute_derivative(c_h, t)

        # Compute first spatial derivatives
        phi_x = self.compute_derivative(phi, x)

        c_cv_x = self.compute_derivative(c_cv, x)
        c_av_x = self.compute_derivative(c_av, x)
        c_h_x = self.compute_derivative(c_h, x)

        # Compute second spatial derivatives
        phi_xx = self.compute_derivative(phi_x, x)

        c_cv_xx = self.compute_derivative(c_cv_x, x)
        c_av_xx = self.compute_derivative(c_av_x, x)
        c_h_xx = self.compute_derivative(c_h_x, x)

        return GradientResults(
            phi=phi, c_cv=c_cv, c_av=c_av, c_h=c_h,
            c_cv_t=c_cv_t, c_av_t=c_av_t, c_h_t=c_h_t,
            phi_x=phi_x, c_cv_x=c_cv_x, c_av_x=c_av_x, c_h_x=c_h_x,
            phi_xx=phi_xx, c_cv_xx=c_cv_xx, c_av_xx=c_av_xx, c_h_xx=c_h_xx
        )


    
