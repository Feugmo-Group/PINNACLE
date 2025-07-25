# losses/losses.py
"""
Loss function calculations for PINNACLE with optional residual extraction for NTK.

Simple functional approach to computing different loss components
for physics-informed neural network training, with the ability to return
raw residuals for NTK weight computation.
"""

import torch
from typing import Dict, Tuple, Any, Union, Optional
torch.manual_seed(995) 

def compute_interior_loss(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                          networks, physics,
                          return_residuals: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """
    Compute interior PDE residual losses.

    See compute_pde_residuals for mathematics of residual calculations

    Args:
        x: Spatial coordinates
        t: Time coordinates
        E: Applied potential
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        return_residuals: If True, also return raw residuals for NTK computation

    Returns:
        If return_residuals=False: Tuple of (total_interior_loss, individual_losses_dict)
        If return_residuals=True: Tuple of (total_interior_loss, individual_losses_dict, residuals_dict)
    """
    # Compute PDE residuals using physics module
    cv_residual, av_residual, h_residual, poisson_residual = physics.compute_pde_residuals(x, t, E, networks)

    # Calculate individual losses
    cv_pde_loss = torch.mean(cv_residual ** 2)
    av_pde_loss = torch.mean(av_residual ** 2)
    if physics.config.pde.physics.include_holes:
        h_pde_loss = torch.mean(h_residual ** 2)
    else:
        h_pde_loss = torch.mean(torch.zeros_like(h_residual))
    poisson_pde_loss = torch.mean(poisson_residual ** 2)

    # Total interior loss
    total_interior_loss = cv_pde_loss + av_pde_loss + h_pde_loss + poisson_pde_loss

    individual_losses = {
        'cv_pde': cv_pde_loss,
        'av_pde': av_pde_loss,
        'h_pde': h_pde_loss,
        'poisson_pde': poisson_pde_loss
    }

    if return_residuals:
        residuals = {
            'cv_pde': cv_residual,
            'av_pde': av_residual,
            'h_pde': h_residual,
            'poisson_pde': poisson_residual
        }
        return total_interior_loss, individual_losses, residuals
    else:
        return total_interior_loss, individual_losses


def compute_boundary_loss(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                          networks, physics,
                          return_residuals: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
    """
    Compute boundary condition losses.
     **Boundary Conditions:**

    **Metal/Film Interface (x̂ = 0):**

    *Cation Vacancy Flux:*

    .. math::
        -D_{cv}\\frac{\\partial \\hat{c}_{cv}}{\\partial \\hat{x}} = \\hat{k}_1 - \\left(U_{cv}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}} - \\frac{d\\hat{L}}{d\\hat{t}}\\right)\\hat{c}_{cv}

    *Anion Vacancy Flux:*

    .. math::
        -D_{av}\\frac{\\partial \\hat{c}_{av}}{\\partial \\hat{x}} = \\frac{4}{3}\\hat{k}_2 + \\left(U_{av}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}} - \\frac{d\\hat{L}}{d\\hat{t}}\\right)\\hat{c}_{av}

    *Potential Boundary:*

    .. math::
        \\varepsilon_f \\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}} = \\frac{\\varepsilon_{Ddl}(\\hat{\\phi} - \\hat{E})}{\\hat{d}_{Ddl}}

    **Film/Solution Interface (x̂ = L̂):**

    *Cation Vacancy Flux:*

    .. math::
        -D_{cv}\\frac{\\partial \\hat{c}_{cv}}{\\partial \\hat{x}} = \\left(\\hat{k}_3 - U_{cv}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}}\\right)\\hat{c}_{cv}

    *Anion Vacancy Flux:*

    .. math::
        -D_{av}\\frac{\\partial \\hat{c}_{av}}{\\partial \\hat{x}} = \\left(\\hat{k}_4 - U_{av}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}}\\right)\\hat{c}_{av}

    *Hole Flux:*

    .. math::
        D_h\\frac{\\partial \\hat{c}_h}{\\partial \\hat{x}} = \\hat{q}\\hat{c}_h

    where :math:`\\hat{q} = -(\\hat{k}_{tp} + \\frac{FD_h}{RT}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}})` for :math:`\\hat{c}_h > 10^{-9}`

    *Potential Boundary:*

    .. math::
        \\varepsilon_f \\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}} = \\varepsilon_{Ddl}\\hat{\\phi}
    Args:
        x: Boundary spatial coordinates
        t: Time coordinates
        E: Applied potential
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        return_residuals: If True, also return raw residuals for NTK computation

    Returns:
        If return_residuals=False: Tuple of (total_boundary_loss, individual_losses_dict)
        If return_residuals=True: Tuple of (total_boundary_loss, individual_losses_dict, combined_residuals)
    """
    batch_size = x.shape[0]
    half_batch = batch_size // 2

    # Split into metal/film and film/solution interfaces
    x_mf = x[:half_batch]
    x_fs = x[half_batch:]
    t_mf = t[:half_batch]
    t_fs = t[half_batch:]
    E_mf = E[:half_batch]
    E_fs = E[half_batch:]

    # Predict L and compute derivative for boundary fluxes
    L_input = torch.cat([t, E], dim=1)
    L_pred = networks['film_thickness'](L_input)
    L_pred_t = physics.grad_computer.compute_derivative(L_pred, t)
    L_pred_t_mf = L_pred_t[:half_batch]

    # Metal/film interface conditions
    inputs_mf = torch.cat([x_mf, t_mf, E_mf], dim=1)
    u_pred_mf = networks['potential'](inputs_mf)
    u_pred_mf_x = physics.grad_computer.compute_derivative(u_pred_mf, x_mf)

    # CV at m/f interface
    cv_pred_mf = networks['cv'](inputs_mf)
    cv_pred_mf_x = physics.grad_computer.compute_derivative(cv_pred_mf, x_mf)
    cv_mf_residual = ((-physics.transport.D_cv * physics.scales.cc / physics.scales.lc) * cv_pred_mf_x -
                      physics.kinetics.k1_0 * torch.exp(
                physics.kinetics.alpha_cv * physics.scales.phic * (E_mf / physics.scales.phic - u_pred_mf)) -
                      (physics.transport.U_cv * physics.scales.phic / physics.scales.lc * u_pred_mf_x -
                       physics.scales.lc / physics.scales.tc * L_pred_t_mf) * physics.scales.cc * cv_pred_mf)
    cv_mf_loss = torch.mean(cv_mf_residual ** 2)

    # AV at m/f interface
    av_pred_mf = networks['av'](inputs_mf)
    av_pred_mf_x = physics.grad_computer.compute_derivative(av_pred_mf, x_mf)
    av_mf_residual = ((-physics.transport.D_av * physics.scales.cc / physics.scales.lc) * av_pred_mf_x -
                      (4 / 3) * physics.kinetics.k2_0 * torch.exp(
                physics.kinetics.alpha_av * physics.scales.phic * (E_mf / physics.scales.phic - u_pred_mf)) -
                      (physics.transport.U_av * physics.scales.phic / physics.scales.lc * u_pred_mf_x -
                       physics.scales.lc / physics.scales.tc * L_pred_t_mf) * av_pred_mf)
    av_mf_loss = torch.mean(av_mf_residual ** 2)

    # Potential at m/f interface
    u_mf_residual = ((physics.materials.eps_film * physics.scales.phic / physics.scales.lc * u_pred_mf_x) -
                     physics.materials.eps_Ddl * physics.scales.phic * (
                             u_pred_mf - E_mf / physics.scales.phic) / physics.geometry.d_Ddl)
    u_mf_loss = torch.mean(u_mf_residual ** 2)

    # Film/solution interface conditions
    inputs_fs = torch.cat([x_fs, t_fs, E_fs], dim=1)
    u_pred_fs = networks['potential'](inputs_fs)
    u_pred_fs_x = physics.grad_computer.compute_derivative(u_pred_fs, x_fs)

    # CV at f/s interface
    cv_pred_fs = networks['cv'](inputs_fs)
    cv_pred_fs_x = physics.grad_computer.compute_derivative(cv_pred_fs, x_fs)
    cv_fs_residual = ((-physics.transport.D_cv * physics.scales.cc / physics.scales.lc) * cv_pred_fs_x -
                      (physics.kinetics.k3_0 * torch.exp(physics.kinetics.beta_cv * physics.scales.phic * u_pred_fs) -
                       physics.transport.U_cv * physics.scales.phic / physics.scales.lc * u_pred_fs_x) * cv_pred_fs * physics.scales.cc)
    cv_fs_loss = torch.mean(cv_fs_residual ** 2)

    # AV at f/s interface
    av_pred_fs = networks['av'](inputs_fs)
    av_pred_fs_x = physics.grad_computer.compute_derivative(av_pred_fs, x_fs)
    av_fs_residual = ((-physics.transport.D_av * physics.scales.cc / physics.scales.lc) * av_pred_fs_x -
                      (physics.kinetics.k4_0 * torch.exp(physics.kinetics.alpha_av * u_pred_fs) -
                       physics.transport.U_av * physics.scales.phic / physics.scales.lc * u_pred_fs_x) * av_pred_fs * physics.scales.cc)
    av_fs_loss = torch.mean(av_fs_residual ** 2)

    # Potential at f/s interface
    u_fs_residual = ((physics.materials.eps_film * physics.scales.phic / physics.scales.lc * u_pred_fs_x) -
                     (physics.materials.eps_Ddl * physics.scales.phic * u_pred_fs))
    u_fs_loss = torch.mean(u_fs_residual ** 2)

    # Holes at f/s interface
    h_fs_pred = networks['h'](inputs_fs)
    h_fs_pred_x = physics.grad_computer.compute_derivative(h_fs_pred, x_fs)

    q = torch.where(h_fs_pred <= (1e-9) / physics.scales.chc,
                    torch.zeros_like(h_fs_pred),
                    -(physics.kinetics.ktp_0 + (physics.constants.F * physics.transport.D_h * physics.scales.phic) /
                      (physics.constants.R * physics.constants.T * physics.scales.lc) * u_pred_fs_x))
    h_fs_residual = ((physics.transport.D_h * physics.scales.chc / physics.scales.lc) * h_fs_pred_x -
                     q * physics.scales.chc * h_fs_pred)
    if physics.config.pde.physics.include_holes:
        h_fs_loss = torch.mean(h_fs_residual ** 2)
    else:
        h_fs_loss = torch.mean(torch.zeros_like(h_fs_residual))

    # Total boundary loss
    total_boundary_loss = cv_mf_loss + u_mf_loss + cv_fs_loss + av_fs_loss + u_fs_loss + h_fs_loss+ av_mf_loss 

    individual_losses = {
        'cv_mf_bc': cv_mf_loss,
        'av_mf_bc': av_mf_loss,
        'u_mf_bc': u_mf_loss,
        'cv_fs_bc': cv_fs_loss,
        'av_fs_bc': av_fs_loss,
        'u_fs_bc': u_fs_loss,
        'h_fs_bc': h_fs_loss
    }

    if return_residuals:
        # Combine all residuals into single tensor for NTK computation

        residuals_dict = {'cv_mf_bc':cv_mf_residual, 'av_mf_bc':av_mf_residual, 'u_mf_bc':u_mf_residual, 
                    'cv_fs_bc':cv_fs_residual, 'av_fs_bc':av_mf_residual, 'u_fs_bc':u_fs_residual, 'h_fs_bc':h_fs_residual}
        combined_residuals = torch.cat([
            cv_mf_residual, cv_fs_residual,
            av_mf_residual, av_fs_residual,
            h_fs_residual, u_mf_residual, u_fs_residual
        ])
        return total_boundary_loss, individual_losses, combined_residuals,residuals_dict
    else:
        return total_boundary_loss, individual_losses
    
def compute_boundary_residuals_for_adaptive(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                                           networks, physics) -> torch.Tensor:
    """
    Compute boundary residuals for adaptive sampling - one residual per point.
    
    Args:
        x: Boundary spatial coordinates [batch_size, 1]
        t: Time coordinates [batch_size, 1] 
        E: Applied potential [batch_size, 1]
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
    
    Returns:
        residuals: One representative residual per boundary point [batch_size]
    """
    batch_size = x.shape[0]
    half_batch = batch_size // 2

    # Split into metal/film and film/solution interfaces
    x_mf = x[:half_batch]
    x_fs = x[half_batch:]
    t_mf = t[:half_batch]
    t_fs = t[half_batch:]
    E_mf = E[:half_batch]
    E_fs = E[half_batch:]

    # Get L derivative for m/f interface
    L_input = torch.cat([t, E], dim=1)
    L_pred = networks['film_thickness'](L_input)
    L_pred_t = physics.grad_computer.compute_derivative(L_pred, t)
    L_pred_t_mf = L_pred_t[:half_batch]

    # === Metal/film interface residuals ===
    inputs_mf = torch.cat([x_mf, t_mf, E_mf], dim=1)
    u_pred_mf = networks['potential'](inputs_mf)
    u_pred_mf_x = physics.grad_computer.compute_derivative(u_pred_mf, x_mf)

    # Compute all m/f residuals
    cv_pred_mf = networks['cv'](inputs_mf)
    cv_pred_mf_x = physics.grad_computer.compute_derivative(cv_pred_mf, x_mf)
    cv_mf_residual = ((-physics.transport.D_cv * physics.scales.cc / physics.scales.lc) * cv_pred_mf_x -
                      physics.kinetics.k1_0 * torch.exp(
                physics.kinetics.alpha_cv * physics.scales.phic * (E_mf / physics.scales.phic - u_pred_mf)) -
                      (physics.transport.U_cv * physics.scales.phic / physics.scales.lc * u_pred_mf_x -
                       physics.scales.lc / physics.scales.tc * L_pred_t_mf) * physics.scales.cc * cv_pred_mf)

    av_pred_mf = networks['av'](inputs_mf)
    av_pred_mf_x = physics.grad_computer.compute_derivative(av_pred_mf, x_mf)
    av_mf_residual = ((-physics.transport.D_av * physics.scales.cc / physics.scales.lc) * av_pred_mf_x -
                      (4 / 3) * physics.kinetics.k2_0 * torch.exp(
                physics.kinetics.alpha_av * physics.scales.phic * (E_mf / physics.scales.phic - u_pred_mf)) -
                      (physics.transport.U_av * physics.scales.phic / physics.scales.lc * u_pred_mf_x -
                       physics.scales.lc / physics.scales.tc * L_pred_t_mf) * av_pred_mf)

    u_mf_residual = ((physics.materials.eps_film * physics.scales.phic / physics.scales.lc * u_pred_mf_x) -
                     physics.materials.eps_Ddl * physics.scales.phic * (
                             u_pred_mf - E_mf / physics.scales.phic) / physics.geometry.d_Ddl)

    # Combine m/f residuals (L2 norm per point)
    mf_combined = torch.stack([cv_mf_residual.squeeze(), av_mf_residual.squeeze(), u_mf_residual.squeeze()], dim=1)
    mf_residuals = torch.norm(mf_combined, dim=1)  # [half_batch]

    # === Film/solution interface residuals ===
    inputs_fs = torch.cat([x_fs, t_fs, E_fs], dim=1)
    u_pred_fs = networks['potential'](inputs_fs)
    u_pred_fs_x = physics.grad_computer.compute_derivative(u_pred_fs, x_fs)

    # Compute all f/s residuals
    cv_pred_fs = networks['cv'](inputs_fs)
    cv_pred_fs_x = physics.grad_computer.compute_derivative(cv_pred_fs, x_fs)
    cv_fs_residual = ((-physics.transport.D_cv * physics.scales.cc / physics.scales.lc) * cv_pred_fs_x -
                      (physics.kinetics.k3_0 * torch.exp(physics.kinetics.beta_cv * physics.scales.phic * u_pred_fs) -
                       physics.transport.U_cv * physics.scales.phic / physics.scales.lc * u_pred_fs_x) * cv_pred_fs * physics.scales.cc)

    av_pred_fs = networks['av'](inputs_fs)
    av_pred_fs_x = physics.grad_computer.compute_derivative(av_pred_fs, x_fs)
    av_fs_residual = ((-physics.transport.D_av * physics.scales.cc / physics.scales.lc) * av_pred_fs_x -
                      (physics.kinetics.k4_0 * torch.exp(physics.kinetics.alpha_av * u_pred_fs) -
                       physics.transport.U_av * physics.scales.phic / physics.scales.lc * u_pred_fs_x) * av_pred_fs * physics.scales.cc)

    u_fs_residual = ((physics.materials.eps_film * physics.scales.phic / physics.scales.lc * u_pred_fs_x) -
                     (physics.materials.eps_Ddl * physics.scales.phic * u_pred_fs))

    h_fs_pred = networks['h'](inputs_fs)
    h_fs_pred_x = physics.grad_computer.compute_derivative(h_fs_pred, x_fs)
    q = torch.where(h_fs_pred <= (1e-9) / physics.scales.chc,
                    torch.zeros_like(h_fs_pred),
                    -(physics.kinetics.ktp_0 + (physics.constants.F * physics.transport.D_h * physics.scales.phic) /
                      (physics.constants.R * physics.constants.T * physics.scales.lc) * u_pred_fs_x))
    h_fs_residual = ((physics.transport.D_h * physics.scales.chc / physics.scales.lc) * h_fs_pred_x -
                     q * physics.scales.chc * h_fs_pred)

    # Combine f/s residuals (L2 norm per point)
    fs_combined = torch.stack([cv_fs_residual.squeeze(), av_fs_residual.squeeze(), 
                              u_fs_residual.squeeze(), h_fs_residual.squeeze()], dim=1)
    fs_residuals = torch.norm(fs_combined, dim=1)  # [half_batch]

    # Concatenate m/f and f/s residuals
    all_residuals = torch.cat([mf_residuals, fs_residuals], dim=0)  # [batch_size]
    
    return all_residuals


def compute_initial_loss(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                         networks, physics,
                         return_residuals: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
    """
    Compute initial condition losses.

    **Initial Conditions (t̂ = 0):**

    **Film Thickness:**

    .. math::
        \\hat{L}(0) = \\frac{L_0}{\\hat{L}_c}

    **Cation Vacancy Concentration:**

    .. math::
        \\hat{c}_{cv}(\\hat{x}, 0) = 0

    .. math::
        \\frac{\\partial \\hat{c}_{cv}}{\\partial \\hat{t}}\\bigg|_{\\hat{t}=0} = 0

    **Anion Vacancy Concentration:**

    .. math::
        \\hat{c}_{av}(\\hat{x}, 0) = 0

    .. math::
        \\frac{\\partial \\hat{c}_{av}}{\\partial \\hat{t}}\\bigg|_{\\hat{t}=0} = 0

    **Potential Distribution:**

    .. math::
        \\hat{\\phi}(\\hat{x}, 0) = \\frac{\\hat{E}}{\\hat{\\phi}_c} - \\frac{10^7 \\hat{L}_c}{\\hat{\\phi}_c}\\hat{x}

    .. math::
        \\frac{\\partial \\hat{\\phi}}{\\partial \\hat{t}}\\bigg|_{\\hat{t}=0} = 0

    **Hole Concentration:**

    .. math::
        \\hat{c}_h(\\hat{x}, 0) = \\frac{c_{h0}}{\\hat{c}_{h,c}} = 1

    .. math::
        \\frac{\\partial \\hat{c}_h}{\\partial \\hat{t}}\\bigg|_{\\hat{t}=0} = 0

    Args:
        x: Spatial coordinates
        t: Time coordinates (should be zeros)
        E: Applied potential
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        return_residuals: If True, also return raw residuals for NTK computation

    Returns:
        If return_residuals=False: Tuple of (total_initial_loss, individual_losses_dict)
        If return_residuals=True: Tuple of (total_initial_loss, individual_losses_dict, combined_residuals)
    """
    L_input = torch.cat([t, E], dim=1)
    L_initial_pred = networks['film_thickness'](L_input)
    inputs = torch.cat([x, t, E], dim=1)

    # Film thickness initial condition
    L_initial_residual = L_initial_pred - physics.domain.L_initial / physics.scales.lc
    L_initial_loss = torch.mean(L_initial_residual ** 2)

    # Cation vacancy initial conditions
    cv_initial_pred = networks['cv'](inputs)
    cv_initial_t = physics.grad_computer.compute_derivative(cv_initial_pred, t)
    cv_initial_residual = cv_initial_pred + cv_initial_t
    cv_initial_loss = torch.mean(cv_initial_pred**2) + torch.mean(cv_initial_t**2)

    # Anion vacancy initial conditions
    av_initial_pred = networks['av'](inputs)
    av_initial_t = physics.grad_computer.compute_derivative(av_initial_pred, t)
    av_initial_residual = av_initial_pred + av_initial_t
    av_initial_loss = torch.mean(av_initial_pred**2) + torch.mean(av_initial_t**2)

    # Potential initial conditions
    u_initial_pred = networks['potential'](inputs)
    u_initial_t = physics.grad_computer.compute_derivative(u_initial_pred, t)
    poisson_initial_residual = (u_initial_pred - (
            (E / physics.scales.phic) - (1e7 * (physics.scales.lc / physics.scales.phic) * x))) + u_initial_t
    poisson_initial_loss = torch.mean((u_initial_pred - (
            (E / physics.scales.phic) - (1e7 * (physics.scales.lc / physics.scales.phic) * x)))**2) + torch.mean(u_initial_t**2)

    # Hole initial conditions
    h_initial_pred = networks['h'](inputs)
    h_initial_t = physics.grad_computer.compute_derivative(h_initial_pred, t)
    h_initial_residual = (h_initial_pred - physics.scales.chc / physics.scales.chc) + h_initial_t
    if physics.config.pde.physics.include_holes:
        h_initial_loss = torch.mean((h_initial_pred - physics.scales.chc / physics.scales.chc)**2) + torch.mean(h_initial_t**2)
    else:
        h_initial_loss = torch.mean(torch.zeros_like((h_initial_pred - physics.scales.chc / physics.scales.chc))) + torch.mean(torch.zeros_like(h_initial_t))

    # Total initial loss
    total_initial_loss = cv_initial_loss + av_initial_loss + poisson_initial_loss + h_initial_loss + L_initial_loss

    individual_losses = {
        'cv_ic': cv_initial_loss,
        'av_ic': av_initial_loss,
        'poisson_ic': poisson_initial_loss,
        'h_ic': h_initial_loss,
        'L_ic': L_initial_loss
    }

    if return_residuals:
        # Combine all residuals into single tensor for NTK computation

        residual_dict = {'cv_ic':cv_initial_residual, 'av_ic':av_initial_residual, 'h_ic':h_initial_residual, 'poisson_ic':poisson_initial_residual, 'L_ic':L_initial_residual}
        combined_residuals = torch.cat([
            L_initial_residual,
            cv_initial_residual,
            av_initial_residual,
            poisson_initial_residual,
            h_initial_residual
        ])
        return total_initial_loss, individual_losses, combined_residuals,residual_dict
    else:
        return total_initial_loss, individual_losses

def compute_initial_residuals_for_adaptive(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                                         networks, physics) -> torch.Tensor:
    """
    Compute initial condition residuals for adaptive sampling - one residual per point.
    
    Returns:
        residuals: One representative residual per initial point [batch_size]
    """
    #TODO: We are not considering intial time derrivatives and probably should be doing so
    # All inputs for networks
    L_input = torch.cat([t, E], dim=1)
    L_initial_pred = networks['film_thickness'](L_input)
    inputs = torch.cat([x, t, E], dim=1)

    # Compute all initial condition residuals
    L_initial_residual = L_initial_pred - physics.domain.L_initial / physics.scales.lc

    cv_initial_pred = networks['cv'](inputs)
    cv_initial_t = physics.grad_computer.compute_derivative(cv_initial_pred, t)
    cv_initial_residual = cv_initial_pred  # Just the concentration residual

    av_initial_pred = networks['av'](inputs)
    av_initial_t = physics.grad_computer.compute_derivative(av_initial_pred, t)
    av_initial_residual = av_initial_pred  # Just the concentration residual

    u_initial_pred = networks['potential'](inputs)
    u_initial_t = physics.grad_computer.compute_derivative(u_initial_pred, t)
    poisson_initial_residual = (u_initial_pred - (
            (E / physics.scales.phic) - (1e7 * (physics.scales.lc / physics.scales.phic) * x)))

    h_initial_pred = networks['h'](inputs)
    h_initial_t = physics.grad_computer.compute_derivative(h_initial_pred, t)
    h_initial_residual = (h_initial_pred - physics.scales.chc / physics.scales.chc)

    # Combine all initial residuals using L2 norm per point
    # Note: L_initial_residual is per-E value, broadcast to match spatial points
    L_residual_expanded = L_initial_residual.expand_as(cv_initial_residual)
    
    combined_residuals = torch.stack([
        cv_initial_residual.squeeze(),
        av_initial_residual.squeeze(), 
        poisson_initial_residual.squeeze(),
        h_initial_residual.squeeze(),
        L_residual_expanded.squeeze()
    ], dim=1)
    
    # L2 norm across all initial conditions for each point
    point_residuals = torch.norm(combined_residuals, dim=1)  # [batch_size]
    
    return point_residuals

def compute_film_physics_loss(t: torch.Tensor, E: torch.Tensor,
                              networks, physics,
                              return_residuals: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute film growth physics loss.

     **Film Growth Equation:**

    .. math::
        \\frac{dL}{dt} = \\Omega (k_2 - k_5)

    **Dimensionless Form:**

    .. math::
        \\frac{d\\hat{L}}{d\\hat{t}} = \\frac{\\hat{t}_c \\Omega}{\\hat{L}_c} (k_2 - k_5)


    Args:
        t: Time coordinates
        E: Applied potential
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        return_residuals: If True, also return raw residuals for NTK computation

    Returns:
        If return_residuals=False: Film physics loss tensor
        If return_residuals=True: Tuple of (film_physics_loss, residuals)
    """
    inputs = torch.cat([t, E], dim=1)
    L_pred = networks['film_thickness'](inputs)

    # Get rate constants
    k1, k2, k3, k4, k5, ktp, ko2 = physics.compute_rate_constants(t, E, networks)

    # Compute predicted and physics-based dL/dt
    dl_dt_pred = physics.grad_computer.compute_derivative(L_pred, t)
    dL_dt_physics = (1 / physics.scales.lc) * physics.scales.tc * physics.materials.Omega * (k2 - k5)

    # Compute residual
    film_residual = dl_dt_pred - dL_dt_physics
    film_loss = torch.mean(film_residual ** 2)

    if return_residuals:
        return film_loss, film_residual
    else:
        return film_loss


def compute_total_loss(x_interior: torch.Tensor, t_interior: torch.Tensor, E_interior: torch.Tensor,
                       x_boundary: torch.Tensor, t_boundary: torch.Tensor, E_boundary: torch.Tensor,
                       x_initial: torch.Tensor, t_initial: torch.Tensor, E_initial: torch.Tensor,
                       t_film: torch.Tensor, E_film: torch.Tensor,
                       networks, physics,
                       weights: Optional[Dict[str, float]] = None,
                       ntk_weights: Optional[Dict[str, float]] = None,
                       return_residuals: bool = False) -> Union[Dict[str, torch.Tensor],
Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """
    Compute all losses and return detailed breakdown.

    Args:
        x_interior, t_interior, E_interior: Interior points
        x_boundary, t_boundary, E_boundary: Boundary points
        x_initial, t_initial, E_initial: Initial points
        t_film, E_film: Film physics points
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        weights: Standard loss weights dictionary (uniform/batch_size weighting)
        ntk_weights: NTK component weights dictionary (takes precedence over weights)
        return_residuals: If True, also return raw residuals for NTK computation

    Returns:
        If return_residuals=False: Dictionary with all loss components for monitoring and optimization
        If return_residuals=True: Tuple of (loss_dict, residuals_dict)
    """
    # Compute individual loss components
    if return_residuals:
        interior_loss, interior_breakdown, interior_residuals = compute_interior_loss(x_interior, t_interior,
                                                                                      E_interior, networks, physics,
                                                                                      return_residuals=True)
        boundary_loss, boundary_breakdown, boundary_residuals,_ = compute_boundary_loss(x_boundary, t_boundary,
                                                                                      E_boundary, networks, physics,
                                                                                      return_residuals=True)
        initial_loss, initial_breakdown, initial_residuals,_ = compute_initial_loss(x_initial, t_initial, E_initial,
                                                                                  networks, physics,
                                                                                  return_residuals=True)
        film_loss, film_residuals = compute_film_physics_loss(t_film, E_film, networks, physics, return_residuals=True)

        all_residuals = {
            **interior_residuals,  # cv_pde, av_pde, h_pde, poisson_pde
            'boundary': boundary_residuals,
            'initial': initial_residuals,
            'film_physics': film_residuals
        }
    else:
        interior_loss, interior_breakdown = compute_interior_loss(x_interior, t_interior, E_interior, networks, physics)
        boundary_loss, boundary_breakdown = compute_boundary_loss(x_boundary, t_boundary, E_boundary, networks, physics)
        initial_loss, initial_breakdown = compute_initial_loss(x_initial, t_initial, E_initial, networks, physics)
        film_loss = compute_film_physics_loss(t_film, E_film, networks, physics)

    # Apply weights - NTK weights take precedence
    if ntk_weights is not None:
        # Use NTK's granular component weights
        weighted_cv_pde = ntk_weights.get('cv_pde') * interior_breakdown['cv_pde']
        weighted_av_pde = ntk_weights.get('av_pde') * interior_breakdown['av_pde']
        weighted_h_pde = ntk_weights.get('h_pde') * interior_breakdown['h_pde']
        weighted_poisson_pde = ntk_weights.get('poisson_pde') * interior_breakdown['poisson_pde']

        weighted_interior = weighted_cv_pde + weighted_av_pde + weighted_h_pde + weighted_poisson_pde
        weighted_boundary = ntk_weights.get('boundary') * boundary_loss
        weighted_initial = ntk_weights.get('initial') * initial_loss
        weighted_film = ntk_weights.get('film_physics') * film_loss

        # Individual boundary and initial components with NTK weighting
        boundary_weight = ntk_weights.get('boundary')
        initial_weight = ntk_weights.get('initial')

    else:
        # Use standard weights (uniform, batch_size, or manual)
        if weights is None:
            weights = {
                'interior': 1.0,
                'boundary': 1.0,
                'initial': 1.0,
                'film_physics': 1.0
            }

        weighted_interior = weights['interior'] * interior_loss
        weighted_boundary = weights['boundary'] * boundary_loss
        weighted_initial = weights['initial'] * initial_loss
        weighted_film = weights['film_physics'] * film_loss

        # Individual components with standard weighting
        weighted_cv_pde = weights['interior'] * interior_breakdown['cv_pde']
        weighted_av_pde = weights['interior'] * interior_breakdown['av_pde']
        weighted_h_pde = weights['interior'] * interior_breakdown['h_pde']
        weighted_poisson_pde = weights['interior'] * interior_breakdown['poisson_pde']

        weighted_interior = weights['interior']*interior_loss 
        boundary_weight = weights['boundary']
        initial_weight = weights['initial']

    # Total loss
    total_loss = weighted_interior + weighted_boundary + weighted_initial + weighted_film

    # Combine all losses into one dictionary
    all_losses = {
        'total': total_loss,
        'interior': weighted_interior,
        'boundary': weighted_boundary,
        'initial': weighted_initial,
        'film_physics': weighted_film,

        # Individual PDE components
        'weighted_cv_pde': weighted_cv_pde,
        'weighted_av_pde': weighted_av_pde,
        'weighted_h_pde': weighted_h_pde,
        'weighted_poisson_pde': weighted_poisson_pde,

        # Individual boundary components
        **{f"weighted_{k}": boundary_weight * v for k, v in boundary_breakdown.items()},

        # Individual initial components
        **{f"weighted_{k}": initial_weight * v for k, v in initial_breakdown.items()}
    }

    if return_residuals:
        return all_losses, all_residuals
    else:
        return all_losses
    
def _extract_constraint_violations_al(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract constraint violations from loss dictionary.
    
    Key insight: In PINNACLE, loss_i = ‖C_i‖₂, so no transformation needed.
    """
    violations = {}
    
    # Boundary constraints - direct extraction
    boundary_constraints = ['cv_mf_bc', 'av_mf_bc', 'u_mf_bc', 
                          'cv_fs_bc', 'av_fs_bc', 'u_fs_bc', 'h_fs_bc']
    for bc_type in boundary_constraints:
        loss_key = f'weighted_{bc_type}'
        if loss_key in loss_dict:
            violations[bc_type] = loss_dict[loss_key]
    
    # Initial constraints - direct extraction  
    initial_constraints = ['cv_ic', 'av_ic', 'h_ic', 'poisson_ic', 'L_ic']
    for ic_type in initial_constraints:
        loss_key = f'weighted_{ic_type}'
        if loss_key in loss_dict:
            violations[ic_type] = loss_dict[loss_key]
    
    return violations


def _compute_al_terms(constraint_violations: Dict[str, torch.Tensor], 
                     al_manager,
                     residuals_dict: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Augmented Lagrangian penalty and Lagrangian terms.
    
    Returns:
        penalty_term: β * Σᵢ ‖Cᵢ‖²₂
        lagrangian_term: Σᵢ ⟨λᵢ, Cᵢ⟩
    """
    total_penalty = torch.tensor(0.0, device=al_manager.device)
    total_lagrangian = torch.tensor(0.0, device=al_manager.device)

    for constraint_name, violation_norm in constraint_violations.items():
        lambda_key = f'lambda_{constraint_name}'
        if lambda_key in al_manager.lambda_params and residuals_dict and constraint_name in residuals_dict:
            raw_residual = residuals_dict[constraint_name]
            lambda_vals = al_manager.lambda_params[lambda_key]
            
            N = len(raw_residual)  # Number of constraint points
            
            # Penalty term: (β/N) * Σ C²
            penalty_contribution = (al_manager.config.beta / N) * torch.sum(raw_residual ** 2)
            total_penalty += penalty_contribution
            
            # Lagrangian term: (1/N) * Σ λ * C  
            lagrangian_contribution = (1.0 / N) * torch.sum(lambda_vals * raw_residual)
            total_lagrangian += lagrangian_contribution
    
    return total_penalty, total_lagrangian

def compute_total_loss_al(x_interior, t_interior, E_interior,
                         x_boundary, t_boundary, E_boundary,
                         x_initial, t_initial, E_initial,
                         t_film, E_film, networks, physics,
                         al_manager,
                         return_residuals=True):
    """
    Compute total loss using Augmented Lagrangian method.
    
    Mathematical Framework:
    L_AL = L_PDE + β‖C‖² + ⟨λ, C⟩
    
    Where:
    - L_PDE: Interior PDE residual loss (objective)
    - C: Constraint violations (BC, IC, film)
    - β: Penalty parameter
    - λ: Learnable Lagrange multipliers
    """
    residuals_dict = {}
    loss_dict, _ = compute_total_loss(
            x_interior, t_interior, E_interior,
            x_boundary, t_boundary, E_boundary,
            x_initial, t_initial, E_initial,
            t_film, E_film, networks, physics,
            weights={'interior': 1.0, 'boundary': 1.0, 'initial': 1.0, 'film_physics': 1.0},
            return_residuals=True
        )
    
    #Construct detailed residuals_dict
    _, _, _, boundary_dict = compute_boundary_loss(x_boundary,t_boundary,E_boundary,networks,physics,return_residuals=True)
    _,_,_, iniital_dict = compute_initial_loss(x_initial,t_initial,E_initial,networks,physics,return_residuals=True)
    residuals_dict = {**boundary_dict, **iniital_dict}

    constraint_violations = _extract_constraint_violations_al(loss_dict)
    penalty_term, lagrangian_term = _compute_al_terms(
        constraint_violations, al_manager, residuals_dict
    )

    al_loss_dict = {
        'total': loss_dict['interior'] + penalty_term + lagrangian_term,
        'interior': loss_dict['interior'],
        'penalty': penalty_term,
        'lagrangian': lagrangian_term,
    }


    if return_residuals:
        return al_loss_dict, residuals_dict, constraint_violations
    return al_loss_dict
