# analysis/analysis.py
"""
Analysis and visualization functions for PINNACLE.

Simple functional approach to generate plots and analyze PINN results.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
import torch.nn as nn
from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics  
from training.training import PINNTrainer
from sampling.sampling import CollocationSampler
from losses.losses import compute_total_loss
from weighting.weighting import NTKWeightManager
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import gaussian_kde

def get_weights(networks: Dict[str,nn.Module]) -> list:
    """Extract parameters from all networks and concatenate into a big list"""
    param_list = []
    for key in networks.keys():
        param_list.extend([p.data for p in networks[key].parameters()])
    return param_list


def get_all_parameters(networks: Dict[str,nn.Module]) -> list:
    param_list = []
    for key in networks.keys():
        param_list.extend(networks[key].parameters())
    return param_list

def set_weights(networks: Dict[str,nn.Module], weights:Dict[str, Dict[str, torch.Tensor]],device:torch.device = None, directions:torch.Tensor=None, step:torch.Tensor=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(get_all_parameters(networks), weights):
            p.data.copy_(w.type(type(p.data),device=device))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(get_all_parameters(networks), weights, changes):
            p.data = w.to(device=device, dtype=w.dtype) + torch.as_tensor(d, device=device, dtype=w.dtype)
        

def get_random_weights(weights:Dict[str, Dict[str, torch.Tensor]],device:torch.device = None ):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size(),device=device) for w in weights]

def get_diff_weights(weights:Dict[str, Dict[str, torch.Tensor]], weights2:Dict[str, Dict[str, torch.Tensor]]):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]

def normalize_direction(direction:torch.Tensor, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction:torch.Tensor, weights:Dict[str, Dict[str, torch.Tensor]], norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)
            
def ignore_biasbn(directions:torch.Tensor):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

             
def create_random_direction(networks:Dict[str,nn.Module],device, ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.
        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'
        Returns:
          direction: a random direction with the same dimension as weights or states.
    """


    weights = get_weights(networks) # a list of parameters.
    direction = get_random_weights(weights,device=device)
    normalize_directions_for_weights(direction, weights, norm, ignore)

    return direction


def create_loss_landscape(networks:Dict[str,nn.Module], physics:ElectrochemicalPhysics , sampler:CollocationSampler, device:torch.device, save_path=None) -> Any:
    """
    Generate a 2D loss landscape visualization around the current trained model parameters.
    
    This function implements the "filter normalization" technique from Basir et al. (2023) to create
    a 2D slice through the high-dimensional loss surface. The landscape shows how the total loss
    changes when model parameters are perturbed along two random directions from the current
    trained state.
    
    Mathematical Approach:
    ----------------------
    1. Extract current trained parameters θ* from all networks
    2. Generate two random directions ζ and γ with filter normalization
    3. Create 2D grid where each point (e₁, e₂) represents: θ* + e₁·ζ + e₂·γ
    4. Compute total physics-informed loss at each grid point using fixed collocation points
    5. Visualize the resulting loss surface as a contour plot
    
    The filter normalization ensures that perturbations are scaled appropriately relative to
    the magnitude of trained weights in each network layer, preventing bias toward layers
    with larger parameter scales.
    
    Parameters:
    -----------
    networks : NetworkManager
        Trained neural networks (potential, cv, av, h, film_thickness networks)
    physics : ElectrochemicalPhysics  
        Physics module for PDE residual computation
    sampler : CollocationSampler
        Sampler for generating collocation points
    save_path : str, optional
        Path to save the resulting contour plot. If None, plot is displayed but not saved.
    
    Returns:
    --------
    landscape : np.ndarray, shape=(grid_size, grid_size)
        2D array of loss values at each grid point
    x_coords : np.ndarray
        X-coordinates of the grid (perturbation steps along direction 1)  
    y_coords : np.ndarray
        Y-coordinates of the grid (perturbation steps along direction 2)
        
    References:
    -----------
    Basir, S. (2023). "Investigating and Mitigating Failure Modes in Physics-informed 
    Neural Networks (PINNs)." Communications in Computational Physics.
    
    Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). "Visualizing the 
    loss landscape of neural nets." Advances in Neural Information Processing Systems.
    """

    num_of_points = 50 
    xcoordinates = np.linspace(-1.0, 1.0, num=num_of_points) 
    ycoordinates = np.linspace(-1.0, 1.0, num=num_of_points)
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)

    weights = get_weights(networks)# Current trained parameters θ*
    random_direction1 = create_random_direction(networks,device=device)  # Random direction ζ  
    random_direction2 = create_random_direction(networks,device=device)  # Random direction γ
    directions = [random_direction1, random_direction2]

    x_interior,t_interior,E_interior  = sampler.sample_interior_points(networks)
    x_boundary,t_boundary,E_boundary = sampler.sample_boundary_points(networks)
    x_initial,t_initial,E_initial = sampler.sample_initial_points(networks)
    t_film,E_film = sampler.sample_film_physics_points()

    loss_components = [
        'total', 'interior', 'boundary', 'initial', 'film_physics',
        # Granular interior components
        'weighted_cv_pde', 'weighted_av_pde', 'weighted_h_pde', 'weighted_poisson_pde',
        # Granular boundary components
        'weighted_cv_mf_bc', 'weighted_av_mf_bc', 'weighted_u_mf_bc', 
        'weighted_cv_fs_bc', 'weighted_av_fs_bc', 'weighted_u_fs_bc', 'weighted_h_fs_bc',
        # Granular initial components
        'weighted_cv_ic', 'weighted_av_ic', 'weighted_poisson_ic', 'weighted_h_ic', 'weighted_L_ic'
    ]
    loss_landscapes = {comp: np.zeros((num_of_points,num_of_points)) for comp in loss_components}

    for row in tqdm(range(num_of_points)):
        for col in range(num_of_points):
            step_x = xcoord_mesh[row][col]
            step_y = ycoord_mesh[row][col]
            step = [step_x, step_y]
            
            set_weights(networks, weights, device, directions, step)
            
            loss_dict = compute_total_loss(x_interior, t_interior, E_interior, 
                                     x_boundary, t_boundary, E_boundary,
                                     x_initial, t_initial, E_initial,
                                     t_film, E_film, networks, physics)

            # Store each loss component
            with torch.no_grad():
                for comp in loss_components:
                    if comp in loss_dict:
                        loss_landscapes[comp][row][col] = loss_dict[comp].item()


    cmap_list = ['jet','YlGnBu','coolwarm','rainbow','magma','plasma','inferno','Spectral','RdBu']
    cm = plt.cm.get_cmap(cmap_list[6]).reversed()

    for comp_name, loss_list in loss_landscapes.items():
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=0.1)
        
        fig = plt.figure(figsize=(5,5))
        ax = plt.subplot(projection='3d')
        
        # Remove gray panes and axis grid
        ax.xaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.fill = False
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.fill = False
        ax.zaxis.pane.set_edgecolor('white')
        ax.grid(False)
        ax.zaxis.set_visible(False)

        vmax = np.max(np.log(loss_list+1e-14))
        vmin = np.min(np.log(loss_list+1e-14))
        
        plot = ax.plot_surface(xcoord_mesh, ycoord_mesh, np.log(loss_list+1e-14),
                            cmap=cm, linewidth=0, vmin=vmin, vmax=vmax)

        cset = ax.contourf(xcoord_mesh, ycoord_mesh, np.log(loss_list+1e-14),
                        zdir='z', offset=np.min(np.log(loss_list+1e-14)-0.2), cmap=cm)

        ax.view_init(elev=25, azim=-45)
        ax.dist=11
        ticks = np.linspace(vmin, vmax, 4, endpoint=True)
        cbar = fig.colorbar(plot, ax=ax, shrink=0.50,ticks=ticks,pad=0.02)
        cbar.formatter.set_powerlimits((1, 1))

        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.set_xlabel(r"$\epsilon_1$")
        ax.set_ylabel(r"$\epsilon_2$")
        ax.set_zlim(vmin, vmax)
        ax.set_title(f'{comp_name.title()} Loss Landscape')

        if save_path:
            plt.savefig(f"{save_path}/loss_landscape_{comp_name}", bbox_inches='tight', pad_inches=0.2)
        plt.close()


def plot_ntk_weight_densities(ntk_weight_manager:NTKWeightManager, save_path: Optional[str] = None) -> None:
    """
    Plot density distributions of NTK weights for each loss component.
    
    Creates a plot similar to Figure 3 in Basir et al. (2023) showing the 
    distribution of NTK weight values assigned to each loss component during training.
    
    Args:
        trainer: PINNTrainer instance with ntk_weight_distributions data
        save_path: Optional path to save the plot
    """
    
    distributions = ntk_weight_manager.ntk_weight_distributions

    print("🔬 Generating NTK weight density plots...")

    plt.figure(figsize=(10, 6))

    component_names = ['cv_pde', 'av_pde', 'h_pde', 'poisson_pde', 
                      'boundary', 'initial', 'film_physics']
    colors = plt.cm.Set1(np.linspace(0, 1, len(component_names)))

    for i, comp in enumerate(component_names):
        if comp in distributions and len(distributions[comp]) > 1:
            weights = np.array(distributions[comp])
            kde = gaussian_kde(weights)
            x_range = np.linspace(weights.min(), weights.max(), 100)
            density = kde(x_range)
            
            plt.plot(x_range, density, color=colors[i], 
                    linewidth=2, label=comp, alpha=0.8)

    plt.xlabel('NTK Weight Values')
    plt.ylabel('Density')
    plt.title('NTK Weight Distributions by Loss Component')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable y-axis limits
    plt.ylim(1e-6, 1e3)
    
    plt.tight_layout()

    plt.savefig(f"{save_path}/ntk_density_plots.png", dpi=300, bbox_inches='tight')



def visualize_predictions(networks, physics, step: str = "final", save_path: Optional[str] = None) -> None:
    """
    Visualize network predictions across input ranges.

    **Visualization Components:**

    - **Potential Field**: φ̂(x̂,t̂) contour plot
    - **Concentration Fields**: ĉ_cv, ĉ_av, ĉ_h contour plots
    - **Film Thickness Evolution**: L̂(t̂) vs time
    - **Potential Profile**: φ̂(x̂) at fixed time slice

    All plots use dimensionless variables as predicted by the networks.

    Args:
        networks: NetworkManager instance with trained networks
        physics: ElectrochemicalPhysics instance
        step: Training step identifier for plot titles
        save_path: Optional path to save plots
    """
    print(f"📊 Generating prediction visualization for step {step}...")

    with torch.no_grad():
        # Define input ranges (all dimensionless)
        n_spatial = 50
        n_temporal = 50

        # Fix a representative dimensionless potential for visualization
        E_hat_fixed = torch.tensor([[0.8 / physics.scales.phic]], device=physics.device)  # Normalized E

        # Dimensionless time range (0 to 1)
        t_hat_range = torch.linspace(0, 1.0, n_temporal, device=physics.device)

        # Get final dimensionless film thickness to set spatial range
        t_hat_final = torch.tensor([[1.0]], device=physics.device)
        L_hat_final = networks['film_thickness'](torch.cat([t_hat_final, E_hat_fixed], dim=1)).item()
        x_hat_range = torch.linspace(0, L_hat_final, n_spatial, device=physics.device)

        print(f"  📐 Dimensionless time range: [0, 1.0]")
        print(f"  📐 Dimensionless spatial range: [0, {L_hat_final:.2f}]")
        print(f"  📐 Fixed dimensionless potential: {E_hat_fixed.item():.3f}")

        # Create 2D grid for contour plots
        T_hat_mesh, X_hat_mesh = torch.meshgrid(t_hat_range, x_hat_range, indexing='ij')
        E_hat_mesh = torch.full_like(T_hat_mesh, E_hat_fixed.item())

        # Stack inputs for 3D networks
        inputs_3d = torch.stack([
            X_hat_mesh.flatten(),
            T_hat_mesh.flatten(),
            E_hat_mesh.flatten()
        ], dim=1)

        # Get network predictions
        phi_hat_2d = networks['potential'](inputs_3d).reshape(n_temporal, n_spatial)
        c_cv_hat_2d = networks['cv'](inputs_3d).reshape(n_temporal, n_spatial)
        c_av_hat_2d = networks['av'](inputs_3d).reshape(n_temporal, n_spatial)
        c_h_hat_2d = networks['h'](inputs_3d).reshape(n_temporal, n_spatial)

        # Film thickness evolution (dimensionless)
        t_hat_1d = t_hat_range.unsqueeze(1)
        E_hat_1d = torch.full_like(t_hat_1d, E_hat_fixed.item())
        L_inputs_1d = torch.cat([t_hat_1d, E_hat_1d], dim=1)
        L_hat_1d = networks['film_thickness'](L_inputs_1d).squeeze()

        # Convert to numpy for plotting
        t_hat_np = t_hat_range.cpu().numpy()
        x_hat_np = x_hat_range.cpu().numpy()
        T_hat_np, X_hat_np = np.meshgrid(t_hat_np, x_hat_np, indexing='ij')

        phi_hat_np = phi_hat_2d.cpu().numpy()
        c_cv_hat_np = c_cv_hat_2d.cpu().numpy()
        c_av_hat_np = c_av_hat_2d.cpu().numpy()
        c_h_hat_np = c_h_hat_2d.cpu().numpy()
        L_hat_np = L_hat_1d.cpu().numpy()

        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Dimensionless potential field
        im1 = axes[0, 0].contourf(X_hat_np, T_hat_np, phi_hat_np, levels=50, cmap='RdBu_r')
        axes[0, 0].set_xlabel('Dimensionless Position x̂')
        axes[0, 0].set_ylabel('Dimensionless Time t̂')
        axes[0, 0].set_title(f'Dimensionless Potential φ̂(x̂,t̂) at Ê={E_hat_fixed.item():.3f}')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Dimensionless cation vacancies
        im2 = axes[0, 1].contourf(X_hat_np, T_hat_np, c_cv_hat_np, levels=50, cmap='Reds')
        axes[0, 1].set_xlabel('Dimensionless Position x̂')
        axes[0, 1].set_ylabel('Dimensionless Time t̂')
        axes[0, 1].set_title(f'Dimensionless Cation Vacancies ĉ_cv at Ê={E_hat_fixed.item():.3f}')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Dimensionless anion vacancies
        im3 = axes[0, 2].contourf(X_hat_np, T_hat_np, c_av_hat_np, levels=50, cmap='Blues')
        axes[0, 2].set_xlabel('Dimensionless Position x̂')
        axes[0, 2].set_ylabel('Dimensionless Time t̂')
        axes[0, 2].set_title(f'Dimensionless Anion Vacancies ĉ_av at Ê={E_hat_fixed.item():.3f}')
        plt.colorbar(im3, ax=axes[0, 2])

        # 4. Dimensionless holes
        im4 = axes[1, 0].contourf(X_hat_np, T_hat_np, c_h_hat_np, levels=50, cmap='Purples')
        axes[1, 0].set_xlabel('Dimensionless Position x̂')
        axes[1, 0].set_ylabel('Dimensionless Time t̂')
        axes[1, 0].set_title(f'Dimensionless Holes ĉ_h at Ê={E_hat_fixed.item():.3f}')
        plt.colorbar(im4, ax=axes[1, 0])

        # 5. Dimensionless film thickness
        axes[1, 1].plot(t_hat_np, L_hat_np, 'k-', linewidth=3)
        axes[1, 1].set_xlabel('Dimensionless Time t̂')
        axes[1, 1].set_ylabel('Dimensionless Film Thickness L̂')
        axes[1, 1].set_title(f'Dimensionless Film Thickness L̂(t̂) at Ê={E_hat_fixed.item():.3f}')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Potential profile vs spatial position at fixed time
        x_hat_sweep = torch.linspace(0, L_hat_final, 50, device=physics.device)
        t_hat_mid = torch.full((50, 1), 0.5, device=physics.device)  # Middle time
        E_hat_mid = torch.full((50, 1), E_hat_fixed.item(), device=physics.device)

        x_sweep_inputs = torch.cat([x_hat_sweep.unsqueeze(1), t_hat_mid, E_hat_mid], dim=1)
        phi_vs_x = networks['potential'](x_sweep_inputs).cpu().numpy()

        axes[1, 2].plot(x_hat_sweep.cpu().numpy(), phi_vs_x, 'r-', linewidth=2)
        axes[1, 2].set_xlabel('Dimensionless Position x̂')
        axes[1, 2].set_ylabel('Dimensionless Potential φ̂')
        axes[1, 2].set_title(f'Potential Profile φ̂(x̂) at t̂=0.5, Ê={E_hat_fixed.item():.3f}')
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'Dimensionless Network Predictions Overview - Step {step}', fontsize=16)
        plt.tight_layout()

        # Save plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"  💾 Saved predictions plot to {save_path}")
        else:
            plt.show()

        plt.close()

        # Print dimensionless statistics
        print(f"\n📈 Dimensionless Prediction Statistics (Step {step}) at Ê={E_hat_fixed.item():.3f}:")
        print("-" * 60)
        print(
            f"Potential φ̂:          {phi_hat_np.min():.3f} to {phi_hat_np.max():.3f} (mean: {phi_hat_np.mean():.3f})")
        print(
            f"Cation Vacancies ĉ_cv: {c_cv_hat_np.min():.3f} to {c_cv_hat_np.max():.3f} (mean: {c_cv_hat_np.mean():.3f})")
        print(
            f"Anion Vacancies ĉ_av:  {c_av_hat_np.min():.3f} to {c_av_hat_np.max():.3f} (mean: {c_av_hat_np.mean():.3f})")
        print(
            f"Holes ĉ_h:             {c_h_hat_np.min():.3f} to {c_h_hat_np.max():.3f} (mean: {c_h_hat_np.mean():.3f})")
        print(f"Film Thickness L̂:      {L_hat_np.min():.3f} to {L_hat_np.max():.3f}")

        # Convert back to dimensional units for reference
        print(f"\n🔧 Corresponding Dimensional Values:")
        print(f"Time scale: {physics.scales.tc:.1e} s")
        print(f"Length scale: {physics.scales.lc:.1e} m")
        print(f"Potential scale: {physics.scales.phic:.3f} V")
        print(f"Final dimensional thickness: {L_hat_np.max() * physics.scales.lc * 1e9:.2f} nm")

def generate_polarization_curve(networks, physics, t_hat_eval: float = 1.0, n_points: int = 50,
                                save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """ 
     Generate polarization curve at specified time.

    ..math::
        $$j_1 = \frac{8}{3} F k_1 [Fe] [v_{Fe}^{3'}]$$

    ..math::
        $$j_2 = \frac{8}{3} F k_2 [Fe]$$

    ..math::
        $$j_3 = \frac{1}{3} F k_3 [Fe_{ox}]$$

    ..math::    
        $$j_{tp} = 1 \times  F k_{tp} [Fe_3O_4] [H^+]^8 [h^+]$$
     
    Args:
        networks: NetworkManager instance with trained networks
        physics: ElectrochemicalPhysics instance
        t_hat_eval: Dimensionless time for evaluation
        n_points: Number of potential points
        save_path: Optional path to save plot

    Returns:
        Tuple of (E_hat_values, I_hat_values, I_c_scale)
    """

    print(f"📈 Generating polarization curve at t̂={t_hat_eval}")

    # Define potential range
    E_hat_min = physics.geometry.E_min/physics.scales.phic 
    E_hat_max = physics.geometry.E_max/physics.scales.phic

    with torch.no_grad():
        E_hat_values = torch.linspace(E_hat_min, E_hat_max, n_points, device=physics.device)
        j={'total':[],'j1':[],'j2':[],'j3':[],'jtp':[]}

        for E_hat_val in E_hat_values:
            # Use dimensionless quantities throughout
            t_hat_tensor = torch.tensor([[t_hat_eval]], device=physics.device)
            E_hat_tensor = torch.tensor([[E_hat_val.item()]], device=physics.device)

            # Get dimensionless film thickness
            L_hat_val = networks['film_thickness'](torch.cat([t_hat_tensor, E_hat_tensor], dim=1))

            # Evaluate at interfaces
            x_hat_fs = L_hat_val  # f/s interface
            x_hat_mf = torch.zeros_like(L_hat_val)  # m/f interface

            inputs_fs = torch.cat([x_hat_fs, t_hat_tensor, E_hat_tensor], dim=1)
            inputs_mf = torch.cat([x_hat_mf, t_hat_tensor, E_hat_tensor], dim=1)

            # Get dimensionless concentrations and rate constants
            k1, k2, k3, k4, k5, ktp, ko2 = physics.compute_rate_constants(t_hat_tensor, E_hat_tensor, networks,
                                                                          single=True)

            h_hat_fs = networks['h'](inputs_fs)  # Dimensionless hole concentration
            cv_hat_mf = networks['cv'](inputs_mf)  # Dimensionless CV concentration
            av_hat_fs = networks['av'](inputs_fs)

            #Compute Currents
            j1 = (8.0/3.0)*physics.constants.F*k1*cv_hat_mf*physics.scales.cc
            j2 = (8.0/3.0)*physics.constants.F*k2
            j3 = (1.0/3.0)*physics.constants.F*k3*av_hat_fs*physics.scales.cc
            jtp = physics.constants.F*ktp*(physics.materials.c_H**8)*h_hat_fs*physics.scales.chc
            j_total = j1 +j2 + j3 + jtp
            j["total"].append(j_total.item())
            j["j1"].append(j1.item())
            j["j2"].append(j2.item())
            j["j3"].append(j3.item())
            j["jtp"].append(jtp.item())

        # Convert to numpy for plotting
        E_np = E_hat_values.cpu().numpy()*physics.scales.phic
        j_np = np.array(j["total"])
        j_1_np = np.array(j['j1'])
        j_2_np = np.array(j['j2'])
        j_3_np = np.array(j['j3'])
        j_tp_np = np.array(j['jtp'])


        # Create polarization curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(E_np, j_np, 'b-', linewidth=2, label='Total Current')
        plt.plot(E_np, j_1_np, 'r-', linewidth=2, label='Current due to R1')
        plt.plot(E_np, j_2_np, 'g-', linewidth=2, label='Current due to R2')
        plt.plot(E_np, j_3_np, 'y-', linewidth=2, label='Current due to R3')
        plt.plot(E_np, j_tp_np, 'm-', linewidth=2, label='Current due to RTP')
        plt.xlabel('Applied Potential E/ V')
        plt.ylabel('Current Density j: A/m^2')
        plt.title(f'Polarization Curve at t̂={t_hat_eval}')
        plt.legend()
        #TODO: Make the X-Axis plots finer. 
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Saved polarization curve to {save_path}")
        else:
            plt.show()

        plt.close()

def plot_training_losses(loss_history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Create comprehensive plots of all loss components.

    **Loss Components Visualization:**

    - **Main Categories**: Total, Interior (PDE), Boundary, Initial, Film Physics
    - **PDE Breakdown**: Individual Nernst-Planck and Poisson residuals
    - **Boundary Breakdown**: Metal/film and film/solution interface losses
    - **Initial Breakdown**: All initial condition components

    Args:
        loss_history: Dictionary of loss histories from training
        save_path: Optional path to save plot
    """
    print("📉 Creating comprehensive loss plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Total and main components
    if loss_history['total']:
        axes[0, 0].semilogy(loss_history['total'], label='Total Loss', linewidth=2)
        axes[0, 0].semilogy(loss_history['interior'], label='Interior (PDE)', alpha=0.8)
        axes[0, 0].semilogy(loss_history['boundary'], label='Boundary', alpha=0.8)
        axes[0, 0].semilogy(loss_history['initial'], label='Initial', alpha=0.8)
        axes[0, 0].semilogy(loss_history['film_physics'], label='Film Thickness', alpha=0.8)
    axes[0, 0].set_title('Main Loss Components')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PDE residuals breakdown
    if loss_history['weighted_cv_pde']:
        axes[0, 1].semilogy(loss_history['weighted_cv_pde'], label='CV PDE', alpha=0.8)
        axes[0, 1].semilogy(loss_history['weighted_av_pde'], label='AV PDE', alpha=0.8)
        axes[0, 1].semilogy(loss_history['weighted_h_pde'], label='Hole PDE', alpha=0.8)
        axes[0, 1].semilogy(loss_history['weighted_poisson_pde'], label='Poisson PDE', alpha=0.8)
    axes[0, 1].set_title('Individual PDE Residuals')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Boundary conditions breakdown
    if loss_history['weighted_cv_mf_bc']:
        axes[1, 0].semilogy(loss_history['weighted_cv_mf_bc'], label='CV (m/f)', alpha=0.8)
        axes[1, 0].semilogy(loss_history['weighted_av_mf_bc'], label='AV (m/f)', alpha=0.8)
        axes[1, 0].semilogy(loss_history['weighted_u_mf_bc'], label='Potential (m/f)', alpha=0.8)
        axes[1, 0].semilogy(loss_history['weighted_cv_fs_bc'], label='CV (f/s)', alpha=0.8)
        axes[1, 0].semilogy(loss_history['weighted_av_fs_bc'], label='AV (f/s)', alpha=0.8)
        axes[1, 0].semilogy(loss_history['weighted_u_fs_bc'], label='Potential (f/s)', alpha=0.8)
        axes[1, 0].semilogy(loss_history['weighted_h_fs_bc'], label='Hole (f/s)', alpha=0.8)
    axes[1, 0].set_title('Boundary Condition Losses')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Initial conditions breakdown
    if loss_history['weighted_cv_ic']:
        axes[1, 1].semilogy(loss_history['weighted_cv_ic'], label='CV IC', alpha=0.8)
        axes[1, 1].semilogy(loss_history['weighted_av_ic'], label='AV IC', alpha=0.8)
        axes[1, 1].semilogy(loss_history['weighted_h_ic'], label='Hole IC', alpha=0.8)
        axes[1, 1].semilogy(loss_history['weighted_poisson_ic'], label='Poisson IC', alpha=0.8)
        axes[1, 1].semilogy(loss_history['weighted_L_ic'], label='Film Thickness IC', alpha=0.8)
    axes[1, 1].set_title('Initial Condition Losses')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"  💾 Saved loss plots to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_training_results(trainer:PINNTrainer, ntk_weight_manager:NTKWeightManager,save_dir: Optional[str] = None) -> None:
    """
    Complete analysis of training results.

    Generates all standard plots and analysis for a trained PINNACLE model.

    Args:
        trainer: PINNTrainer instance with completed training
        save_dir: Optional directory to save all plots
    """
    print("🔬 Performing complete training analysis...")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        loss_plot_path = os.path.join(save_dir, "training_losses.png")
        predictions_plot_path = os.path.join(save_dir, "predictions_overview.png")
        polarization_plot_path = os.path.join(save_dir, "polarization_curve.png")
    else:
        loss_plot_path = None
        predictions_plot_path = None
        polarization_plot_path = None

    # Plot training losses
    plot_training_losses(trainer.loss_history, save_path=loss_plot_path)

    # Visualize predictions
    visualize_predictions(trainer.networks, trainer.physics, step="final", save_path=predictions_plot_path)

    # Generate polarization curve
    generate_polarization_curve(trainer.networks, trainer.physics, save_path=polarization_plot_path)

    #Plot ntk weight density plots
    plot_ntk_weight_densities(ntk_weight_manager,save_path=save_dir)

    # Print training statistics
    stats = trainer.get_training_stats()
    print(f"\n📊 Training Summary:")
    print(f"  Steps completed: {stats['current_step']}/{stats['total_steps']}")
    print(f"  Final loss: {stats['final_loss']:.6f}")
    print(f"  Best loss: {stats['best_loss']:.6f}")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    if stats['training_time_minutes']:
        print(f"  Training time: {stats['training_time_minutes']:.1f} minutes")

    print("✅ Analysis complete!")
