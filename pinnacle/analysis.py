# pinnacle/analysis.py
"""
Analysis and visualization functions for PINNACLE.

Simple functional approach to generate plots and analyze PINN results.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, List, Optional, Tuple
import os


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

    **Polarization Curve Calculation:**

    The current density is computed from electrochemical rate constants:

    .. math::
        I = \\frac{8}{3}k_1 \\hat{c}_{cv} + \\frac{8}{3}k_2 F + \\frac{1}{3}k_3 F - k_{tp} \\hat{c}_h F

    where the rate constants follow Butler-Volmer kinetics:

    .. math::
        k_i = k_{i,0} \\exp\\left(\\frac{\\alpha_i n_i F}{RT}(E - \\phi)\\right)

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

    # Define dimensionless potential range
    E_hat_min = physics.geometry.E_min / physics.scales.phic
    E_hat_max = physics.geometry.E_max / physics.scales.phic

    # Define characteristic current density scale
    # I_c = F * cc / tc (charge per unit time per unit area)
    I_c = physics.constants.F * physics.scales.cc / physics.scales.tc

    with torch.no_grad():
        # Create dimensionless potential sweep
        E_hat_values = torch.linspace(E_hat_min, E_hat_max, n_points, device=physics.device)
        currents_hat = []

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

            # Calculate dimensionless current contributions
            #TO-DO: Need to fix this with the new equations Conrard sent
            # Note: k1, k2, etc. have units of 1/time, need to multiply by tc to make dimensionless
            current_k1_hat = (8.0 / 3.0) * k1 * cv_hat_mf
            current_k2_hat = (8.0 / 3.0) * k2 * physics.constants.F
            current_k3_hat = (1.0 / 3.0) * k3 * physics.constants.F
            current_ktp_hat = (-1.0) * ktp * h_hat_fs * physics.constants.F
            total_current_hat = current_k1_hat + current_k2_hat + current_k3_hat + current_ktp_hat
            currents_hat.append(total_current_hat.item())

        # Convert to numpy for plotting
        E_hat_np = E_hat_values.cpu().numpy()
        I_hat_np = np.array(currents_hat)

        # Create polarization curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(E_hat_np, I_hat_np, 'b-', linewidth=2, label='Total Dimensionless Current')
        plt.xlabel('Dimensionless Applied Potential Ê = E/φc')
        plt.ylabel('Dimensionless Current Density Î = I/Ic')
        plt.title(f'Dimensionless Polarization Curve at t̂={t_hat_eval}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add dimensional scale info to plot
        plt.figtext(0.02, 0.02, f'Current scale Ic = {I_c:.2e} A/m²\nPotential scale φc = {physics.scales.phic:.3f} V',
                    fontsize=8, ha='left')

        # Save plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Saved polarization curve to {save_path}")
        else:
            plt.show()

        plt.close()

        print(f"📊 Dimensionless current range: {I_hat_np.min():.2e} to {I_hat_np.max():.2e}")
        print(f"📊 Corresponding dimensional range: {I_hat_np.min() * I_c:.2e} to {I_hat_np.max() * I_c:.2e} A/m²")

        return E_hat_np, I_hat_np, I_c


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


def analyze_training_results(trainer, save_dir: Optional[str] = None) -> None:
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


# Convenience functions for quick analysis
def quick_analysis(trainer) -> None:
    """Quick analysis without saving plots."""
    analyze_training_results(trainer, save_dir=None)


def save_all_plots(trainer, output_dir: str) -> None:
    """Save all analysis plots to specified directory."""
    analyze_training_results(trainer, save_dir=output_dir)