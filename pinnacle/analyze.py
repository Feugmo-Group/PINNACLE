#!/usr/bin/env python3
import torch
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics
from sampling.sampling import CollocationSampler
from training.training import PINNTrainer
from analysis.analysis import analyze_training_results, create_loss_landscape

def load_fem_data(fem_data_dir):
    """
    Load FEM validation data from txt files.
    
    Args:
        fem_data_dir: Directory containing FEM txt files with format "X.X V.txt"
        
    Returns:
        dict: Dictionary with voltage as keys and pandas DataFrames as values
    """
    fem_data = {}
    fem_dir = Path(fem_data_dir)
    
    if not fem_dir.exists():
        print(f"Warning: FEM data directory {fem_data_dir} does not exist")
        return fem_data
    
    # Look for txt files in the directory
    txt_files = list(fem_dir.glob("*.txt"))
    
    for txt_file in txt_files:
        try:
            # Extract voltage from filename like "0.1 V.txt"
            filename = txt_file.stem
            
            # Try to extract voltage from filename
            import re
            voltage_match = re.search(r'(\d+\.?\d*)\s*V', filename)
            if voltage_match:
                voltage = float(voltage_match.group(1))
            else:
                print(f"Warning: Could not extract voltage from filename {filename}")
                continue
            
            # Load the data assuming tab-separated with headers
            try:
                df = pd.read_csv(txt_file, sep='\t')
                # Expected columns: Time/s, Potential/V, Filmthickness/m
                
                if 'Time/s' not in df.columns or 'Filmthickness/m' not in df.columns:
                    print(f"Warning: Expected columns not found in {txt_file}")
                    print(f"Available columns: {df.columns.tolist()}")
                    continue
                
                fem_data[voltage] = df
                print(f"Loaded FEM data for {voltage}V: {len(df)} time points")
                
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
                    
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
    
    return fem_data

def compare_film_thickness_curves(networks, physics, fem_data, step="final", 
                                save_path=None, voltages_to_compare=None,
                                max_time=None):
    """
    Compare PINN film thickness predictions with FEM temporal data.
    
    Args:
        networks: NetworkManager instance with trained networks
        physics: Physics object  
        fem_data: Dictionary of FEM validation DataFrames
        step: Training step identifier
        save_path: Path to save the plot
        voltages_to_compare: List of voltages to compare (if None, uses all available)
        max_time: Maximum time for PINN predictions (if None, uses FEM max time)
    """
    import matplotlib.pyplot as plt
    
    if voltages_to_compare is None:
        voltages_to_compare = sorted(list(fem_data.keys()))
    
    n_voltages = len(voltages_to_compare)
    if n_voltages == 0:
        print("No FEM data available for comparison")
        return
    
    # Create subplots - 2 rows: individual curves and comparison
    fig = plt.figure(figsize=(15, 10))
    
    # Individual voltage plots
    n_cols = min(4, n_voltages)
    n_rows = (n_voltages + n_cols - 1) // n_cols + 1  # +1 for comparison plot
    
    for i, voltage in enumerate(voltages_to_compare):
        try:
            # Get FEM data for this voltage
            fem_df = fem_data[voltage]
            fem_time = fem_df['Time/s'].values
            fem_thickness = fem_df['Filmthickness/m'].values
            
            # Determine time range for PINN predictions
            if max_time is None:
                t_max = fem_time.max()
            else:
                t_max = max_time
            
            # Create time points for PINN prediction
            time_points = np.linspace(0, t_max, len(fem_time))
            
            # Generate PINN predictions
            # Create input tensor: [time, voltage] for film thickness network
            # Based on the architecture, film thickness network takes (t, E) as input
            pinn_input = torch.tensor(np.column_stack([
                time_points,
                np.full(len(time_points), voltage)
            ]), dtype=torch.float32, device=networks.device)
            
            with torch.no_grad():
                # Get network outputs - should be a dictionary
                if hasattr(networks, 'networks'):
                    # If networks is a NetworkManager
                    film_net = networks.networks.get('film_thickness', None) or networks.networks.get('L', None)
                    if film_net is not None:
                        pinn_thickness = film_net(pinn_input).cpu().numpy().flatten()
                    else:
                        # Try the forward method which should return a dict
                        outputs = networks.forward(pinn_input)
                        if isinstance(outputs, dict):
                            pinn_thickness = outputs.get('film_thickness', outputs.get('L', None))
                            if pinn_thickness is not None:
                                pinn_thickness = pinn_thickness.cpu().numpy().flatten()
                            else:
                                print(f"Warning: Could not find film thickness output for voltage {voltage}")
                                print(f"Available outputs: {list(outputs.keys())}")
                                continue
                        else:
                            print(f"Warning: Unexpected output format for voltage {voltage}")
                            continue
                else:
                    # Direct network access
                    outputs = networks(pinn_input)
                    if isinstance(outputs, dict):
                        pinn_thickness = outputs.get('film_thickness', outputs.get('L', None))
                        if pinn_thickness is not None:
                            pinn_thickness = pinn_thickness.cpu().numpy().flatten()
                        else:
                            print(f"Warning: Could not find film thickness output")
                            continue
                    else:
                        pinn_thickness = outputs.cpu().numpy().flatten()
            
            # Plot individual comparison
            plt.subplot(n_rows, n_cols, i + 1)
            plt.plot(fem_time, fem_thickness * 1e9, 'b-', label='FEM', linewidth=2)
            plt.plot(time_points, pinn_thickness * 1e9, 'r--', label='PINN', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Film Thickness (nm)')
            plt.title(f'{voltage}V')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting voltage {voltage}: {e}")
            continue
    
    # Overall comparison plot
    plt.subplot(n_rows, 1, n_rows)
    colors = plt.cm.tab10(np.linspace(0, 1, len(voltages_to_compare)))
    
    for i, voltage in enumerate(voltages_to_compare):
        try:
            fem_df = fem_data[voltage]
            fem_time = fem_df['Time/s'].values
            fem_thickness = fem_df['Filmthickness/m'].values
            
            # Get PINN predictions (reuse logic from above)
            if max_time is None:
                t_max = fem_time.max()
            else:
                t_max = max_time
            
            time_points = np.linspace(0, t_max, len(fem_time))
            pinn_input = torch.tensor(np.column_stack([
                time_points,
                np.full(len(time_points), voltage)
            ]), dtype=torch.float32, device=networks.device)
            
            with torch.no_grad():
                if hasattr(networks, 'networks'):
                    film_net = networks.networks.get('film_thickness', None) or networks.networks.get('L', None)
                    if film_net is not None:
                        pinn_thickness = film_net(pinn_input).cpu().numpy().flatten()
                    else:
                        outputs = networks.forward(pinn_input)
                        if isinstance(outputs, dict):
                            pinn_thickness = outputs.get('film_thickness', outputs.get('L', None))
                            if pinn_thickness is not None:
                                pinn_thickness = pinn_thickness.cpu().numpy().flatten()
                            else:
                                continue
                        else:
                            continue
                else:
                    outputs = networks(pinn_input)
                    if isinstance(outputs, dict):
                        pinn_thickness = outputs.get('film_thickness', outputs.get('L', None))
                        if pinn_thickness is not None:
                            pinn_thickness = pinn_thickness.cpu().numpy().flatten()
                        else:
                            continue
                    else:
                        pinn_thickness = outputs.cpu().numpy().flatten()
            
            color = colors[i]
            plt.plot(fem_time, fem_thickness * 1e9, '-', color=color, 
                    label=f'FEM {voltage}V', linewidth=2)
            plt.plot(time_points, pinn_thickness * 1e9, '--', color=color, 
                    label=f'PINN {voltage}V', linewidth=2, alpha=0.8)
            
        except Exception as e:
            print(f"Error in overall plot for voltage {voltage}: {e}")
            continue
    
    plt.xlabel('Time (s)')
    plt.ylabel('Film Thickness (nm)')
    plt.title('Film Thickness Evolution: PINN vs FEM Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Film thickness comparison plot saved to: {save_path}")
    
    plt.show()

def compute_temporal_metrics(networks, physics, fem_data, voltages_to_compare=None):
    """
    Compute quantitative comparison metrics between PINN and FEM temporal data.
    
    Returns:
        dict: Comparison metrics for each voltage
    """
    if voltages_to_compare is None:
        voltages_to_compare = list(fem_data.keys())
    
    metrics = {}
    
    for voltage in voltages_to_compare:
        try:
            fem_df = fem_data[voltage]
            fem_time = fem_df['Time/s'].values
            fem_thickness = fem_df['Filmthickness/m'].values
            
            # Generate PINN predictions at FEM time points
            pinn_input = torch.tensor(np.column_stack([
                fem_time,
                np.full(len(fem_time), voltage)
            ]), dtype=torch.float32, device=networks.device)
            
            with torch.no_grad():
                if hasattr(networks, 'networks'):
                    film_net = networks.networks.get('film_thickness', None) or networks.networks.get('L', None)
                    if film_net is not None:
                        pinn_thickness = film_net(pinn_input).cpu().numpy().flatten()
                    else:
                        outputs = networks.forward(pinn_input)
                        if isinstance(outputs, dict):
                            pinn_thickness = outputs.get('film_thickness', outputs.get('L', None))
                            if pinn_thickness is not None:
                                pinn_thickness = pinn_thickness.cpu().numpy().flatten()
                            else:
                                continue
                        else:
                            continue
                else:
                    outputs = networks(pinn_input)
                    if isinstance(outputs, dict):
                        pinn_thickness = outputs.get('film_thickness', outputs.get('L', None))
                        if pinn_thickness is not None:
                            pinn_thickness = pinn_thickness.cpu().numpy().flatten()
                        else:
                            continue
                    else:
                        pinn_thickness = outputs.cpu().numpy().flatten()
                
                # Compute metrics
                mse = np.mean((pinn_thickness - fem_thickness)**2)
                mae = np.mean(np.abs(pinn_thickness - fem_thickness))
                rmse = np.sqrt(mse)
                
                # Relative error
                rel_error = np.mean(np.abs(pinn_thickness - fem_thickness) / (np.abs(fem_thickness) + 1e-12))
                
                # Correlation coefficient
                corr = np.corrcoef(pinn_thickness, fem_thickness)[0, 1]
                
                # R-squared
                ss_res = np.sum((fem_thickness - pinn_thickness) ** 2)
                ss_tot = np.sum((fem_thickness - np.mean(fem_thickness)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Final thickness comparison (steady-state)
                final_fem = fem_thickness[-1]
                final_pinn = pinn_thickness[-1]
                final_error = abs(final_pinn - final_fem) / final_fem * 100  # percentage
                
                metrics[voltage] = {
                    'MSE': mse,
                    'MAE': mae, 
                    'RMSE': rmse,
                    'Relative_Error': rel_error,
                    'Correlation': corr,
                    'R_squared': r2,
                    'Final_thickness_FEM_nm': final_fem * 1e9,
                    'Final_thickness_PINN_nm': final_pinn * 1e9,
                    'Final_thickness_error_%': final_error
                }
                    
        except Exception as e:
            print(f"Error computing metrics for voltage {voltage}: {e}")
    
    return metrics

def load_and_analyze(checkpoint_path, config_path, fem_data_dir=None, output_dir=None):
    # Load config
    cfg = OmegaConf.load(config_path)
   
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if output_dir is None:
        run_dir = Path(checkpoint_path).parent.parent
        output_dir = str(run_dir / "analysis")
   
    # Create trainer and load checkpoint
    run_dir = Path(checkpoint_path).parent.parent
    trainer = PINNTrainer(cfg, device, output_dir=str(run_dir))
    trainer.load_checkpoint(checkpoint_path)
   
    # Load checkpoint data for stats
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
   
    # Load FEM data if provided
    fem_data = {}
    if fem_data_dir:
        fem_data = load_fem_data(fem_data_dir)
        print(f"Loaded FEM data for {len(fem_data)} voltages: {sorted(list(fem_data.keys()))}")
   
    # Run analysis
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
   
    # Import specific functions to avoid the problematic analyze_training_results
    from analysis.analysis import (plot_training_losses, visualize_predictions,
                                  generate_polarization_curve, plot_ntk_weight_densities)
   
    # Generate individual plots
    if trainer.loss_history.get('total'):
        plot_training_losses(trainer.loss_history,
                           save_path=os.path.join(plots_dir, "training_losses.png"))
   
    # Film thickness temporal comparison with FEM data
    if fem_data:
        compare_film_thickness_curves(
            trainer.networks, trainer.physics, fem_data, step="final",
            save_path=os.path.join(plots_dir, "film_thickness_vs_fem_comparison.png")
        )
        
        # Compute and save comparison metrics
        metrics = compute_temporal_metrics(trainer.networks, trainer.physics, fem_data)
        if metrics:
            metrics_path = os.path.join(plots_dir, "fem_temporal_comparison_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write("PINN vs FEM Film Thickness Temporal Comparison Metrics\n")
                f.write("=" * 60 + "\n")
                f.write(f"{'Voltage (V)':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<8} {'Final Error (%)':<15}\n")
                f.write("-" * 80 + "\n")
                
                for voltage in sorted(metrics.keys()):
                    voltage_metrics = metrics[voltage]
                    f.write(f"{voltage:<12.1f} {voltage_metrics['MSE']:<12.2e} "
                           f"{voltage_metrics['MAE']:<12.2e} {voltage_metrics['RMSE']:<12.2e} "
                           f"{voltage_metrics['R_squared']:<8.4f} {voltage_metrics['Final_thickness_error_%']:<15.2f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Detailed Metrics by Voltage:\n")
                f.write("=" * 60 + "\n")
                
                for voltage, voltage_metrics in sorted(metrics.items()):
                    f.write(f"\nVoltage: {voltage}V\n")
                    f.write("-" * 20 + "\n")
                    for metric, value in voltage_metrics.items():
                        if isinstance(value, float):
                            if 'Error' in metric or 'MSE' in metric or 'MAE' in metric or 'RMSE' in metric:
                                f.write(f"{metric}: {value:.6e}\n")
                            else:
                                f.write(f"{metric}: {value:.6f}\n")
                        else:
                            f.write(f"{metric}: {value}\n")
            print(f"Temporal comparison metrics saved to: {metrics_path}")
    else:
        # Fall back to original visualization if no FEM data
        visualize_predictions(trainer.networks, trainer.physics, step="final",
                             save_path=os.path.join(plots_dir, "predictions_overview.png"))
   
    generate_polarization_curve(trainer.networks, trainer.physics,
                               save_path=os.path.join(plots_dir, "polarization_curve.png"))
   
    if trainer.ntk_manager:
        plot_ntk_weight_densities(trainer.ntk_manager, save_path=plots_dir)
   
    create_loss_landscape(trainer.networks, trainer.physics, trainer.sampler,
                         device=trainer.device, save_path=plots_dir)
   
    # Print basic checkpoint info
    print(f"Checkpoint step: {checkpoint.get('step', 'N/A')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'N/A')}")
    print(f"Analysis saved to: {plots_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_checkpoint.py <checkpoint_path> <config_path> [fem_data_dir] [output_dir]")
        print("  checkpoint_path: Path to the model checkpoint")
        print("  config_path: Path to the configuration file")
        print("  fem_data_dir: Optional directory containing FEM validation txt files (format: 'X.X V.txt')")
        print("  output_dir: Optional output directory for analysis results")
        sys.exit(1)
   
    checkpoint_path = 
    config_path = sys.argv[2]
    fem_data_dir = sys.argv[3] if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else None
   
    load_and_analyze(checkpoint_path, config_path, fem_data_dir, output_dir)