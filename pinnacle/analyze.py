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
                
                fem_data[voltage] = df.dropna().reset_index(drop=True)
                print(f"Loaded FEM data for {voltage}V: {len(df)} time points")
                
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
                    
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
    
    return fem_data

def compare_film_thickness_curves(networks, physics, fem_data, step="final", 
                                save_path=None, voltages_to_compare=None):
    """Bare bones FEM vs PINN comparison"""
    import matplotlib.pyplot as plt

    # Scales
    lc = 1e-9
    F = 96485
    R = 8.3145
    T = 293
    D_cv = 1.0e-21
    phic = (R*T)/F
    tc = (lc ** 2) / D_cv

    if voltages_to_compare is None:
        voltages_to_compare = sorted(list(fem_data.keys()))
        
   
    for voltage in voltages_to_compare:

        # Get FEM data
        fem_df = fem_data[voltage]
        fem_time_s = fem_df['Time/s'].values
        fem_thickness_m = fem_df['Filmthickness/m'].values
        
        # Non-dimensionalize inputs for PINN
        time_hat = fem_time_s / tc

        voltage_hat = voltage / phic
        
        # PINN input
        pinn_input = torch.tensor(np.column_stack([
            time_hat,
            np.full(len(time_hat), voltage_hat)
        ]), dtype=torch.float32, device=networks.device)
        
        # Get PINN prediction
        with torch.no_grad():
            if hasattr(networks, 'networks'):
                film_net = networks.networks.get('L', networks.networks.get('film_thickness', None))
                pinn_L_hat = film_net(pinn_input).cpu().numpy().flatten()
            else:
                pinn_L_hat = networks(pinn_input).cpu().numpy().flatten()
        
        # Convert to nm
        pinn_thickness_nm = pinn_L_hat 
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(fem_time_s/3600, fem_thickness_m*1e9, 'b-', label='FEM', linewidth=2)
        plt.plot(fem_time_s/3600, pinn_thickness_nm, 'r--', label='PINN', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('Film Thickness (nm)')
        plt.title(f'Film Thickness at {voltage}V')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(f"{save_path}_{voltage}.png", dpi=300, bbox_inches='tight')
    

def compute_temporal_metrics(networks, physics, fem_data, voltages_to_compare=None):
    """Compute quantitative comparison metrics between PINN and FEM temporal data"""
    
    # Scales
    lc = 1e-9
    F = 96485
    R = 8.3145
    T = 293
    D_cv = 1.0e-21
    phic = (R*T)/F
    tc = (lc ** 2) / D_cv
    
    if voltages_to_compare is None:
        voltages_to_compare = list(fem_data.keys())
   
    metrics = {}
   
    for voltage in voltages_to_compare:
        fem_df = fem_data[voltage]
        fem_time = fem_df['Time/s'].values
        fem_thickness = fem_df['Filmthickness/m'].values
        
        # Non-dimensionalize inputs
        time_hat = fem_time / tc
        voltage_hat = voltage / phic
        
        # PINN input
        pinn_input = torch.tensor(np.column_stack([
            time_hat,
            np.full(len(time_hat), voltage_hat)
        ]), dtype=torch.float32, device=networks.device)
        
        # Get PINN prediction
        with torch.no_grad():
            if hasattr(networks, 'networks'):
                film_net = networks.networks.get('L', networks.networks.get('film_thickness', None))
                pinn_L_hat = film_net(pinn_input).cpu().numpy().flatten()
            else:
                pinn_L_hat = networks(pinn_input).cpu().numpy().flatten()
        
        # Convert to dimensional
        pinn_thickness = pinn_L_hat * lc
        
        # Compute metrics
        mse = np.mean((pinn_thickness - fem_thickness)**2)
        mae = np.mean(np.abs(pinn_thickness - fem_thickness))
        rmse = np.sqrt(mse)
        rel_error = np.mean(np.abs(pinn_thickness - fem_thickness) / (np.abs(fem_thickness) + 1e-12))
        corr = np.corrcoef(pinn_thickness, fem_thickness)[0, 1]
        
        ss_res = np.sum((fem_thickness - pinn_thickness) ** 2)
        ss_tot = np.sum((fem_thickness - np.mean(fem_thickness)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        final_fem = fem_thickness[-1]
        final_pinn = pinn_thickness[-1]
        final_error = abs(final_pinn - final_fem) / final_fem * 100
        
        metrics[voltage] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Relative_Error': rel_error,
            'Correlation': corr,
            'R_squared': r2,
            'Final_thickness_FEM_nm': final_fem * 1e9,
            'Final_thickness_PINN_nm': str(final_pinn * 1e9),
            'Final_thickness_error_%': str(final_error)
        }
   
    return metrics

def get_hybrid_data_info(cfg, checkpoint):
    """Extract hybrid training data point information if available"""
    hybrid_info = {}
    
    # Check if hybrid training was used
    if hasattr(cfg, 'hybrid') and cfg.hybrid.get('use_data', False):
        hybrid_info['enabled'] = True
        hybrid_info['batch_size'] = cfg.hybrid.get('fem_batch_size', 'N/A')
        hybrid_info['random_seed'] = cfg.hybrid.get('random_seed', 'N/A')
        
        # Try to get actual data point from checkpoint if stored
    if 'hybrid_data_point' in checkpoint:
            dp = checkpoint['hybrid_data_point']
            hybrid_info['data_point'] = {
                't': float(dp['t'][0]) if len(dp['t']) > 0 else 'N/A',
                'E': float(dp['E'][0]) if len(dp['E']) > 0 else 'N/A', 
                'L': float(dp['L'][0]) if len(dp['L']) > 0 else 'N/A'
            }
    else:
        hybrid_info['enabled'] = False

    return hybrid_info

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

    # Get hybrid training info
    hybrid_info = get_hybrid_data_info(cfg, checkpoint)
    if hybrid_info['enabled']:
        print(f"\n=== Hybrid Training Info ===")
        print(f"Data points used: {hybrid_info['batch_size']}")
        print(f"Random seed: {hybrid_info['random_seed']}")
        if 'data_point' in hybrid_info:
            print(f"Data point: {hybrid_info['data_point']}")
        print("=" * 30 + "\n")
   
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
                                  generate_polarization_curve, potential_investigations)
   
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
                # Add hybrid info if available
                if hybrid_info['enabled']:
                    f.write(f"Hybrid Training: Yes (Seed: {hybrid_info['random_seed']}, Points: {hybrid_info['batch_size']})\n")
                    f.write(f"Data Point: {hybrid_info['data_point']} \n" if 'data_point' in hybrid_info else "")
                else:
                    f.write("Hybrid Training: No (Pure PINN)\n")
                f.write("=" * 60 + "\n")
                f.write(f"{'Voltage (V)':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<8} {'Final Error (%)':<15}\n")
                f.write("-" * 80 + "\n")
                for voltage in sorted(metrics.keys()):
                    voltage_metrics = metrics[voltage]
                    
                    try:
                        f.write(f"{float(voltage):<12.1f} {float(voltage_metrics['MSE']):<12.2e} "
                            f"{float(voltage_metrics['MAE']):<12.2e} {float(voltage_metrics['RMSE']):<12.2e} "
                            f"{float(voltage_metrics['R_squared']):<8.4f} {float(voltage_metrics['Final_thickness_error_%']):<15.2f}\n")
                    except (ValueError, TypeError) as e:
                        # Handle NaN or invalid values
                        f.write(f"{voltage}: MSE={voltage_metrics.get('MSE', 'N/A')} "
                            f"MAE={voltage_metrics.get('MAE', 'N/A')} "
                            f"RMSE={voltage_metrics.get('RMSE', 'N/A')} "
                            f"R²={voltage_metrics.get('R_squared', 'N/A')} "
                            f"Final Error={voltage_metrics.get('Final_thickness_error_%', 'N/A')}%\n")    

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
    
    potential_investigations(trainer.networks, trainer.physics,cfg,
                             save_path=plots_dir)
   
    #create_loss_landscape(trainer.networks, trainer.physics, trainer.sampler,
    #                   device=trainer.device, save_path=plots_dir)
   
    # Print basic checkpoint info
    print(f"Checkpoint step: {checkpoint.get('step', 'N/A')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'N/A')}")
    print(f"Analysis saved to: {plots_dir}")

if __name__ == "__main__":

    checkpoint_path = "/home/mohidfarooqi/PINNACLE/sensitivity_analysis_best/seed_46/checkpoints/best_model.pt"
    config_path = "sensitivity_analysis_best/seed_46/config.yaml"
    fem_data_dir = "/home/mohidfarooqi/PINNACLE/pinnacle/FEM"
    output_dir = "/home/mohidfarooqi/PINNACLE/pinnacle/FEM"
   
    load_and_analyze(checkpoint_path, config_path, fem_data_dir, output_dir)