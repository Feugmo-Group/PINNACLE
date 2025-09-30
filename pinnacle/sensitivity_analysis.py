#!/usr/bin/env python3
# run_sensitivity_analysis.py
"""
Sensitivity analysis for hybrid PINN training with random FEM data points.
"""

import torch
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from datetime import datetime

from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics
from sampling.sampling import CollocationSampler
from training.training import PINNTrainer
from analyze import load_fem_data, compute_temporal_metrics

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def run_single_experiment(config: DictConfig, 
                         random_seed: int, 
                         experiment_name: str,
                         base_output_dir: str,
                         device: torch.device):
    """Run a single training with a specific random seed for data selection."""
    
    # Set seed for this experiment
    set_seed(random_seed)
    
    # Create experiment-specific output directory
    exp_output_dir = os.path.join(base_output_dir, f"seed_{random_seed}")
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Update config with experiment-specific settings
    config.hybrid.use_data = True
    config.hybrid.fem_batch_size = 1  # Single data point
    config.hybrid.random_seed = random_seed
    
    # Save config for this run
    config_path = os.path.join(exp_output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)
    
    print(f"\n{'='*60}")
    print(f"Running experiment with seed {random_seed}")
    print(f"Output directory: {exp_output_dir}")
    print(f"{'='*60}")
    
    # Create trainer
    trainer = PINNTrainer(config, device, output_dir=exp_output_dir)
    
    # Run training
    loss_history = trainer.train()
    
    # Save loss history
    loss_path = os.path.join(exp_output_dir, "loss_history.json")
    serializable_history = {k: [float(v) for v in vals] for k, vals in loss_history.items()}
    with open(loss_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    # Load FEM data for evaluation
    fem_data = load_fem_data(config.hybrid.fem_data_path)
    
    # Compute metrics
    metrics = compute_temporal_metrics(trainer.networks, trainer.physics, fem_data)
    
    # Save metrics
    metrics_path = os.path.join(exp_output_dir, "fem_comparison_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Get training statistics
    stats = trainer.get_training_stats()
    stats['random_seed'] = random_seed
    stats['metrics'] = metrics
    
    return stats

def aggregate_results(results_list, output_dir):
    """Aggregate results from multiple runs and compute statistics."""
    
    print(f"\n{'='*60}")
    print("AGGREGATE SENSITIVITY ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Collect metrics across all runs
    all_metrics = {}
    for voltage in results_list[0]['metrics'].keys():
        all_metrics[voltage] = {
            'RMSE': [],
            'MAE': [],
            'R_squared': [],
            'Final_thickness_error_%': []
        }
    
    for result in results_list:
        for voltage, metrics in result['metrics'].items():
            for key in all_metrics[voltage].keys():
                if key in metrics:
                    all_metrics[voltage][key].append(metrics[key])
    
    # Compute statistics
    statistics = {}
    for voltage in all_metrics.keys():
        statistics[voltage] = {}
        for metric_name, values in all_metrics[voltage].items():
            if values:
                statistics[voltage][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
    
    # Save aggregate statistics
    stats_path = os.path.join(output_dir, "aggregate_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    # Create summary report
    report_path = os.path.join(output_dir, "sensitivity_report.txt")
    with open(report_path, 'w') as f:
        f.write("HYBRID PINN SENSITIVITY ANALYSIS REPORT\n")
        f.write(f"{'='*80}\n")
        f.write(f"Number of experiments: {len(results_list)}\n")
        f.write(f"Random seeds tested: {[r['random_seed'] for r in results_list]}\n\n")
        
        f.write("AGGREGATE METRICS BY VOLTAGE\n")
        f.write(f"{'-'*80}\n")
        
        for voltage in sorted(statistics.keys()):
            f.write(f"\nVoltage: {voltage} V\n")
            f.write(f"{'-'*40}\n")
            
            for metric_name, stats in statistics[voltage].items():
                f.write(f"\n{metric_name}:\n")
                f.write(f"  Mean ± Std: {stats['mean']:.3e} ± {stats['std']:.3e}\n")
                f.write(f"  Min / Max: {stats['min']:.3e} / {stats['max']:.3e}\n")
                f.write(f"  Median: {stats['median']:.3e}\n")
        
        # Add variation analysis
        f.write(f"\n{'='*80}\n")
        f.write("VARIATION ANALYSIS\n")
        f.write(f"{'-'*80}\n")
        
        for voltage in sorted(statistics.keys()):
            f.write(f"\nVoltage {voltage} V - Coefficient of Variation:\n")
            for metric_name, stats in statistics[voltage].items():
                if stats['mean'] != 0:
                    cv = (stats['std'] / abs(stats['mean'])) * 100
                    f.write(f"  {metric_name}: {cv:.1f}%\n")
    
    print(f"Results saved to {output_dir}")
    return statistics

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function for sensitivity analysis."""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of random seeds to test')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting random seed')
    parser.add_argument('--output_base', type=str, 
                       default='sensitivity_analysis',
                       help='Base output directory name')
    args, unknown = parser.parse_known_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = f"{args.output_base}_{timestamp}"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Generate random seeds
    seeds = [args.start_seed + i for i in range(args.num_seeds)]
    
    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS CONFIGURATION")
    print(f"{'='*60}")
    print(f"Number of experiments: {args.num_seeds}")
    print(f"Random seeds: {seeds}")
    print(f"Output directory: {main_output_dir}")
    print(f"{'='*60}")
    
    # Run experiments
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{args.num_seeds}] Running experiment with seed {seed}")
        
        try:
            stats = run_single_experiment(
                cfg, 
                seed, 
                f"exp_{i+1}",
                main_output_dir,
                device
            )
            results.append(stats)
            
        except Exception as e:
            print(f"ERROR in experiment {i+1}: {e}")
            continue
    
    # Aggregate and analyze results
    if results:
        aggregate_results(results, main_output_dir)
    
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS COMPLETE!")
    print(f"Results saved to: {main_output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()