#!/usr/bin/env python3
# generate_sensitivity_plots.py
"""Generate plots for sensitivity analysis results."""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_aggregate_stats(results_dir):
    """Load aggregate statistics from sensitivity analysis."""
    stats_path = Path(results_dir) / "aggregate_statistics.json"
    with open(stats_path, 'r') as f:
        return json.load(f)

def plot_metric_distributions(stats, output_dir):
    """Create box plots for each metric across voltages."""
    
    metrics = ['RMSE', 'MAE', 'R_squared', 'Final_thickness_error_%']
    voltages = sorted([float(v) for v in stats.keys()])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Collect data for box plot
        data = []
        labels = []
        
        for v in voltages:
            v_str = str(v)
            if v_str in stats and metric in stats[v_str]:
                # Create synthetic data from statistics
                mean = stats[v_str][metric]['mean']
                std = stats[v_str][metric]['std']
                # Generate approximate distribution
                synthetic = np.random.normal(mean, std, 100)
                data.append(synthetic)
                labels.append(f"{v}V")
        
        if data:
            bp = ax.boxplot(data, labels=labels)
            ax.set_title(f'{metric} Distribution Across Voltages')
            ax.set_xlabel('Voltage')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "metric_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_analysis(stats, output_dir):
    """Plot coefficient of variation for each metric."""
    
    voltages = sorted([float(v) for v in stats.keys()])
    metrics = ['RMSE', 'MAE', 'R_squared', 'Final_thickness_error_%']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric in metrics:
        cvs = []
        for v in voltages:
            v_str = str(v)
            if v_str in stats and metric in stats[v_str]:
                mean = stats[v_str][metric]['mean']
                std = stats[v_str][metric]['std']
                cv = (std / abs(mean)) * 100 if mean != 0 else 0
                cvs.append(cv)
        
        if cvs:
            ax.plot(voltages, cvs, marker='o', label=metric, linewidth=2)
    
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('Sensitivity Analysis: Variation in Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(Path(output_dir) / "coefficient_variation.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True,
                       help='Directory containing sensitivity analysis results')
    args = parser.parse_args()
    
    # Load statistics
    stats = load_aggregate_stats(args.results_dir)
    
    # Generate plots
    plot_metric_distributions(stats, args.results_dir)
    plot_cv_analysis(stats, args.results_dir)
    
    print(f"Plots saved to {args.results_dir}")

if __name__ == "__main__":
    main()