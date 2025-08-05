#!/usr/bin/env python3

import torch
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics
from sampling.sampling import CollocationSampler
from training.training import PINNTrainer
from analysis.analysis import analyze_training_results, create_loss_landscape


def load_and_analyze(checkpoint_path, config_path, output_dir=None):
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Run analysis (minimal version to avoid stats errors)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Import specific functions to avoid the problematic analyze_training_results
    from analysis.analysis import (plot_training_losses, visualize_predictions, 
                                  generate_polarization_curve, plot_ntk_weight_densities)
    
    # Generate individual plots
    if trainer.loss_history.get('total'):
        plot_training_losses(trainer.loss_history, 
                           save_path=os.path.join(plots_dir, "training_losses.png"))
    
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
        print("Usage: python analyze_checkpoint.py <checkpoint_path> <config_path> [output_dir]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    config_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    load_and_analyze(checkpoint_path, config_path, output_dir)