# main.py
"""
Main execution script for PINNACLE.

Simple script to demonstrate the complete workflow:
networks → physics → sampling → training → analysis 
"""
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from hydra.core.hydra_config import HydraConfig

# Import all our modules
from networks.networks import NetworkManager
from  physics.physics import  ElectrochemicalPhysics
from sampling.sampling import CollocationSampler
from training.training import PINNTrainer
from analysis.analysis import analyze_training_results
from analysis.analysis import create_loss_landscape
import numpy as np

torch.manual_seed(46)
np.random.seed(46)
if torch.cuda.is_available():
    torch.cuda.manual_seed(46)
    torch.cuda.manual_seed_all(46)


def run_training(config: DictConfig, device: torch.device) -> PINNTrainer:
    """
    Run full training process.

    Args:
        config: Hydra configuration
        device: PyTorch device

    Returns:
        Trained PINNTrainer instance
    """
    print(" Starting full PINNACLE training...")

    # Get output directory from Hydra
    output_dir = HydraConfig.get().runtime.output_dir
    print(f" Output directory: {output_dir}")

    # Create trainer
    trainer = PINNTrainer(config, device, output_dir=output_dir)

    # Run training
    loss_history = trainer.train()

    print(" Training completed successfully!")
    return trainer


def run_analysis(trainer: PINNTrainer, ntk_weight_manager: NetworkManager) -> None:
    """
    Run complete analysis of training results.

    Args:
        trainer: Trained PINNTrainer instance
    """
    print(" Running complete analysis...")

    # Get output directory
    output_dir = trainer.output_dir
    plots_dir = os.path.join(output_dir, "plots")

    # Generate all analysis plots
    analyze_training_results(trainer, ntk_weight_manager, save_dir=plots_dir)
    create_loss_landscape(trainer.networks,trainer.physics,trainer.sampler,device=trainer.device,save_path=plots_dir)

    print(f" Analysis complete! Plots saved to: {plots_dir}")


@hydra.main(config_path="../conf/", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main execution function for PINNACLE.

    **Complete PINNACLE Workflow:**

    1. **Setup**: Initialize device and display configuration
    2. **Components**: Test all individual components  
    3. **Training**: Run physics-informed neural network training
    4. **Analysis**: Generate comprehensive plots and analysis
    5. **Results**: Save all outputs for review

    This script demonstrates the complete pipeline from raw configuration
    to trained models and publication-ready plots.

    Args:
        cfg: Hydra configuration loaded from config.yaml
    """
    print("=" * 60)
    print(" PINNACLE: Physics-Informed Neural Networks")
    print("    Analyzing Corrosion Layers in Electrochemistry")
    print("=" * 60)

    # Apply config seed (overrides module-level default of 46)
    _seed = int(cfg.get("seed", 46))
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)

    # Precision — float32 is the fast default; float64 only for inverse problem
    _precision = str(cfg.get("precision", "float32")).lower()
    _dtype = torch.float64 if _precision == "float64" else torch.float32
    torch.set_default_dtype(_dtype)
    print(f"  Precision: {_precision}")

    # Flush denormals to zero: some random seeds drive intermediate values into
    # the subnormal range, where CPU float ops slow by ~40x (observed: seed 2 at
    # ~850 ms/step vs ~20 ms/step). Flushing keeps every seed fast.
    torch.set_flush_denormal(True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    if device.type == 'cuda':
        print(f" GPU: {torch.cuda.get_device_name()}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        _mem_frac = float(os.environ.get("CUDA_MEM_FRACTION", "1.0"))
        if _mem_frac < 1.0:
            torch.cuda.set_per_process_memory_fraction(_mem_frac)
            print(f" GPU memory capped at {_mem_frac*100:.0f}%")

    # Display configuration
    print("\n Configuration Overview:")
    print(f"  Training steps: {cfg.training.max_steps:,}")
    print(f"  Batch sizes: Interior={cfg.batch_size.interior}, BC={cfg.batch_size.BC}, IC={cfg.batch_size.IC}")
    print(f"  Learning rate: {cfg.optimizer.adam.lr}")
    print(f"  Architecture: {cfg.arch.potential.hidden_layers} layers, {cfg.arch.potential.layer_size} neurons")

    # Get output directory info
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"  Output directory: {output_dir}")


    # Full training mode
    print("\n" + "=" * 30)
    print(" FULL TRAINING MODE")
    print("=" * 30)

    # Step 1: Run training
    trainer = run_training(cfg, device)

    # Step 2: Run analysis
    run_analysis(trainer,trainer.ntk_manager)

    # Step 3: Print final summary
    print("\n" + "=" * 60)
    print(" PINNACLE EXECUTION COMPLETE!")
    print("=" * 60)

    stats = trainer.get_training_stats()
    print(f" Final loss: {stats['final_loss']:.6f}")
    print(f" Best loss: {stats['best_loss']:.6f}")
    print(f"⏱  Training time: {stats['training_time_minutes']:.1f} minutes")
    print(f" Parameters: {stats['total_parameters']:,}")

    print(f"\n Results saved to: {output_dir}")
    print(f"   Training plots: {os.path.join(output_dir, 'plots')}")
    print(f"   Checkpoints: {os.path.join(output_dir, 'checkpoints')}")

    print("\n Key files generated:")
    print(f"  • training_losses.png - Loss evolution plots")
    print(f"  • predictions_overview.png - Solution visualization")
    print(f"  • polarization_curve.png - Electrochemical analysis")
    print(f"  • best_model.pt - Best trained model")
    print(f"  • final_model.pt - Final model checkpoint")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run main function with Hydra
    main()