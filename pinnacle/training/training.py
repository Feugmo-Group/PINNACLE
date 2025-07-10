# training/training.py
"""
Training orchestration for PINNACLE with integrated NTK weighting.

Class-based approach for managing the complex state and lifecycle
of physics-informed neural network training with automatic loss balancing.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import os
import time

from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics
from sampling.sampling import CollocationSampler
from losses.losses import compute_total_loss
from weighting.weighting import (
    NTKWeightManager,
    setup_ntk_weighting,
    create_loss_weights
)
torch.manual_seed(995) 

class PINNTrainer:
    """
    Physics-Informed Neural Network trainer for electrochemical systems with NTK weighting.

    **Training Process:**

    1. **Initialization Phase:**
       - Initialize networks, physics, and sampling
       - Setup optimizer and learning rate scheduler
       - Configure loss weighting strategy (uniform, batch_size, manual, or NTK)

    2. **Training Phase:**
       - Sample collocation points (interior, boundary, initial, film physics)
       - Compute physics-informed losses using governing equations
       - Update NTK weights periodically (if using NTK strategy)
       - Perform gradient descent optimization
       - Track progress and save checkpoints

    3. **Monitoring Phase:**
       - Log detailed loss breakdowns
       - Save best models automatically
       - Generate training curves and diagnostics

    This class manages all the complex state needed for PINN training while
    providing a clean, simple interface for research use with automatic loss balancing.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, output_dir: Optional[str] = None):
        """
        Initialize the PINN trainer with configurable loss weighting.

        Args:
            config: Complete configuration dictionary
            device: PyTorch device for computation
            output_dir: Directory for saving outputs and checkpoints
        """
        self.config = config
        self.device = device
        self.output_dir = output_dir or "outputs"

        # Initialize core components
        print("📦 Initializing PINNACLE components...")
        self.networks = NetworkManager(config, device)
        self.physics = ElectrochemicalPhysics(config, device)
        self.sampler = CollocationSampler(config, self.physics, device)

        # Setup optimization
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_step = 0
        self.best_loss = float('inf')
        self.best_checkpoint_path = None

        # Setup directories
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Loss history tracking
        self.loss_history = {
            'total': [], 'interior': [], 'boundary': [], 'initial': [], 'film_physics': [],
            'weighted_cv_pde': [], 'weighted_av_pde': [], 'weighted_h_pde': [], 'weighted_poisson_pde': [],
            'weighted_cv_ic': [], 'weighted_av_ic': [], 'weighted_poisson_ic': [], 'weighted_h_ic': [],
            'weighted_L_ic': [],
            'weighted_cv_mf_bc': [], 'weighted_av_mf_bc': [], 'weighted_u_mf_bc': [],
            'weighted_cv_fs_bc': [], 'weighted_av_fs_bc': [], 'weighted_u_fs_bc': [], 'weighted_h_fs_bc': []
        }

        # Training configuration
        self.max_steps = config['training']['max_steps']
        self.print_freq = config['training']['rec_results_freq']
        self.save_freq = config['training']['save_network_freq']
        self.current_weighting_mode = ""
        self.ntk_steps = self.config.training.ntk_steps
        # Setup loss weighting strategy
        self._setup_loss_weighting()

        # Training statistics
        self.start_time = None
        self.total_params = sum(p.numel() for p in self.networks.get_all_parameters() if p.requires_grad)

        print(f"✅ Initialization complete!")
        print(f"📊 Total parameters: {self.total_params:,}")
        print(f"⚖️  Loss weighting strategy: {self.weighting_strategy}")

    def _setup_loss_weighting(self):
        """Setup the loss weighting strategy based on configuration."""
        self.weighting_strategy = self.config.training.weight_strat 

        if self.weighting_strategy == 'ntk':
            # Setup NTK weight manager
            print("🧠 Setting up NTK-based automatic loss weighting...")
            self.ntk_manager = setup_ntk_weighting(
                self.networks,
                self.physics,
                self.sampler,
                self.config
            )
            # Start with uniform weights until first NTK update
            self.loss_weights = create_loss_weights(self.config)
            self.ntk_weights = None  # Will be set when NTK weights are computed
        elif self.weighting_strategy == 'hybrid_ntk_batch':
            self.ntk_steps = self.config['training']['ntk_steps']
            self.current_weighting_mode = 'ntk'
            self.ntk_manager = setup_ntk_weighting(self.networks, self.physics, self.sampler, self.config)
            self.loss_weights = create_loss_weights(self.config)
        else:
            # Use static weighting strategy
            self.loss_weights = create_loss_weights(self.config)
            self.ntk_manager = None
            self.ntk_weights = None

        print(f"📊 Initial loss weights: {self.loss_weights}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create and configure the optimizer."""
        params = self.networks.get_all_parameters()
        optimizer_config = self.config['optimizer']['adam']

        optimizer = optim.AdamW(
            params,
            lr=optimizer_config['lr'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )

        # Set initial_lr for scheduler compatibility
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create and configure the learning rate scheduler."""
        scheduler_config = self.config['scheduler']

        if scheduler_config['type'] == "RLROP":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_config['RLROP']['factor'],
                patience=scheduler_config['RLROP']['patience'],
                threshold=scheduler_config['RLROP']['threshold'],
                min_lr=scheduler_config['RLROP']['min_lr'],
            )
        elif scheduler_config['type'] == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config['tf_exponential_lr']['decay_rate'],
                last_epoch=scheduler_config['tf_exponential_lr']['decay_steps']
            )
        else:
            return None

    def sample_training_points(self) -> Tuple[torch.Tensor, ...]:
        """
        Sample all collocation points needed for one training step.

        Returns:
            Tuple of all sampled points for loss computation
        """
        # Sample different types of points
        x_interior, t_interior, E_interior = self.sampler.sample_interior_points(self.networks)
        x_boundary, t_boundary, E_boundary = self.sampler.sample_boundary_points(self.networks)
        x_initial, t_initial, E_initial = self.sampler.sample_initial_points(self.networks)
        t_film, E_film = self.sampler.sample_film_physics_points()

        return (x_interior, t_interior, E_interior,
                x_boundary, t_boundary, E_boundary,
                x_initial, t_initial, E_initial,
                t_film, E_film)

    def _update_loss_weights(self):
        """Update loss weights using the configured strategy."""
        if self.weighting_strategy == 'ntk' or (self.weighting_strategy == "hybrid_ntk_batch" and self.current_step < self.ntk_steps) and self.ntk_manager is not None:
            # Update NTK weights if needed
            updated_weights = self.ntk_manager.update_weights(self.current_step)
            if updated_weights is not None:
                # Store NTK weights for passing to compute_total_loss
                self.ntk_weights = updated_weights
        elif self.weighting_strategy == "hybrid_ntk_batch":
            if self.current_step == self.ntk_steps:
                print(f"🔄 SWITCHING: NTK → Batch Size Weighting at step {self.current_step}")
                self.current_weighting_mode = 'batch_size'
                self.ntk_weights = None  # Clear NTK weights
                # Set batch size weights
                batch_config = self.config.get('batch_size', {})
                self.loss_weights = {
                    'interior': 1.0 / batch_config.get('interior', 1),
                    'boundary': 1.0 / batch_config.get('BC', 1), 
                    'initial': 1.0 / batch_config.get('IC', 1),
                    'film_physics': 1.0 / batch_config.get('L', 1)
                }
                print(f"📊 New batch size weights: {self.loss_weights}")
                return 
        else:
            # Clear NTK weights if not using NTK strategy
            self.ntk_weights = None

    def compute_losses(self) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses for current step with dynamic weighting.

        Returns:
            Dictionary of all loss components
        """
        # Update weights if using dynamic strategy
        self._update_loss_weights()

        # Sample training points
        (x_interior, t_interior, E_interior,
         x_boundary, t_boundary, E_boundary,
         x_initial, t_initial, E_initial,
         t_film, E_film) = self.sample_training_points()

        # Compute all losses with current weights
        if self.ntk_weights is not None:
            # Use NTK weights (granular component weighting)
            loss_dict = compute_total_loss(
                x_interior, t_interior, E_interior,
                x_boundary, t_boundary, E_boundary,
                x_initial, t_initial, E_initial,
                t_film, E_film,
                self.networks, self.physics,
                weights=None,  # Don't use standard weights
                ntk_weights=self.ntk_weights  # Use NTK component weights
            )
        else:
            # Use standard weights (uniform, batch_size, manual)
            loss_dict = compute_total_loss(
                x_interior, t_interior, E_interior,
                x_boundary, t_boundary, E_boundary,
                x_initial, t_initial, E_initial,
                t_film, E_film,
                self.networks, self.physics,
                weights=self.loss_weights,
                ntk_weights=None
            )

        return loss_dict

    def training_step(self) -> Dict[str, float]:
        """
        Perform one complete training step with automatic weight balancing.

        Returns:
            Dictionary of loss values (as floats)
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Compute losses (includes weight updates)
        loss_dict = self.compute_losses()

        # Backward pass
        total_loss = loss_dict['total']
        total_loss.backward()

        # Optimizer step
        self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss)
            else:
                self.scheduler.step()

        # Convert tensors to floats for logging
        loss_dict_float = {k: v.item() if torch.is_tensor(v) else v
                           for k, v in loss_dict.items()}

        # Update training state
        self.current_step += 1

        return loss_dict_float

    def update_loss_history(self, loss_dict: Dict[str, float]) -> None:
        """Update the loss history tracking."""
        for key in self.loss_history.keys():
            if key in loss_dict:
                self.loss_history[key].append(loss_dict[key])
            else:
                self.loss_history[key].append(0.0)  # Default for missing keys

    def print_progress(self, loss_dict: Dict[str, float]) -> None:
        """Print detailed training progress including weight information."""
        if self.current_step % self.print_freq == 0:
            print(f"\n=== Step {self.current_step} ===")
            if self.weighting_strategy == 'hybrid_ntk_batch':
                if self.current_step < self.ntk_steps:
                    phase = f"NTK Phase ({self.current_step}/{self.ntk_steps})"
                else:
                    batch_step = self.current_step - self.ntk_steps
                    phase = f"Batch Phase ({batch_step}/{self.max_steps-self.ntk_steps})"
                    print(f"\n=== Step {self.current_step} - {phase} ===")
            else:
                print(f"\n=== Step {self.current_step} ===")
            print(f"Total Loss: {loss_dict['total']:.6f}")
            print(f"Interior: {loss_dict['interior']:.6f} | Boundary: {loss_dict['boundary']:.6f} | "
                  f"Initial: {loss_dict['initial']:.6f} | Film Physics: {loss_dict['film_physics']:.6f}")

            # Show current weights
            if self.weighting_strategy == 'ntk' or self.weighting_strategy == "hybrid_ntk_batch" and self.ntk_weights is not None:
                print(f"NTK weights: CV={self.ntk_weights.get('cv_pde', 1.0):.3f}, "
                      f"AV={self.ntk_weights.get('av_pde', 1.0):.3f}, "
                      f"H={self.ntk_weights.get('h_pde', 1.0):.3f}, "
                      f"Poisson={self.ntk_weights.get('poisson_pde'):.3f}, "
                      f"BC={self.ntk_weights.get('boundary'):.3f}, "
                      f"IC={self.ntk_weights.get('initial'):.3f}, "
                      f"Film={self.ntk_weights.get('film_physics'):.10f}")
            elif self.weighting_strategy != 'uniform':
                print(f"Standard weights: Interior={self.loss_weights['interior']:.3f}, "
                      f"Boundary={self.loss_weights['boundary']:.3f}, "
                      f"Initial={self.loss_weights['initial']:.3f}, "
                      f"Film={self.loss_weights['film_physics']:.3f}")

            # PDE breakdown
            if 'weighted_cv_pde' in loss_dict:
                print("\nPDE Residuals:")
                print(f"  CV: {loss_dict['weighted_cv_pde']:.6f} | AV: {loss_dict['weighted_av_pde']:.6f}")
                print(f"  Hole: {loss_dict['weighted_h_pde']:.6f} | Poisson: {loss_dict['weighted_poisson_pde']:.6f}")

            # Boundary breakdown
            if 'weighted_cv_mf_bc' in loss_dict:
                print("\nBoundary Conditions:")
                print(
                    f"  m/f - CV: {loss_dict['weighted_cv_mf_bc']:.6f} | AV: {loss_dict['weighted_av_mf_bc']:.6f} | U: {loss_dict['weighted_u_mf_bc']:.6f}")
                print(
                    f"  f/s - CV: {loss_dict['weighted_cv_fs_bc']:.6f} | AV: {loss_dict['weighted_av_fs_bc']:.6f} | U: {loss_dict['weighted_u_fs_bc']:.6f} | H: {loss_dict['weighted_h_fs_bc']:.6f}")

            # Initial conditions breakdown
            if 'weighted_cv_ic' in loss_dict:
                print("\nInitial Conditions:")
                print(
                    f"  CV: {loss_dict['weighted_cv_ic']:.6f} | AV: {loss_dict['weighted_av_ic']:.6f} | H: {loss_dict['weighted_h_ic']:.6f}")
                print(f"  Poisson: {loss_dict['weighted_poisson_ic']:.6f} | L: {loss_dict['weighted_L_ic']:.6f}")

    def save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save current training state to checkpoint.

        Args:
            checkpoint_name: Name for the checkpoint file
        """
        checkpoint = {
            'networks': self.networks.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'step': self.current_step,
            'loss': self.loss_history['total'][-1] if self.loss_history['total'] else float('inf'),
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'loss_weights': self.loss_weights,
            'weighting_strategy': self.weighting_strategy,
            'ntk_weights': self.ntk_weights  # Save current NTK weights
        }

        # Save NTK manager state if using NTK
        if self.ntk_manager is not None:
            checkpoint['ntk_current_weights'] = self.ntk_manager.get_current_weights()
            checkpoint['ntk_batch_sizes'] = self.ntk_manager.optimal_batch_sizes

        save_path = os.path.join(self.checkpoints_dir, f"{checkpoint_name}.pt")
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.networks.load_state_dict(checkpoint['networks'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])


    def handle_checkpointing(self, current_loss: float) -> None:
        """Handle automatic checkpointing logic."""
        # Save best model
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_checkpoint_path = os.path.join(self.checkpoints_dir, "best_model")
            self.save_checkpoint("best_model")

        # Periodic checkpoints
        if self.current_step % self.save_freq == 0 and self.current_step > 0:
            self.save_checkpoint(f"model_step_{self.current_step}")

    def train(self, manual_loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
        """
        Run the complete training process with automatic loss balancing.

        **Training Algorithm with NTK:**

        For each training step:
        1. Sample collocation points from domain
        2. Update NTK weights periodically (if using NTK strategy)
        3. Compute physics-informed loss function with current weights
        4. Perform gradient descent optimization
        5. Update learning rate schedule
        6. Track progress and save checkpoints

        Args:
            manual_loss_weights: Optional manual override for loss weights

        Returns:
            Complete loss history for analysis
        """
        print(f"🚀 Starting PINNACLE training for {self.max_steps} steps...")
        print(f"⚖️  Using {self.weighting_strategy} loss weighting strategy")

        # Start timing
        self.start_time = time.time()

        # Main training loop
        for step in tqdm(range(self.max_steps), desc="Training Progress"):
            # Perform training step (includes automatic weight updates)
            loss_dict = self.training_step()

            # Update tracking
            self.update_loss_history(loss_dict)

            # Print progress
            self.print_progress(loss_dict)

            # Handle checkpointing
            current_loss = loss_dict['total']
            self.handle_checkpointing(current_loss)

        # Training completed
        self._finish_training()

        return self.loss_history

    def _finish_training(self) -> None:
        """Complete training and print summary."""
        elapsed_time = time.time() - self.start_time
        final_loss = self.loss_history['total'][-1] if self.loss_history['total'] else float('inf')

        print(f"\n✅ Training completed!")
        print(f"📊 Final loss: {final_loss:.6f}")
        print(f"🏆 Best loss: {self.best_loss:.6f}")
        print(f"⏱️  Training time: {elapsed_time / 60:.1f} minutes")
        print(f"🎯 Training speed: {self.max_steps / (elapsed_time / 60):.1f} steps/minute")

        if self.weighting_strategy == 'ntk' and self.ntk_weights is not None:
            print(f"⚖️  Final NTK weights: {self.ntk_weights}")
        else:
            print(f"⚖️  Final weights: {self.loss_weights}")

        # Save final checkpoint
        self.save_checkpoint("final_model")

        # Load best model for inference
        if self.best_checkpoint_path:
            print(f"\n🔄 Loading best checkpoint for inference...")
            self.load_checkpoint(f"{self.best_checkpoint_path}.pt")
            print(f"✅ Using best model (loss: {self.best_loss:.6f})")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the current model.

        Returns:
            Dictionary of evaluation metrics
        """
        self.networks.eval()

        with torch.no_grad():
            loss_dict = self.compute_losses()
            loss_dict_float = {k: v.item() if torch.is_tensor(v) else v
                               for k, v in loss_dict.items()}

        self.networks.train()
        return loss_dict_float

    def set_loss_weights(self, weights: Dict[str, float]) -> None:
        """Set loss weights manually (overrides automatic weighting)."""
        self.loss_weights = weights
        print(f"📊 Updated loss weights: {weights}")

    def get_ntk_diagnostics(self) -> Dict[str, Any]:
        """Get NTK weighting diagnostics if using NTK strategy."""
        if self.ntk_manager is None:
            return {"strategy": self.weighting_strategy, "ntk_active": False}

        return {
            "strategy": self.weighting_strategy,
            "ntk_active": True,
            "current_weights": self.ntk_manager.get_current_weights(),
            "optimal_batch_sizes": self.ntk_manager.optimal_batch_sizes,
            "last_update_step": self.ntk_manager.last_update_step,
            "update_frequency": self.ntk_manager.config.training.ntk_update_freq
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.loss_history['total']:
            return {"status": "No training completed"}

        stats = {
            "current_step": self.current_step,
            "total_steps": self.max_steps,
            "final_loss": self.loss_history['total'][-1],
            "best_loss": self.best_loss,
            "total_parameters": self.total_params,
            "training_time_minutes": (time.time() - self.start_time) / 60 if self.start_time else None,
            "loss_history_length": len(self.loss_history['total']),
            "weighting_strategy": self.weighting_strategy,
            "current_weights": self.loss_weights
        }

        # Add NTK diagnostics if available
        stats.update(self.get_ntk_diagnostics())

        # Add current NTK weights if using NTK
        if self.ntk_weights is not None:
            stats["current_ntk_weights"] = self.ntk_weights

        return stats