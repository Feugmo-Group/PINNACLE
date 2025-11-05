# PINNACLE

**Physics-Informed Neural Networks Analyzing Corrosion Layers in Electrochemistry**

This repository contains the implementation of physics-informed neural networks (PINNs) for modeling electrochemical corrosion processes at the Metal-Film-Solution interface, along with extensive benchmarking experiments using various optimization algorithms.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Running the Code](#running-the-code)
  - [1. Jupyter Notebooks](#1-jupyter-notebooks)
  - [2. Main PINNACLE Training](#2-main-pinnacle-training)
  - [3. RLA-PINNs Benchmark Experiments](#3-rla-pinns-benchmark-experiments)
  - [4. Analysis and Visualization](#4-analysis-and-visualization)
- [Configuration](#configuration)
- [Expected Outputs](#expected-outputs)
- [Key Concepts](#key-concepts)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

**For Reviewers: Fastest Way to See Results**

```bash
# 1. Install dependencies
poetry install

# 2. Run a demonstration notebook (choose one):
jupyter notebook pinnacle/NTKPinnacle.ipynb      # NTK weighting demo
jupyter notebook pinnacle/ALPinnacle.ipynb       # Augmented Lagrangian demo
jupyter notebook pinnacle/pinnacle.ipynb         # General overview

# 3. Or run the full training pipeline:
python -m pinnacle.main
```

---

## Installation

### Requirements

- Python 3.11-3.13
- PyTorch ≥ 2.8.0
- CUDA-capable GPU (recommended but not required)

### Setup

**Option 1: Using Poetry (Recommended)**

```bash
# Clone the repository
git clone <repository-url>
cd PINNACLE

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

**Option 2: Using pip**

```bash
# Clone the repository
git clone <repository-url>
cd PINNACLE

# Install PyTorch (visit pytorch.org for platform-specific instructions)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy matplotlib tqdm hydra-core hessianfree wandb jupyter
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Repository Structure

```
PINNACLE/
├── pinnacle/                          # Main package
│   ├── main.py                       # Main entry point for PINNACLE training
│   ├── analyze.py                    # Post-training analysis script
│   ├── sweep.py                      # Hyperparameter sweep runner
│   ├── sensitivity_analysis.py       # Sensitivity analysis
│   │
│   ├── networks/networks.py          # Neural network architectures
│   ├── physics/physics.py            # Electrochemical PDE definitions
│   ├── losses/losses.py              # Loss function implementations
│   ├── weighting/weighting.py        # Loss weighting strategies (NTK, AL)
│   ├── sampling/sampling.py          # Collocation point sampling
│   ├── training/training.py          # Training orchestration
│   ├── gradients/gradients.py        # Automatic differentiation utilities
│   ├── analysis/analysis.py          # Visualization and analysis tools
│   │
│   ├── FEM/                          # Finite Element Method reference data
│   │   ├── 0.1V.txt                 # FEM solutions at various voltages
│   │   ├── 0.4V.txt
│   │   ├── 1.0V.txt
│   │   ├── 1.6V.txt
│   │   └── 1.8V.txt
│   │
│   ├── rla_pinns/                    # RLA-PINNs benchmark experiments
│   │   ├── train.py                 # Universal training script
│   │   ├── optim/                   # Optimizer implementations
│   │   │   ├── adam.py, sgd.py, lbfgs.py
│   │   │   ├── engd.py              # Ensemble NGD
│   │   │   ├── spring.py            # SPRING optimizer
│   │   │   ├── kfac.py              # K-FAC optimizer
│   │   │   └── hessianfree.py       # Hessian-free optimizer
│   │   │
│   │   ├── exp1_poisson5d/          # 19 experiment directories
│   │   ├── exp14_heat4d/            # (one for each benchmark)
│   │   └── ...                      # Each contains sweep configs
│   │
│   ├── NTKPinnacle.ipynb            # Jupyter notebooks (demonstrations)
│   ├── ALPinnacle.ipynb
│   ├── NTK+ALPinnacle.ipynb
│   ├── ENGDPinnacle.ipynb
│   └── pinnacle.ipynb
│
├── Misc/                             # Additional analysis notebooks
│   ├── pdm.ipynb                    # Reaction scheme chemistry details
│   ├── analysis.ipynb               # Post-training analysis
│   └── dimension_group_analysis.ipynb
│
├── conf/
│   └── config.yaml                  # Main Hydra configuration file
│
├── pyproject.toml                   # Poetry dependencies
└── README.md                        # This file
```

---

## Running the Code

### 1. Jupyter Notebooks

The easiest way to understand the methods is through the interactive notebooks. Each demonstrates a different aspect of the PINNACLE framework.

**Start Jupyter:**

```bash
jupyter notebook
```

**Key Notebooks:**

| Notebook | Description | Runtime |
|----------|-------------|---------|
| `pinnacle/NTKPinnacle.ipynb` | Demonstrates Neural Tangent Kernel (NTK) loss weighting for balanced training | ~10-15 min |
| `pinnacle/ALPinnacle.ipynb` | Augmented Lagrangian method for constraint satisfaction | ~10-15 min |
| `pinnacle/NTK+ALPinnacle.ipynb` | Combined NTK + AL approach for hybrid training | ~15-20 min |
| `pinnacle/ENGDPinnacle.ipynb` | Ensemble Natural Gradient Descent optimizer demonstration | ~15-20 min |
| `pinnacle/pinnacle.ipynb` | General PINNACLE workflow overview | ~10 min |
| `Misc/pdm.ipynb` | Detailed electrochemical reaction schemes and chemistry | ~5 min |
| `Misc/analysis.ipynb` | Post-training analysis and visualization examples | ~5 min |

**What Each Notebook Shows:**

- **NTKPinnacle.ipynb**:
  - How NTK weighting balances multiple loss components
  - Comparison of training dynamics with/without NTK
  - Convergence analysis

- **ALPinnacle.ipynb**:
  - Augmented Lagrangian penalty method
  - Hybrid training with FEM reference data
  - Constraint satisfaction monitoring

- **NTK+ALPinnacle.ipynb**:
  - Combined approach leveraging both methods
  - Enhanced performance on challenging regions
  - Best practices for hybrid physics-data training

- **ENGDPinnacle.ipynb**:
  - Advanced second-order optimization
  - Comparison with first-order methods (Adam, SGD)
  - Computational efficiency analysis

---

### 2. Main PINNACLE Training

Run the full electrochemical corrosion simulation with default settings:

```bash
# Basic training (uses conf/config.yaml)
python -m pinnacle.main

# Training with custom configuration
python -m pinnacle.main optimizer.lr=0.001 training.max_steps=50000

# Run on CPU (if no GPU available)
python -m pinnacle.main device=cpu
```

**Expected Runtime:** 1-3 hours on GPU, 6-12 hours on CPU (for 20,000 training steps)

**Outputs:** Results saved to `outputs/experiments/pinnacle/[timestamp]/`

**What This Does:**
1. Initializes neural networks for electric potential (φ) and concentrations (c_cv, c_av, c_h)
2. Trains using physics-informed loss (PDE residuals + boundary/initial conditions)
3. Applies NTK weighting for balanced multi-objective training
4. Saves checkpoints (`best_model.pt`, `final_model.pt`)
5. Generates visualizations (training curves, predictions, polarization curves)

---

### 3. RLA-PINNs Benchmark Experiments

These experiments test various optimizers on canonical PDEs (Poisson, Heat, Fokker-Planck equations).

#### Running Individual Experiments

```bash
cd pinnacle/rla_pinns

# Example: Train Poisson equation with Adam
python train.py --equation=poisson --model=mlp-tanh-64 --optimizer=Adam --lr=1e-3

# Example: Train Heat equation with ENGD
python train.py --equation=heat --model=mlp-tanh-128 --optimizer=ENGD --lr=1e-3

# Example: Train with SPRING optimizer
python train.py --equation=log_fokker_planck --model=mlp-tanh-64 --optimizer=SPRING
```

**Available Optimizers:**
- First-order: `Adam`, `SGD`
- Second-order: `LBFGS`, `HessianFree`, `KFAC`
- Advanced: `ENGD`, `SPRING`, `RNGD`

**Available Equations:**
- `poisson` - Poisson equation (various dimensions: 5D, 10D, 100D)
- `heat` - Heat equation (4D time-dependent)
- `log_fokker_planck` - Fokker-Planck equation

#### Running Full Experiment Sweeps

Each experiment directory contains pre-configured sweeps testing multiple optimizers:

```bash
# Navigate to specific experiment
cd pinnacle/rla_pinns/exp14_heat4d

# Generate sweep configurations (if not already present)
bash create_sweeps.sh

# Launch all sweeps (19 optimizer variants)
bash launch_sweeps.sh

# Monitor progress (if using Weights & Biases)
# Results logged to wandb dashboard
```

**Example Experiments:**

| Experiment | Equation | Dimension | Purpose |
|------------|----------|-----------|---------|
| `exp1_poisson5d` | Poisson | 5D | Low-dimensional baseline |
| `exp2_poisson10d` | Poisson | 10D | Medium-dimensional test |
| `exp3_poisson100d` | Poisson | 100D | High-dimensional scaling |
| `exp5_log_fokker_planck` | Fokker-Planck | Variable | Non-linear PDE |
| `exp14_heat4d` | Heat | 4D | Time-dependent PDE |
| `exp15_heat4d_fixed` | Heat | 4D | Fixed learning rate variant |

**Expected Runtime:** 30 min - 4 hours per optimizer (depending on complexity)

---

### 4. Analysis and Visualization

#### Post-Training Analysis

After training completes, analyze the results:

```bash
# Analyze a trained model
python pinnacle/analyze.py --checkpoint outputs/experiments/pinnacle/[timestamp]/checkpoints/best_model.pt

# Generate sensitivity analysis
python pinnacle/sensitivity_analysis.py

# Create sensitivity plots
python pinnacle/generate_sensitivity_plots.py
```

#### Visualizations Generated

The code automatically generates the following plots:

1. **Training Curves** (`training_losses.png`)
   - Total loss evolution
   - Individual loss components (PDE, BC, IC, film physics)
   - Smoothed curves with confidence intervals

2. **Solution Predictions** (`predictions_overview.png`)
   - Potential distribution φ(x,t)
   - Concentration profiles (c_cv, c_av, c_h)
   - Film thickness evolution L(t)
   - Comparison with FEM reference data (if available)

3. **Polarization Curves** (`polarization_curve.png`)
   - Current vs. voltage relationship
   - Comparison with experimental data

4. **Loss Landscape** (`loss_landscape.png`)
   - 2D cross-section of loss surface around trained model
   - Identifies local minima properties

5. **NTK Weight Evolution** (`ntk_weights.png`)
   - Adaptive weight trajectories during training
   - Shows how NTK balances different loss terms

6. **Residual Analysis** (`residuals_heatmap.png`)
   - Spatial distribution of PDE residuals
   - Identifies problematic regions

---

## Configuration

### Main Configuration File: `conf/config.yaml`

The Hydra configuration file controls all aspects of training. Key sections:

```yaml
# Neural Network Architecture
architecture:
  layer_size: 20              # Neurons per hidden layer
  num_hidden_layers: 5        # Depth of network
  activation: swish           # Activation function (swish, tanh, relu, swoosh)

# Optimizer Settings
optimizer:
  lr: 0.001                   # Learning rate
  type: adam                  # Optimizer type
  scheduler:
    enabled: false            # Learning rate scheduling

# Sampling Strategy
sampling:
  adaptive: true              # Use adaptive sampling
  hybrid_ratio_uniform: 0.6   # 60% uniform, 40% adaptive
  n_interior: 8000            # Interior collocation points
  n_boundary: 2000            # Boundary points
  n_initial: 1500             # Initial condition points
  n_film: 1000                # Film physics points

# Training Configuration
training:
  max_steps: 20000            # Total training steps
  weighting_method: ntk       # Loss weighting (ntk, al, uniform)
  checkpoint_freq: 1000       # Checkpoint saving frequency

# Physical Parameters
physics:
  transport:
    D_cv: 1.0e-14             # Diffusion coefficients
    D_av: 1.0e-14
    D_h: 1.0e-14
  temperature: 298.15         # Temperature (K)

# Hybrid Training (FEM data integration)
hybrid:
  use_fem_data: true          # Enable hybrid training
  fem_data_dir: pinnacle/FEM  # FEM reference data directory
  al_penalty: 500.0           # Augmented Lagrangian penalty parameter
```

### Modifying Configuration

**Command-line overrides:**

```bash
# Change learning rate
python -m pinnacle.main optimizer.lr=0.0005

# Increase training steps
python -m pinnacle.main training.max_steps=50000

# Disable NTK weighting
python -m pinnacle.main training.weighting_method=uniform

# Change network architecture
python -m pinnacle.main architecture.layer_size=40 architecture.num_hidden_layers=8

# Multiple overrides
python -m pinnacle.main optimizer.lr=0.001 training.max_steps=30000 device=cuda
```

**Creating custom config files:**

```yaml
# conf/my_experiment.yaml
defaults:
  - config

optimizer:
  lr: 0.0005
training:
  max_steps: 50000
architecture:
  layer_size: 40
```

```bash
python -m pinnacle.main --config-name=my_experiment
```

---

## Expected Outputs

### Directory Structure After Training

```
outputs/experiments/pinnacle/[timestamp]/
├── .hydra/
│   ├── config.yaml              # Full resolved configuration
│   ├── hydra.yaml              # Hydra runtime config
│   └── overrides.yaml          # Command-line overrides
│
├── checkpoints/
│   ├── best_model.pt           # Best model (lowest validation loss)
│   └── final_model.pt          # Final model after all training steps
│
├── plots/
│   ├── training_losses.png     # Loss evolution curves
│   ├── predictions_overview.png # Solution visualizations
│   ├── polarization_curve.png  # Electrochemical analysis
│   ├── loss_landscape.png      # Loss surface topology
│   ├── ntk_weights.png         # Adaptive weight evolution
│   └── residuals_heatmap.png   # Spatial residual distribution
│
└── pinnacle.log                # Training log file
```

### Checkpoint Contents

Each checkpoint (`.pt` file) contains:

```python
{
    'model_state_dict': {...},      # Network parameters
    'optimizer_state_dict': {...},  # Optimizer state
    'step': 15000,                  # Training step
    'loss': 0.00234,               # Total loss value
    'config': {...},               # Full configuration
    'physics_params': {...},       # Physical constants
}
```

**Loading a checkpoint:**

```python
import torch

checkpoint = torch.load('outputs/experiments/pinnacle/[timestamp]/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Key Concepts

### Physics-Informed Neural Networks (PINNs)

PINNs embed physical laws (PDEs) directly into the loss function:

```
Total Loss = w₁·L_PDE + w₂·L_BC + w₃·L_IC + w₄·L_film
```

Where:
- **L_PDE**: Residual of governing PDEs (Poisson-Nernst-Planck system)
- **L_BC**: Boundary condition violations
- **L_IC**: Initial condition violations
- **L_film**: Film physics constraints

### Neural Tangent Kernel (NTK) Weighting

Automatically balances loss components based on their training dynamics:

```
w_i(t) ∝ 1 / ||∇_θ L_i||
```

Prevents gradient domination by any single loss term. See: *Wang et al., "Understanding and mitigating gradient flow pathologies in physics-informed neural networks" (SIAM J. Sci. Comput., 2021)*

### Augmented Lagrangian (AL) Method

Enforces constraints (e.g., FEM data matching) using penalty + multiplier approach:

```
L_AL = L_PINN + β/2·||g(x)||² + λᵀg(x)
```

Where:
- β: Penalty parameter (increases constraint enforcement)
- λ: Lagrange multipliers (updated adaptively)
- g(x): Constraint violations

### Adaptive Sampling

Dynamically allocates collocation points to high-error regions:

1. Evaluate PDE residuals on large base set (e.g., 10,000 points)
2. Select top-k highest residual points (e.g., 2,000)
3. Combine with uniform random points (hybrid strategy)
4. Update every N steps

Improves efficiency by focusing computation on difficult regions.

### Hybrid Training

Combines physics-based learning (PINNs) with data-driven learning (FEM reference):

```
L_total = L_PINN + L_AL(FEM_data)
```

Advantages:
- Physics constraints prevent overfitting
- FEM data guides solution toward known behavior
- Better generalization than pure data-driven methods

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch sizes in config.yaml
sampling:
  n_interior: 4000  # Reduce from 8000
  n_boundary: 1000  # Reduce from 2000

# Or use CPU
python -m pinnacle.main device=cpu
```

**2. Training Diverges (NaN losses)**

```bash
# Reduce learning rate
python -m pinnacle.main optimizer.lr=0.0001

# Enable gradient clipping (add to config.yaml)
optimizer:
  clip_grad_norm: 1.0
```

**3. Very Slow Training**

```bash
# Disable adaptive sampling temporarily
python -m pinnacle.main sampling.adaptive=false

# Reduce sampling points
python -m pinnacle.main sampling.n_interior=2000
```

**4. Import Errors**

```bash
# Ensure you're in the PINNACLE root directory
cd /path/to/PINNACLE

# Activate poetry environment
poetry shell

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/PINNACLE"
```

**5. Hydra Configuration Errors**

```bash
# Verify config syntax
python -m pinnacle.main --help

# See full resolved config without running
python -m pinnacle.main --cfg job
```

### Performance Tips

**For Faster Iteration:**
- Use fewer training steps: `training.max_steps=5000`
- Reduce network size: `architecture.layer_size=10 architecture.num_hidden_layers=3`
- Disable expensive analysis: Comment out plotting code in `main.py`

**For Better Accuracy:**
- Increase sampling density: `sampling.n_interior=16000`
- Use deeper networks: `architecture.num_hidden_layers=8`
- Enable NTK weighting: `training.weighting_method=ntk`
- Add FEM hybrid training: `hybrid.use_fem_data=true`

**For Large-Scale Experiments:**
- Use W&B for experiment tracking: Set `WANDB_PROJECT` environment variable
- Leverage multi-GPU: Modify `main.py` for `DataParallel`
- Batch process sweeps: Use `launch_sweeps.sh` in experiment directories

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pinnacle2024,
  title={PINNACLE: Physics-Informed Neural Networks Analyzing Corrosion Layers in Electrochemistry},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [contact email]

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
