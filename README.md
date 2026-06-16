# PINNACLE

**P**hysics-**I**nformed **N**eural **N**etworks **A**nalyzing **C**orrosion **L**ayers in **E**lectrochemistry.

PINNACLE solves the electrochemical Point Defect Model (PDM) for passive oxide-film
growth using physics-informed neural networks, with NTK adaptive loss weighting and
optional FEM-anchored hybrid training.

---

## Installation

Requires **Python 3.11–3.13** and, ideally, a CUDA-capable GPU (CPU works too, just slower).

```bash
git clone https://github.com/Feugmo-Group/PINNACLE.git
cd PINNACLE
```

### With uv (recommended)

```bash
uv sync
```

This creates a virtual environment and installs everything pinned in `pyproject.toml`.
For a specific CUDA build of PyTorch, add the matching wheel index:

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu124   # or cu121, cu118, cpu
```

Then run commands with `uv run`, e.g. `uv run python -m pinnacle.main`.

### With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Dependencies

Installed automatically by the commands above:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | neural networks and autograd |
| `numpy`, `scipy` | numerics |
| `hydra-core`, `omegaconf`, `hydra-colorlog` | configuration |
| `matplotlib` | plots |
| `pandas` | data handling |
| `tqdm` | progress bars |
| `hessianfree` | second-order optimizer (research module) |

---

## Quick start

```bash
# default run (reads conf/config.yaml)
python -m pinnacle.main

# override any config value on the command line (Hydra syntax)
python -m pinnacle.main training.max_steps=10000 training.weight_strat=uniform device=cpu

# quick smoke test (small network, few steps)
python -m pinnacle.main training.max_steps=2000 arch.potential.layer_size=10 arch.potential.hidden_layers=3
```

Outputs go to `outputs/experiments/<name>/<timestamp>/` and contain checkpoints,
plots, and a Hydra config snapshot.

Reproduce the main FEM-anchored hybrid result:

```bash
python -m pinnacle.main training.weight_strat=ntk hybrid.use_data=true \
    hybrid.anchor_t=150000 hybrid.anchor_E=0.1 training.max_steps=50000
```

The FEM reference data used as anchors live in `pinnacle/FEM/`.

---

## Repository layout

```
conf/              Hydra configuration (config.yaml)
pinnacle/
  main.py          Entry point
  networks/        Segregated feed-forward networks (potential, CV, AV, L)
  physics/         PDE residuals and physical constants
  losses/          Loss terms (interior, BC, IC, film growth, data)
  weighting/       NTK, uniform, and batch-size weight managers
  sampling/        Collocation and adaptive samplers
  training/        Training loop
  analysis/        Post-training plots and diagnostics
  FEM/             Reference FEM solutions at five voltages
  rla_pinns/       Research reference: second-order optimizers (not used in main loop)
```

---

## Configuration

All parameters live in `conf/config.yaml` and can be overridden on the command line.
Common knobs:

| Parameter | Default | Effect |
|---|---|---|
| `training.weight_strat` | `ntk` | `ntk` / `uniform` / `batch_size` |
| `training.max_steps` | 20000 | Training steps |
| `hybrid.use_data` | `true` | Include FEM anchor in the loss |
| `hybrid.anchor_t` | 76000 | Anchor time (s) |
| `hybrid.anchor_E` | 0.4 | Anchor voltage (V) |
| `arch.potential.layer_size` | 20 | Neurons per hidden layer |
| `arch.potential.hidden_layers` | 5 | Network depth |
| `optimizer.adam.lr` | 1e-3 | Learning rate |

Print the resolved config without training:

```bash
python -m pinnacle.main --cfg job
```

---

## Citation

```bibtex
@article{farooqi2025pinnacle,
  title   = {Stiff Multiphysics in Physics-Informed Neural Networks:
             Failure Modes and a Stabilization Framework},
  author  = {Farooqi, Mohid and B{\"o}sing, Ingmar and
             Tetsassi Feugmo, Conrard Giresse},
  journal = {APL Machine Learning},
  year    = {2025},
}
```

---

## License

MIT
