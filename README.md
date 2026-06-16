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

First install uv (a fast Python package manager) if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via pip / pipx / Homebrew
pip install uv            # or: pipx install uv  /  brew install uv
```

Then set up the project:

```bash
uv sync
```

This creates a virtual environment and installs everything pinned in `pyproject.toml`.
For a specific CUDA build of PyTorch, add the matching wheel index:

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu124   # or cu121, cu118, cpu
```

`uv run <command>` then runs inside that environment without activating it manually —
all commands below use `uv run`.

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

## Train a model

A single command trains the model with the defaults in `conf/config.yaml`:

```bash
uv run python -m pinnacle.main
```

Everything is configured through Hydra, so any setting can be overridden on the
command line as `key=value`:

```bash
# name the run, pick a weighting strategy, set the step budget, choose the device
uv run python -m pinnacle.main experiment.name=my_run training.weight_strat=ntk \
    training.max_steps=10000 device=cuda

# fast smoke test (small network, few steps) — finishes in a minute or two
uv run python -m pinnacle.main training.max_steps=2000 \
    arch.potential.layer_size=10 arch.potential.hidden_layers=3 device=cpu
```

Each run writes to `outputs/experiments/<experiment.name>/<timestamp>/`, containing
`checkpoints/` (including `best_model.pt`), `plots/` (film-thickness vs. FEM, loss
curves, etc.), the resolved Hydra config under `.hydra/`, and `pinnacle.log`.

Reproduce the main FEM-anchored hybrid result (NTK weighting + one FEM anchor):

```bash
uv run python -m pinnacle.main training.weight_strat=ntk hybrid.use_data=true \
    hybrid.anchor_t=150000 hybrid.anchor_E=0.1 training.max_steps=50000
```

The FEM reference data used as anchors live in `pinnacle/FEM/`.

---

## Reproducing the paper experiments

The `scripts/` folder holds the drivers for every experiment in the paper, plus the
tooling that turns the resulting runs into figures and tables. The experiment drivers
wrap training in Docker to match the paper's GPU environment, but Docker is **not
required** — every experiment is just a call to `python -m pinnacle.main` with Hydra
overrides, so you can reproduce any of them directly with `uv run` on your own machine
(see the W7 example below). The figure/analysis scripts are plain Python and run
locally once the experiments have finished.

**Experiment drivers** (`scripts/run_*.sh`):

| Script | What it produces |
|---|---|
| `run_e1.sh` | Loss-weighting ablation: NTK vs. uniform vs. batch-size (timing + GPU memory) |
| `run_e2.sh` | Per-interface boundary-condition residuals over training |
| `run_e3.sh` | Anchor robustness across random-anchor seeds |
| `run_e3b_positions.sh` | Anchor robustness across a systematic `(t,E)` position sweep |
| `run_e4e5_fixed.sh` | Data-efficiency sweep (number of FEM anchors) and noise sweep |
| `run_e5_n10.sh` | Noise robustness at a fixed budget of 10 anchors |
| `run_e6.sh` | Inverse problem: assimilate sparse `L(t)` and target `k3_0`, `D_CV` |
| `run_e7.sh` | Wall-clock cost comparison vs. FEM |
| `run_w7.sh` | Reproducibility across random seeds |
| `run_revision.sh` | Runs the full set of experiments in order |

**Figures and tables** (run locally after the experiments):

| Script | What it produces |
|---|---|
| `make_e4e5_figures.py` | Data-efficiency and noise-robustness figures |
| `make_e2_figure.py` | Boundary-residual figure |
| `aggregate_revision_results.py` | Summary tables (whole-curve relative-L2 errors) |
| `compute_table1.py` | Main results table |

```bash
# example: run the data-efficiency / noise sweep, then build the figures
bash scripts/run_e4e5_fixed.sh
uv run python scripts/make_e4e5_figures.py
```

### Reproduce the headline result without Docker (W7)

The multi-seed reproducibility check (W7) is the easiest end-to-end example: it trains
the NTK + single-random-anchor hybrid model across several seeds. Most seeds converge
to a few-percent whole-curve film-thickness error; the occasional divergence is the
initialisation-driven branch selection discussed in the paper. Run it directly with
`uv` — no container needed:

```bash
# train the hybrid model at four seeds (≈45 min/seed on a single GPU)
for SEED in 1 2 3 4; do
  uv run python -m pinnacle.main \
    training.weight_strat=ntk training.max_steps=50000 \
    hybrid.use_data=true precision=float64 seed=$SEED \
    experiment.name=hybrid_seed$SEED
done

# no GPU? add device=cpu (slower); for a quick smoke test drop max_steps to ~2000
```

Each run writes to `outputs/experiments/hybrid_seed<SEED>/<timestamp>/`, including
`plots/film_thickness_E*.png` (PINN vs. FEM at each voltage) and a `checkpoints/`
folder. The convenience driver `scripts/run_w7.sh` does the same across seeds inside
Docker if you prefer a pinned environment.

Every other experiment is the same `python -m pinnacle.main` call with different Hydra
overrides (the exact flags live at the top of each `run_*.sh`). To run them with `uv`
instead of Docker:

```bash
# E1 — loss-weighting ablation (repeat with uniform, batch_size)
uv run python -m pinnacle.main training.weight_strat=ntk training.max_steps=50000 \
  experiment.name=e1_ablation_ntk

# E2 — per-interface boundary-condition residual diagnostics
uv run python -m pinnacle.main training.weight_strat=ntk training.max_steps=50000 \
  precision=float64 experiment.name=e2_bc_residuals

# E3 — anchor-robustness sweep (vary seed=0..6)
uv run python -m pinnacle.main training.weight_strat=ntk hybrid.use_data=true \
  hybrid.anchor_mode=seed hybrid.anchor_seed=0 precision=float64 \
  experiment.name=e3a_seed0

# E6 — inverse problem: recover one parameter from sparse L(t)
uv run python -m pinnacle.main inverse.enabled=true 'inverse.unknown_params=[k2_0]' \
  'inverse.obs_voltages=[0.1]' training.max_steps=30000 experiment.name=e6_recover_k2_0
```

The directory also contains variant and exploratory drivers (`run_*_v2.sh`,
`probe_*.sh`, `run_test_*.sh`) kept for completeness; the table above lists the
canonical entry points.

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
uv run python -m pinnacle.main --cfg job
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
