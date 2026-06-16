#!/usr/bin/env bash
# Inspect FEM loading: glob order, per-file row count, total points, and
# what torch.randperm(total, seed=0)[0] currently picks. This will tell
# us why OLD seed=0 → (76000s, 0.4V) but current seed=0 → (148000s, 0.1V).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

docker run --rm -i --init --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$REPO_DIR:/app" --workdir="/app" \
  "$DOCKER_IMAGE" bash -s <<'EOF'
pip install pandas -q
python - <<'PY'
from pathlib import Path
import pandas as pd, numpy as np, torch

fem_dir = Path("pinnacle/FEM")
txt_files = list(fem_dir.glob("*.txt"))
print("Glob order (as iterated):")
for f in txt_files:
    print(f"  {f.name}")

# Reproduce the exact loader concatenation
all_t, all_E, all_L, sizes = [], [], [], []
for txt_file in txt_files:
    voltage = float(txt_file.stem.split()[0])
    df = pd.read_csv(txt_file, sep='\t')
    t_vals = df['Time/s'].values
    E_vals = np.full(len(t_vals), voltage)
    L_vals = df['Filmthickness/m'].values
    valid = ~(np.isnan(t_vals) | np.isnan(E_vals) | np.isnan(L_vals))
    t_vals, E_vals, L_vals = t_vals[valid], E_vals[valid], L_vals[valid]
    all_t.append(t_vals); all_E.append(E_vals); all_L.append(L_vals)
    sizes.append((txt_file.name, len(t_vals)))
print("\nRows per file:")
for n, c in sizes:
    print(f"  {n}: {c}")
t_full = np.concatenate(all_t)
E_full = np.concatenate(all_E)
L_full = np.concatenate(all_L)
N = len(t_full)
print(f"\nTotal points: {N}")

for dev in ["cpu", "cuda"]:
    g = torch.Generator(device=dev).manual_seed(0)
    idx = torch.randperm(N, generator=g, device=dev)[0].item()
    print(f"\nDevice={dev}: randperm(N, seed=0)[0] = idx {idx}")
    print(f"  → t={t_full[idx]} s, E={E_full[idx]} V, L={L_full[idx]*1e9:.3f} nm")
PY
EOF
