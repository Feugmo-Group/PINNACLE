#!/usr/bin/env bash
# E6 — Inverse problem: recover k3_0 from sparse L(t) observations
# Sub-experiments E6-A through E6-F (see revision_plan.md).
# Paper placement: new Sec. IV.G (headline result of revision)
#
# All sub-runs execute inside a single Docker container (install once).
#
# Optional env vars:
#   REPO_DIR     – absolute path to repo root  (default: auto-detected via git)
#   DOCKER_IMAGE – container image to use      (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E6 — Inverse problem: rate-constant identification (15 runs)"
echo " REPO_DIR     : $REPO_DIR"
echo " DOCKER_IMAGE : $DOCKER_IMAGE"
echo "================================================================"

docker run --rm -i --init \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" \
  --workdir="/app" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -euo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=100000

echo "================================================================"
echo " E6 — Inverse problem: rate-constant identification"
echo "================================================================"

# E6-A: 5 observations, 0% noise, 0.1 V
echo "--- E6-A: N_obs=5, sigma=0, V=[0.1] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        precision=float64 \
    +experiments=e6_inverse \
    training.max_steps="${STEPS}" \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.0 \
    "inverse.obs_voltages=[0.1]" \
    experiment.name=e6_A

# E6-B: 5 observations, 5% noise, 0.1 V
echo "--- E6-B: N_obs=5, sigma=0.05, V=[0.1] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        precision=float64 \
    +experiments=e6_inverse \
    training.max_steps="${STEPS}" \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1]" \
    experiment.name=e6_B

# E6-C: 5 observations, 10% noise, 0.1 V
echo "--- E6-C: N_obs=5, sigma=0.10, V=[0.1] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        precision=float64 \
    +experiments=e6_inverse \
    training.max_steps="${STEPS}" \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.10 \
    "inverse.obs_voltages=[0.1]" \
    experiment.name=e6_C

# E6-D: 10 observations, 5% noise, 0.1+1.0 V
echo "--- E6-D: N_obs=10, sigma=0.05, V=[0.1,1.0] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        precision=float64 \
    +experiments=e6_inverse \
    training.max_steps="${STEPS}" \
    inverse.n_obs=10 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1,1.0]" \
    experiment.name=e6_D

# E6-E: 20 observations, 5% noise, 0.1+1.0 V
echo "--- E6-E: N_obs=20, sigma=0.05, V=[0.1,1.0] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        precision=float64 \
    +experiments=e6_inverse \
    training.max_steps="${STEPS}" \
    inverse.n_obs=20 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1,1.0]" \
    experiment.name=e6_E

# E6-F: ensemble (10 seeds) — uncertainty quantification
echo "--- E6-F: ensemble UQ (10 seeds) ---"
for SEED in $(seq 0 9); do
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        precision=float64 \
        +experiments=e6_inverse \
        training.max_steps="${STEPS}" \
        inverse.n_obs=5 inverse.obs_noise_sigma=0.05 \
        "inverse.obs_voltages=[0.1]" \
        seed="${SEED}" \
        "experiment.name=e6_F_seed${SEED}"
done

echo ""
echo "================================================================"
echo " E6 done. Check recovered k3_0 in each run's timing.json / plots."
echo " True k3_0 = 4.5e-9 mol/(m^2 s)"
echo "================================================================"
CONTAINER_EOF
