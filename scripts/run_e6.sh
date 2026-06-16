#!/usr/bin/env bash
# E6 — Inverse problem: jointly recover k3_0 and D_CV from sparse L(t).
# Sub-experiments E6-A through E6-F (see revision_plan.md).
# Paper placement: Sec. IV.G (headline result of revision)
#
# Uses ntk_l2 weighting (cheaper than full Jacobian NTK) and float64
# for numerical stability of the Butler-Volmer boundary terms.
#
# Optional env vars:
#   REPO_DIR     – repo root (default: auto-detected via git)
#   DOCKER_IMAGE – container image (default: nvcr.io/nvidia/pytorch:26.01-py3)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E6 — Inverse problem: k3_0 + D_CV recovery (15 runs)"
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
  --env CUDA_MEM_FRACTION="${CUDA_MEM_FRACTION:-0.5}" \
  "$DOCKER_IMAGE" \
  bash -s << 'CONTAINER_EOF'
set -uo pipefail

pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=50000
STRAT=ntk_l2

echo "================================================================"
echo " E6 — Inverse problem (${STEPS} steps, ${STRAT}, float64)"
echo " True k3_0 = 4.5e-9 mol/(m^2 s)  |  True D_CV = 1.0e-21 m^2/s"
echo "================================================================"

# E6-A: 5 obs, 0% noise, 0.1 V
echo "--- E6-A: N_obs=5, sigma=0, V=[0.1] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.0 \
    "inverse.obs_voltages=[0.1]" \
    experiment.name=e6_A_ntk_l2 || true

# E6-B: 5 obs, 5% noise, 0.1 V
echo "--- E6-B: N_obs=5, sigma=0.05, V=[0.1] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1]" \
    experiment.name=e6_B_ntk_l2 || true

# E6-C: 5 obs, 10% noise, 0.1 V
echo "--- E6-C: N_obs=5, sigma=0.10, V=[0.1] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.10 \
    "inverse.obs_voltages=[0.1]" \
    experiment.name=e6_C_ntk_l2 || true

# E6-D: 10 obs, 5% noise, 0.1+1.0 V
echo "--- E6-D: N_obs=10, sigma=0.05, V=[0.1,1.0] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=10 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1,1.0]" \
    experiment.name=e6_D_ntk_l2 || true

# E6-E: 20 obs, 5% noise, 0.1+1.0 V
echo "--- E6-E: N_obs=20, sigma=0.05, V=[0.1,1.0] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=20 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1,1.0]" \
    experiment.name=e6_E_ntk_l2 || true

# E6-F: ensemble UQ — 10 seeds
echo "--- E6-F: ensemble UQ (10 seeds) ---"
for SEED in $(seq 0 9); do
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat="${STRAT}" \
        training.max_steps="${STEPS}" \
        precision=float64 \
        inverse.enabled=true \
        inverse.n_obs=5 inverse.obs_noise_sigma=0.05 \
        "inverse.obs_voltages=[0.1]" \
        seed="${SEED}" \
        "experiment.name=e6_F_seed${SEED}_ntk_l2" || true
done

echo ""
echo "================================================================"
echo " E6 done. Outputs in outputs/experiments/e6_*_ntk_l2/"
echo " True k3_0 = 4.5e-9   True D_CV = 1.0e-21"
echo " Check loss_history['k3_0'] and loss_history['D_cv'] for recovery."
echo "================================================================"
CONTAINER_EOF
