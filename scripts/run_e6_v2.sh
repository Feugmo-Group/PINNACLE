#!/usr/bin/env bash
# E6-v2 — Inverse problem with two-stage training fix.
#
# Root cause of E6-v1 failure: obs_loss = ||L_network(t,E) - L_obs||² has
# zero gradient w.r.t. k3_0/D_cv (L_network doesn't depend on them).
#
# Fix: two-stage training (inverse.stage2_start_step=40000):
#   Stage 1 (0-40k): networks learn to fit FEM data + physics.
#   Stage 2 (40k-50k): networks frozen, only k3_0/D_cv optimised via physics losses.
#   The physics losses (film_physics, boundary_cv) depend on k3_0/D_cv directly,
#   so Stage 2 drives them toward the true values that satisfy the frozen network.
#
# Also adds more observation voltages (3 voltages) for better identifiability.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"

echo "================================================================"
echo " E6-v2 — Inverse problem (two-stage fix)"
echo " REPO_DIR : $REPO_DIR"
echo "================================================================"

docker run --rm -i --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" \
  --volume="$REPO_DIR:/app" --workdir="/app" \
  --env CUDA_MEM_FRACTION="${CUDA_MEM_FRACTION:-0.5}" \
  "$DOCKER_IMAGE" bash -s << 'CONTAINER_EOF'
set -uo pipefail
pip install hydra-core omegaconf scipy matplotlib tqdm hessianfree pandas hydra-colorlog -q

STEPS=50000
STRAT=ntk_l2
STAGE2=40000   # freeze networks at step 40k, last 10k steps = Stage 2
LR_PARAMS=1e-3 # higher lr for Stage 2 convergence (10x vs v1's 1e-4)

echo "True k3_0=4.5e-9  True D_cv=1.0e-21  Stage2 starts at step ${STAGE2}"

# E6-A-v2: 5 obs, 0% noise, 3 voltages
echo "--- E6-A-v2: N_obs=5, sigma=0, V=[0.1,0.4,1.0] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.0 \
    "inverse.obs_voltages=[0.1,0.4,1.0]" \
    inverse.lr_params="${LR_PARAMS}" \
    inverse.stage2_start_step="${STAGE2}" \
    experiment.name=e6_A_v2 || true

# E6-B-v2: 5 obs, 5% noise, 3 voltages
echo "--- E6-B-v2: N_obs=5, sigma=0.05, V=[0.1,0.4,1.0] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=5 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1,0.4,1.0]" \
    inverse.lr_params="${LR_PARAMS}" \
    inverse.stage2_start_step="${STAGE2}" \
    experiment.name=e6_B_v2 || true

# E6-C-v2: 10 obs, 5% noise, 3 voltages
echo "--- E6-C-v2: N_obs=10, sigma=0.05, V=[0.1,0.4,1.0] ---"
PYTHONPATH=/app/pinnacle python -m pinnacle.main \
    training.weight_strat="${STRAT}" \
    training.max_steps="${STEPS}" \
    precision=float64 \
    inverse.enabled=true \
    inverse.n_obs=10 inverse.obs_noise_sigma=0.05 \
    "inverse.obs_voltages=[0.1,0.4,1.0]" \
    inverse.lr_params="${LR_PARAMS}" \
    inverse.stage2_start_step="${STAGE2}" \
    experiment.name=e6_C_v2 || true

# E6-F-v2: ensemble UQ — 5 seeds
echo "--- E6-F-v2: ensemble UQ (5 seeds) ---"
for SEED in $(seq 0 4); do
    PYTHONPATH=/app/pinnacle python -m pinnacle.main \
        training.weight_strat="${STRAT}" \
        training.max_steps="${STEPS}" \
        precision=float64 \
        inverse.enabled=true \
        inverse.n_obs=5 inverse.obs_noise_sigma=0.05 \
        "inverse.obs_voltages=[0.1,0.4,1.0]" \
        inverse.lr_params="${LR_PARAMS}" \
        inverse.stage2_start_step="${STAGE2}" \
        seed="${SEED}" \
        "experiment.name=e6_F_v2_seed${SEED}" || true
done

echo "E6-v2 done. Results in outputs/experiments/e6_*_v2/"
CONTAINER_EOF
