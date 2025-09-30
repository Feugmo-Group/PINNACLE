#!/bin/bash
#SBATCH --account=def-ctetsass
#SBATCH --job-name=PINNACLE_Sensitivity
#SBATCH --time=23:00:30
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mu2faroo@uwaterloo.ca
#SBATCH --output=sensitivity_%j.out
#SBATCH --error=sensitivity_%j.err

# Load modules
module load StdEnv/2023
module load python/3.12
module load cuda/12.3

# Setup environment
source venv/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

pip install pandas 

# Configuration
RESULTS_BASE="sensitivity_${SLURM_JOB_ID}"
mkdir -p $RESULTS_BASE

# Run multiple seeds sequentially
for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running with seed ${SEED}..."
    
    OUTPUT_DIR="${RESULTS_BASE}/seed_${SEED}"
    
    python main.py \
        hydra.run.dir=${OUTPUT_DIR} \
        training.max_steps=20000 \
        training.weight_strat="ntk" \
        hybrid.use_data=true \
        hybrid.fem_batch_size=1 \
        hybrid.random_seed=${SEED} \
        hybrid.fem_data_path="pinnacle/FEM"
    
    echo "Completed seed ${SEED}"
done

echo "All runs completed"