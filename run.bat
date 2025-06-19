#!/bin/bash
#SBATCH --job-name=pinn_training
#SBATCH --partition=sbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j_%Y%m%d_%H%M%S.out    # %x=job name, %j=job ID, date/time
#SBATCH --error=slurm_logs/%x_%j_%Y%m%d_%H%M%S.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Load modules
module load anaconda3
module load cuda12.3

# Activate existing environment
conda activate pinn_env

# Run training
echo "Starting training..."
python nexPinnacle.py

echo "End time: $(date)"