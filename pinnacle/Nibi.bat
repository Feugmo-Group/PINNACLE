#!/bin/bash
#SBATCH --account=def-ctetsass
#SBATCH --job-name=PINNACLE_Training
#SBATCH --time=1:00:30
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=slurm_logs/base_increased_sampling_250h_%j.out
#SBATCH --error=slurm_logs/base_increased_sampling_250h_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Load modules
module load StdEnv/2023
module load python/3.12
module load cuda12.3

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/.installed" ]; then
  echo "Installing requirements..."
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install hydra-core omegaconf tqdm matplotlib numpy scipy
  touch venv/.installed
fi

# Set matplotlib backend for headless environment
export MPLBACKEND=Agg

# Run training with config overrides
echo "Starting training..."
python main.py batch_size.BC=5120 batch_size.interior=10240 batch_size.inference=5120 batch_size.IC=5120 batch_size.L=10240 experiment.name="base_increased_sampling_ntk_no_holes_no_sched_250_hours"

echo "End time: $(date)"