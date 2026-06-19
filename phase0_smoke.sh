#!/bin/bash
#SBATCH --job-name=phase0_smoke
#SBATCH --output=gpu_%j.out
#SBATCH --error=gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1               # <-- REQUIRED to actually get a GPU
#SBATCH --time=00:05:00

# Activate conda the reliable way inside a batch job:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion

echo "Running on $(hostname)"
nvidia-smi                         # shows the GPU Slurm gave you
python phase0_smoke.py
