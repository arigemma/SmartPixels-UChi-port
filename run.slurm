#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-7   # number of runs - 1

# Define argument sets
ARGS=(
  "-m towards-model-2-brevitas-w5a10 -o results/SLURM/2026_04_19/towards-model-2-brevitas-w5a10"
  "-m towards-model-2-brevitas-w4a8 -o results/SLURM/2026_04_19/towards-model-2-brevitas-w4a8"
  "-m towards-model-2-brevitas-w2a6 -o results/SLURM/2026_04_19/towards-model-2-brevitas-w2a6"
  "-m towards-model-2-brevitas-w2a4 -o results/SLURM/2026_04_19/towards-model-2-brevitas-w2a4"
  "-m towards-model-3-brevitas-w5a10 -o results/SLURM/2026_04_19/towards-model-3-brevitas-w5a10"
  "-m towards-model-3-brevitas-w4a8 -o results/SLURM/2026_04_19/towards-model-3-brevitas-w4a8"
  "-m towards-model-3-brevitas-w2a6 -o results/SLURM/2026_04_19/towards-model-3-brevitas-w2a6"
  "-m towards-model-3-brevitas-w2a4 -o results/SLURM/2026_04_19/towards-model-3-brevitas-w2a4"
)

# Run the command corresponding to this array index
python train.py -c config/10783560/v3.yml --balance-classes --device cuda -bs 128 -lf cross_entropy -lr 0.01 ${ARGS[$SLURM_ARRAY_TASK_ID]}
