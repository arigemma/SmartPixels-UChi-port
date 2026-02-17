#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=1:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-5   # number of runs - 1

# Define argument sets
ARGS=(
  # "-m towards-model-3 -o results/SLURM/2026_02_16/10783560/v3/towards-model-3/cross_entropy -lf cross_entropy"
  "-m towards-model-3 -o results/SLURM/2026_02_16/10783560/v3/towards-model-3/bce2 -lf bce"
  # "-m towards-model-2 -o results/SLURM/2026_02_16/10783560/v3/towards-model-2/cross_entropy -lf cross_entropy"
  "-m towards-model-2 -o results/SLURM/2026_02_16/10783560/v3/towards-model-2/bce2 -lf bce"
  # "-m towards-model-1 -o results/SLURM/2026_02_16/10783560/v3/towards-model-1/cross_entropy -lf cross_entropy"
  "-m towards-model-1 -o results/SLURM/2026_02_16/10783560/v3/towards-model-1/bce2 -lf bce"
)

# Run the command corresponding to this array index
python train.py -c config/10783560/v3.yml --balance-classes --device cuda -bs 1024 ${ARGS[$SLURM_ARRAY_TASK_ID]}
