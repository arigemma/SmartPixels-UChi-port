#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=1:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-11   # number of runs - 1

# Define argument sets
ARGS=(
  # "-m lgn-dense-1-small-2 -o results/SLURM/2026_02_17/lgn-dense-1-small-2"
  # "-m lgn-dense-1-medium-2 -o results/SLURM/2026_02_17/lgn-dense-1-medium-2"
  # "-m lgn-dense-1-large-2 -o results/SLURM/2026_02_17/lgn-dense-1-large-2"
  # "-m lgn-dense-1-huge-2 -o results/SLURM/2026_02_17/lgn-dense-1-huge-2"
  # "-m lgn-dense-1-small-5 -o results/SLURM/2026_02_17/lgn-dense-1-small-5"
  # "-m lgn-dense-1-medium-5 -o results/SLURM/2026_02_17/lgn-dense-1-medium-5"
  # "-m lgn-dense-1-large-5 -o results/SLURM/2026_02_17/lgn-dense-1-large-5"
  # "-m lgn-dense-1-huge-5 -o results/SLURM/2026_02_17/lgn-dense-1-huge-5"
  # "-m lgn-dense-1-small-10 -o results/SLURM/2026_02_17/lgn-dense-1-small-10"
  # "-m lgn-dense-1-medium-10 -o results/SLURM/2026_02_17/lgn-dense-1-medium-10"
  # "-m lgn-dense-1-large-10 -o results/SLURM/2026_02_17/lgn-dense-1-large-10"
  # "-m lgn-dense-1-huge-10 -o results/SLURM/2026_02_17/lgn-dense-1-huge-10"
  # "-m lgn-dense-2-small-2 -o results/SLURM/2026_02_17/lgn-dense-2-small-2"
  # "-m lgn-dense-2-medium-2 -o results/SLURM/2026_02_17/lgn-dense-2-medium-2"
  # "-m lgn-dense-2-large-2 -o results/SLURM/2026_02_17/lgn-dense-2-large-2"
  # "-m lgn-dense-2-huge-2 -o results/SLURM/2026_02_17/lgn-dense-2-huge-2"
  # "-m lgn-dense-2-small-5 -o results/SLURM/2026_02_17/lgn-dense-2-small-5"
  # "-m lgn-dense-2-medium-5 -o results/SLURM/2026_02_17/lgn-dense-2-medium-5"
  # "-m lgn-dense-2-large-5 -o results/SLURM/2026_02_17/lgn-dense-2-large-5"
  # "-m lgn-dense-2-huge-5 -o results/SLURM/2026_02_17/lgn-dense-2-huge-5"
  # "-m lgn-dense-2-small-10 -o results/SLURM/2026_02_17/lgn-dense-2-small-10"
  # "-m lgn-dense-2-medium-10 -o results/SLURM/2026_02_17/lgn-dense-2-medium-10"
  # "-m lgn-dense-2-large-10 -o results/SLURM/2026_02_17/lgn-dense-2-large-10"
  # "-m lgn-dense-2-huge-10 -o results/SLURM/2026_02_17/lgn-dense-2-huge-10"
  "-m lgn-dense-1-small-20 -o results/SLURM/2026_02_17/lgn-dense-1-small-20"
  "-m lgn-dense-1-medium-20 -o results/SLURM/2026_02_17/lgn-dense-1-medium-20"
  "-m lgn-dense-1-large-20 -o results/SLURM/2026_02_17/lgn-dense-1-large-20"
  "-m lgn-dense-1-huge-20 -o results/SLURM/2026_02_17/lgn-dense-1-huge-20"
  "-m lgn-dense-1-small-50 -o results/SLURM/2026_02_17/lgn-dense-1-small-50"
  "-m lgn-dense-1-medium-50 -o results/SLURM/2026_02_17/lgn-dense-1-medium-50"
  "-m lgn-dense-1-large-50 -o results/SLURM/2026_02_17/lgn-dense-1-large-50"
  "-m lgn-dense-1-huge-50 -o results/SLURM/2026_02_17/lgn-dense-1-huge-50"
  "-m lgn-dense-2-small-20 -o results/SLURM/2026_02_17/lgn-dense-2-small-20"
  "-m lgn-dense-2-medium-20 -o results/SLURM/2026_02_17/lgn-dense-2-medium-20"
  "-m lgn-dense-2-large-20 -o results/SLURM/2026_02_17/lgn-dense-2-large-20"
  "-m lgn-dense-2-huge-20 -o results/SLURM/2026_02_17/lgn-dense-2-huge-20"
  "-m lgn-dense-2-small-50 -o results/SLURM/2026_02_17/lgn-dense-2-small-50"
  "-m lgn-dense-2-medium-50 -o results/SLURM/2026_02_17/lgn-dense-2-medium-50"
  "-m lgn-dense-2-large-50 -o results/SLURM/2026_02_17/lgn-dense-2-large-50"
  "-m lgn-dense-2-huge-50 -o results/SLURM/2026_02_17/lgn-dense-2-huge-50"
)

# Run the command corresponding to this array index
python train.py -c config/10783560/v3.yml --balance-classes --device cuda -bs 1024 -lf cross_entropy -lr 0.01 --compile-model ${ARGS[$SLURM_ARRAY_TASK_ID]}
