#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-5   # number of runs - 1

# Define argument sets
ARGS=(
  "-m lgn-conv-6-100-orpool_tau10 -o results/SLURM/2026_03_15/lgn-conv-6-100-orpool_tau10"
  "-m lgn-conv-6-100-orpool_tau20 -o results/SLURM/2026_03_15/lgn-conv-6-100-orpool_tau20"
  "-m lgn-conv-6-100-orpool_tau40 -o results/SLURM/2026_03_15/lgn-conv-6-100-orpool_tau40"
  "-m lgn-conv-6-100-orpool_raw_tau10 -o results/SLURM/2026_03_15/lgn-conv-6-100-orpool_raw_tau10"
  "-m lgn-conv-6-100-orpool_raw_tau20 -o results/SLURM/2026_03_15/lgn-conv-6-100-orpool_raw_tau20"
  "-m lgn-conv-6-100-orpool_raw_tau40 -o results/SLURM/2026_03_15/lgn-conv-6-100-orpool_raw_tau40"
)

# Run the command corresponding to this array index
python train.py -c config/10783560/v3.yml --balance-classes --device cuda -bs 128 -lf cross_entropy -lr 0.01 --compile-model ${ARGS[$SLURM_ARRAY_TASK_ID]}
