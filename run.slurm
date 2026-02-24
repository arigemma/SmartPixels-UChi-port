#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=2:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-3   # number of runs - 1

# Define argument sets
ARGS=(
  "-m lgn-conv-3-200-deepconv4-deepfc4_tau10_fb -o results/SLURM/2026_02_23/lgn-conv-3-200-deepconv4-deepfc4_tau10_fb"
  "-m lgn-conv-3-50-deepconv4-deepfc6_tau10_fb  -o results/SLURM/2026_02_23/lgn-conv-3-50-deepconv4-deepfc6_tau10_fb"
  "-m lgn-dense-2-100-deepconv2-deepfc5_k512_tau8 -o results/SLURM/2026_02_23/lgn-dense-2-100-deepconv2-deepfc5_k512_tau8_fb"
  "-m lgn-dense-2-y0-100-deepconv2-fuse3_tau10_fb -o results/SLURM/2026_02_23/lgn-dense-2-y0-100-deepconv2-fuse3_tau10_fb"
)

# Run the command corresponding to this array index
python train.py -c config/10783560/v3.yml --balance-classes --device cuda -bs 1024 -lf cross_entropy -lr 0.01 --compile-model ${ARGS[$SLURM_ARRAY_TASK_ID]}
