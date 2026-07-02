#!/bin/bash
#SBATCH --job-name=combineBitFlipPlots
#SBATCH --output=logs/combineBitFlipPlots_%j.out
#SBATCH --error=logs/combineBitFlipPlots_%j.err
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

python lgnStudies/bitFlipStudy/combine_results.py \
  --root results/SLURM/2026_07_02_model2_bit_flip_robustness \
  --train-flip-probability 0.001
