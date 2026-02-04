#!/usr/bin/env bash

module load anaconda3/2025.12
module load cudatoolkit/13.0
conda activate smart_pixels

read -p "Request GPU resources? (y/n): " use_gpu

if [[ "$use_gpu" =~ ^[yY]$ ]]; then
  echo "Starting interactive session WITH GPU..."
  srun --gres=gpu:1 -c 8 --mem=32G -t 2:00:00 --pty bash
else
  echo "Continuing on current machine without GPU."
fi
