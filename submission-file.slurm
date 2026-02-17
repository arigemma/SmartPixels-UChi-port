#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --time=1:59:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

# Environment setup
echo "========== ENVIRONMENT SETUP =========="
echo "Loading required modules..."
module load anaconda3/2024.6
module load cudatoolkit/12.6

echo "Activating conda environment: torchlogix"
conda activate torchlogix

echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "========================================"

echo "========== HOST GPU INFORMATION =========="
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi is available on the host"
    echo "Full nvidia-smi output:"
    nvidia-smi
    echo "NVIDIA driver version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo "CUDA version:"
    nvidia-smi --query-gpu=cuda_version --format=csv,noheader
else
    echo "nvidia-smi command not found on the host"
    echo "Checking for GPU devices:"
    ls -la /dev/nvidia* 2>/dev/null || echo "No NVIDIA device files found"
    echo "Checking CUDA installation:"
    ls -la /usr/local/cuda* 2>/dev/null || echo "No CUDA installation found in /usr/local"
fi
echo "==========================================="

# Create logs directory
mkdir -p slurm_logs

# Parameter arrays (auto-generated from config)
seeds=(0)
sizes=(2 4 8 16 32)
architectures=(conv2d conv3d)
aggregations=(max mean flatten)

# Calculate indices
seed_idx=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))
remainder=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))

size_idx=$((remainder % ${#sizes[@]}))
remainder=$((remainder / ${#sizes[@]}))

arch_idx=$((remainder % ${#architectures[@]}))
remainder=$((remainder / ${#architectures[@]}))

aggregation_idx=$((remainder % ${#aggregations[@]}))
remainder=$((remainder / ${#aggregations[@]}))

# Get parameter values
seed=${seeds[$seed_idx]}
size=${sizes[$size_idx]}
arch=${architectures[$arch_idx]}
aggregation=${aggregations[$aggregation_idx]}

# Output directory
CAMPAIGN_ID="${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="/home/lg0508/Projects/SmartPixels/results/campaign_${CAMPAIGN_ID}/seed_${seed// /_}_size_${size// /_}_arch_${arch// /_}_aggregation_${aggregation// /_}"
mkdir -p "$OUTPUT_DIR"

echo "========== EXPERIMENT PARAMETERS =========="
echo "Running experiment with parameters:"
echo "  seed: '$seed'"
echo "  size: '$size'"
echo "  architecture: '$arch'"
echo "  aggregation: '$aggregation'"
echo "  Output directory: $OUTPUT_DIR"
echo "  Array indices: seed=$seed_idx, size=$size_idx, architecture=$arch_idx, aggregation=$aggregation_idx"
echo "==========================================="

# Run Python script
python train.py \
    --model-size "$size" \
    --architecture $arch \
    --aggregation $aggregation \
    --device cuda \
    --batch-size 64 \
    --seed "$seed" \
    --output "$OUTPUT_DIR"

# Capture exit status and log
EXIT_STATUS=$?

echo "========== EXPERIMENT COMPLETION =========="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "SUCCESS: Experiment completed successfully"
    log_message="SUCCESS - threshold='$threshold', nkernel=$nkernel, stride=$stride, treedepth=$treedepth, Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, Output: $OUTPUT_DIR"
else
    echo "FAILED: Experiment failed with exit status $EXIT_STATUS"
    log_message="FAILED - threshold='$threshold', nkernel=$nkernel, stride=$stride, treedepth=$treedepth, Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, Exit Status: $EXIT_STATUS"
fi

# Thread-safe logging
{
    flock -x 200
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $log_message" >> "/home/lg0508/Projects/SmartPixels/results/experiment_log.txt"
} 200>>"/home/lg0508/Projects/SmartPixels/results/experiment_log.txt.lock"

echo "Experiment completed with status: $EXIT_STATUS"
echo "==========================================="

exit $EXIT_STATUS
