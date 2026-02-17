#!/bin/bash

# HPO Configuration and Submission Script
# Define ALL parameters here - this is the single source of truth

# =============================================================================
# PARAMETER CONFIGURATION (Edit these as needed)
# =============================================================================
SEEDS=(0)
SIZES=(2 4 8 16 32)
ARCHITECTURES=("conv2d" "conv3d")
# AGGREGATIONS=("max")
AGGREGATIONS=("max" "mean" "flatten")

# SLURM Configuration
JOB_NAME="model_training"
TIME_LIMIT="1:59:59"
CPUS_PER_TASK=4
MEMORY="16G"
GPUS=1

# Paths
BASE_OUTPUT_DIR="$PWD/results"
PYTHON_SCRIPT="train.py"
# =============================================================================

# Calculate array size
TOTAL_COMBINATIONS=$((${#SEEDS[@]} * ${#SIZES[@]} * ${#ARCHITECTURES[@]} * ${#AGGREGATIONS[@]}))
MAX_ARRAY_INDEX=$((TOTAL_COMBINATIONS - 1))

echo "========== HPO CAMPAIGN CONFIGURATION =========="
echo "Parameter space:"
echo "  Seeds: ${#SEEDS[@]} options: (${SEEDS[*]})"
echo "  Sizes: ${#SIZES[@]} options: (${SIZES[*]})"
echo "  Architectures: ${#ARCHITECTURES[@]} options: (${ARCHITECTURES[*]})"
echo "  Aggregations: ${#AGGREGATIONS[@]} options: (${AGGREGATIONS[*]})"
echo "  Total combinations: $TOTAL_COMBINATIONS"
echo "  Array range: 0-$MAX_ARRAY_INDEX"
echo "================================================="

# Function to generate the SLURM script content
generate_slurm_script() {
    cat << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --gres=gpu:$GPUS
#SBATCH --array=0-$MAX_ARRAY_INDEX

# Environment setup
echo "========== ENVIRONMENT SETUP =========="
echo "Loading required modules..."
module load anaconda3/2024.6
module load cudatoolkit/12.6

echo "Activating conda environment: torchlogix"
conda activate torchlogix

echo "Python path: \$(which python)"
echo "Python version: \$(python --version)"
echo "Conda environment: \$CONDA_DEFAULT_ENV"
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
seeds=(${SEEDS[@]})
sizes=(${SIZES[@]})
architectures=(${ARCHITECTURES[@]})
aggregations=(${AGGREGATIONS[@]})

# Calculate indices
seed_idx=\$((SLURM_ARRAY_TASK_ID % \${#seeds[@]}))
remainder=\$((SLURM_ARRAY_TASK_ID / \${#seeds[@]}))

size_idx=\$((remainder % \${#sizes[@]}))
remainder=\$((remainder / \${#sizes[@]}))

arch_idx=\$((remainder % \${#architectures[@]}))
remainder=\$((remainder / \${#architectures[@]}))

aggregation_idx=\$((remainder % \${#aggregations[@]}))
remainder=\$((remainder / \${#aggregations[@]}))

# Get parameter values
seed=\${seeds[\$seed_idx]}
size=\${sizes[\$size_idx]}
arch=\${architectures[\$arch_idx]}
aggregation=\${aggregations[\$aggregation_idx]}

# Output directory
CAMPAIGN_ID="\${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="$BASE_OUTPUT_DIR/campaign_\${CAMPAIGN_ID}/seed_\${seed// /_}_size_\${size// /_}_arch_\${arch// /_}_aggregation_\${aggregation// /_}"
mkdir -p "\$OUTPUT_DIR"

echo "========== EXPERIMENT PARAMETERS =========="
echo "Running experiment with parameters:"
echo "  seed: '\$seed'"
echo "  size: '\$size'"
echo "  architecture: '\$arch'"
echo "  aggregation: '\$aggregation'"
echo "  Output directory: \$OUTPUT_DIR"
echo "  Array indices: seed=\$seed_idx, size=\$size_idx, architecture=\$arch_idx, aggregation=\$aggregation_idx"
echo "==========================================="

# Run Python script
python $PYTHON_SCRIPT \\
    --model-size "\$size" \\
    --architecture \$arch \\
    --aggregation \$aggregation \\
    --device cuda \\
    --batch-size 64 \\
    --seed "\$seed" \\
    --output "\$OUTPUT_DIR"

# Capture exit status and log
EXIT_STATUS=\$?

echo "========== EXPERIMENT COMPLETION =========="
if [ \$EXIT_STATUS -eq 0 ]; then
    echo "SUCCESS: Experiment completed successfully"
    log_message="SUCCESS - threshold='\$threshold', nkernel=\$nkernel, stride=\$stride, treedepth=\$treedepth, Job ID: \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}, Output: \$OUTPUT_DIR"
else
    echo "FAILED: Experiment failed with exit status \$EXIT_STATUS"
    log_message="FAILED - threshold='\$threshold', nkernel=\$nkernel, stride=\$stride, treedepth=\$treedepth, Job ID: \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}, Exit Status: \$EXIT_STATUS"
fi

# Thread-safe logging
{
    flock -x 200
    echo "\$(date '+%Y-%m-%d %H:%M:%S') - \$log_message" >> "$BASE_OUTPUT_DIR/experiment_log.txt"
} 200>>"$BASE_OUTPUT_DIR/experiment_log.txt.lock"

echo "Experiment completed with status: \$EXIT_STATUS"
echo "==========================================="

exit \$EXIT_STATUS
EOF
}

# Ask for confirmation
read -p "Generate and submit $TOTAL_COMBINATIONS HPO jobs? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Generate the SLURM script
    SLURM_FILE="submission-file.slurm"
    generate_slurm_script > "$SLURM_FILE"
    chmod +x "$SLURM_FILE"
    
    echo "Generated SLURM script: $SLURM_FILE"
    echo "Submitting job array..."
    
    sbatch "$SLURM_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Job array submitted successfully!"
        echo "Monitor with: squeue -u \$USER"
        echo "Generated script saved as: $SLURM_FILE"
    else
        echo "❌ Submission failed"
        exit 1
    fi
else
    echo "Submission cancelled"
    echo "You can run this script anytime to generate and submit the HPO campaign"
fi
