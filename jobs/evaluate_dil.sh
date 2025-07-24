#!/bin/bash
#BSUB -J dil_eval
#BSUB -o logs/dil_eval_%J.out
#BSUB -e logs/dil_eval_%J.err
#BSUB -W 4:00
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 4

# Dynamic Information Lattices Evaluation Job
# This script evaluates the DIL model on HPC cluster with LSF

echo "Starting Dynamic Information Lattices evaluation job..."
echo "Job ID: $LSB_JOBID"
echo "Host: $LSB_HOSTS"
echo "Start time: $(date)"

# Load required modules
echo "Loading modules..."
module load cuda/12.9.1
module list

# Activate conda environment (modify as needed)
echo "Activating conda environment..."
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Check environment
echo "Environment check:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Navigate to project directory
cd $LS_SUBCWD
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0

# Evaluation parameters (modify as needed)
DATASET=${DATASET:-"etth1"}
CHECKPOINT=${CHECKPOINT:-"./experiments/training_results.json"}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "Evaluation parameters:"
echo "  Dataset: $DATASET"
echo "  Checkpoint: $CHECKPOINT"
echo "  Batch size: $BATCH_SIZE"

# Run evaluation
echo "Starting evaluation..."
python examples/evaluate_dil.py \
    --dataset $DATASET \
    --checkpoint $CHECKPOINT \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --output_dir ./experiments/eval_$LSB_JOBID

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed with exit code $?"
fi

echo "End time: $(date)"
echo "Job completed."
