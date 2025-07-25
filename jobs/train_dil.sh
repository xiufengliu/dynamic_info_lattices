#!/bin/bash
#BSUB -J dil_train
#BSUB -o logs/dil_train_%J.out
#BSUB -e logs/dil_train_%J.err
#BSUB -W 24:00
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4

# Dynamic Information Lattices Training Job
# This script trains the DIL model on HPC cluster with LSF

echo "Starting Dynamic Information Lattices training job..."
echo "Job ID: $LSB_JOBID"
echo "Host: $LSB_HOSTS"
echo "Start time: $(date)"

# Load required modules
echo "Loading modules..."
module load cuda/12.9.1
module list

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Check environment
echo "Environment check:"
python --version

# Check if PyTorch is available, install if needed
echo "Checking PyTorch installation..."
python -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch not found. Installing...')
    import subprocess
    import sys
    result = subprocess.run(['conda', 'install', '-y', 'pytorch', 'torchvision', 'pytorch-cuda=12.1', '-c', 'pytorch', '-c', 'nvidia'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print('PyTorch installation successful!')
        import torch
        print(f'PyTorch: {torch.__version__}')
        print(f'CUDA available: {torch.cuda.is_available()}')
    else:
        print(f'PyTorch installation failed: {result.stderr}')
        exit(1)
"

# Navigate to project directory
cd $LS_SUBCWD
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Install project dependencies if needed
echo "Installing project dependencies..."
conda install -y pip
pip install -e . || echo "Project installation failed, continuing..."
pip install -r requirements.txt || echo "Requirements installation failed, continuing..."

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0

# Training parameters (modify as needed)
DATASET=${DATASET:-"etth1"}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EPOCHS=${NUM_EPOCHS:-100}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-96}
PREDICTION_LENGTH=${PREDICTION_LENGTH:-24}

echo "Training parameters:"
echo "  Dataset: $DATASET"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Sequence length: $SEQUENCE_LENGTH"
echo "  Prediction length: $PREDICTION_LENGTH"

# Run training
echo "Starting training..."
python examples/train_dil.py \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --sequence_length $SEQUENCE_LENGTH \
    --prediction_length $PREDICTION_LENGTH \
    --device cuda \
    --output_dir ./experiments/cluster_run_$LSB_JOBID \
    --seed 42

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
fi

echo "End time: $(date)"
echo "Job completed."
