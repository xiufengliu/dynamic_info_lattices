#!/bin/bash
#SBATCH --job-name=dil_experiments
#SBATCH --output=logs/dil_experiments_%j.out
#SBATCH --error=logs/dil_experiments_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Dynamic Information Lattices Full Experiments
# This script runs experiments on multiple datasets to reproduce paper results

echo "Starting Dynamic Information Lattices full experiments..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Create logs and experiments directories
mkdir -p logs
mkdir -p experiments

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Define datasets and their configurations
declare -A DATASETS
DATASETS[etth1]="96,24"
DATASETS[etth2]="96,24"
DATASETS[ettm1]="96,24"
DATASETS[ettm2]="96,24"
DATASETS[ecl]="96,24"

# Training parameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=100

echo "Running experiments on multiple datasets..."

# Loop through datasets
for dataset in "${!DATASETS[@]}"; do
    echo "=" * 60
    echo "Training on dataset: $dataset"
    echo "=" * 60
    
    # Parse sequence and prediction lengths
    IFS=',' read -r seq_len pred_len <<< "${DATASETS[$dataset]}"
    
    # Create experiment directory
    exp_dir="./experiments/${dataset}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $exp_dir
    
    echo "Experiment directory: $exp_dir"
    echo "Sequence length: $seq_len"
    echo "Prediction length: $pred_len"
    
    # Run training
    echo "Starting training for $dataset..."
    python examples/train_dil.py \
        --dataset $dataset \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --sequence_length $seq_len \
        --prediction_length $pred_len \
        --device cuda \
        --output_dir $exp_dir \
        --seed 42
    
    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for $dataset!"
        
        # Run evaluation
        echo "Starting evaluation for $dataset..."
        python examples/evaluate_dil.py \
            --dataset $dataset \
            --checkpoint $exp_dir/training_results.json \
            --batch_size $BATCH_SIZE \
            --device cuda \
            --output_dir $exp_dir
        
        if [ $? -eq 0 ]; then
            echo "Evaluation completed successfully for $dataset!"
        else
            echo "Evaluation failed for $dataset"
        fi
    else
        echo "Training failed for $dataset"
    fi
    
    echo "Completed experiment for $dataset"
    echo ""
done

echo "All experiments completed!"
echo "End time: $(date)"

# Generate summary report
echo "Generating summary report..."
python -c "
import os
import json
from pathlib import Path

print('\\n' + '='*60)
print('EXPERIMENT SUMMARY')
print('='*60)

exp_dirs = [d for d in Path('./experiments').iterdir() if d.is_dir() and 'etth' in d.name or 'ecl' in d.name]

for exp_dir in sorted(exp_dirs):
    results_file = exp_dir / 'training_results.json'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            dataset = exp_dir.name.split('_')[0]
            best_loss = results.get('best_val_loss', 'N/A')
            total_time = results.get('total_time', 'N/A')
            
            print(f'{dataset:10} | Loss: {best_loss:8.6f} | Time: {total_time:8.2f}s')
        except:
            print(f'{exp_dir.name:10} | Error reading results')
    else:
        print(f'{exp_dir.name:10} | No results file')

print('='*60)
"

echo "Job completed."
