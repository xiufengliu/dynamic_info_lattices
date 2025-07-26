#!/bin/bash
#BSUB -J kdd_dil[1-10]
#BSUB -o logs/kdd_dil_%I_%J.out
#BSUB -e logs/kdd_dil_%I_%J.err
#BSUB -W 12:00
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1

# KDD Comprehensive Experimental Suite for Dynamic Information Lattices
# This script runs comprehensive experiments across all 13 datasets with multiple baselines
#
# Submitter: Xiufeng Liu (xiuli@dtu.dk) - Senior Researcher
# Location: Birkerod (outside DTU campus)
# Fixed memory allocation per DTU HPC support requirements

echo "Starting KDD Comprehensive Experimental Suite..."
echo "Job ID: $LSB_JOBID"
echo "Job Index: $LSB_JOBINDEX"
echo "Host: $LSB_HOSTS"
echo "Start time: $(date)"

# Define reasonable experimental configurations for initial run
# Start with smaller subset to avoid overwhelming the cluster
DATASETS=("illness" "exchange_rate" "etth1" "weather" "electricity")
METHODS=("dil" "dlinear")
SEQUENCE_LENGTHS=(96)
PREDICTION_LENGTHS=(24)
SEEDS=(42)

# Calculate experiment configuration based on job index
TOTAL_CONFIGS=0
for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
            for pred_len in "${PREDICTION_LENGTHS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    if [ $pred_len -lt $seq_len ]; then
                        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
                    fi
                done
            done
        done
    done
done

echo "Total experimental configurations: $TOTAL_CONFIGS"

# Map job index to specific configuration
CONFIG_INDEX=$((LSB_JOBINDEX - 1))
if [ $CONFIG_INDEX -ge $TOTAL_CONFIGS ]; then
    echo "Job index $LSB_JOBINDEX exceeds total configurations $TOTAL_CONFIGS. Exiting."
    exit 0
fi

# Find the specific configuration for this job index
current_index=0
found=false

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
            for pred_len in "${PREDICTION_LENGTHS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    if [ $pred_len -lt $seq_len ]; then
                        if [ $current_index -eq $CONFIG_INDEX ]; then
                            CURRENT_DATASET=$dataset
                            CURRENT_METHOD=$method
                            CURRENT_SEQ_LEN=$seq_len
                            CURRENT_PRED_LEN=$pred_len
                            CURRENT_SEED=$seed
                            found=true
                            break 5
                        fi
                        current_index=$((current_index + 1))
                    fi
                done
            done
        done
    done
done

if [ "$found" = false ]; then
    echo "Could not find configuration for index $CONFIG_INDEX"
    exit 1
fi

echo "Experiment Configuration:"
echo "  Dataset: $CURRENT_DATASET"
echo "  Method: $CURRENT_METHOD"
echo "  Sequence Length: $CURRENT_SEQ_LEN"
echo "  Prediction Length: $CURRENT_PRED_LEN"
echo "  Random Seed: $CURRENT_SEED"

# Load required modules
echo "Loading modules..."
module load cuda/12.9.1

# Activate conda environment
echo "Activating conda environment..."
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base
else
    echo "Conda not found, using system python"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/zhome/bb/9/101964/xiuli/dynamic_info_lattices:$PYTHONPATH
export PYTHONHASHSEED=$CURRENT_SEED

# Check environment
echo "Environment check:"
python --version
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTHONHASHSEED: $PYTHONHASHSEED"

# Quick PyTorch check
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
"

# Navigate to project directory
cd /zhome/bb/9/101964/xiuli/dynamic_info_lattices
echo "Working directory: $(pwd)"

# Create output directories
mkdir -p logs
mkdir -p experiments/kdd/$CURRENT_METHOD/$CURRENT_DATASET

# Install project dependencies
echo "Installing project dependencies..."
pip install -e . --quiet

# Determine epochs based on dataset size
case $CURRENT_DATASET in
    "illness"|"gefcom2014"|"southern_china")
        NUM_EPOCHS=100
        ;;
    "exchange_rate"|"etth1"|"etth2")
        NUM_EPOCHS=80
        ;;
    "ettm1"|"ettm2"|"weather"|"solar")
        NUM_EPOCHS=60
        ;;
    "ecl"|"electricity"|"traffic")
        NUM_EPOCHS=50
        ;;
    *)
        NUM_EPOCHS=60
        ;;
esac

echo "Final training parameters:"
echo "  Dataset: $CURRENT_DATASET"
echo "  Method: $CURRENT_METHOD"
echo "  Sequence length: $CURRENT_SEQ_LEN"
echo "  Prediction length: $CURRENT_PRED_LEN"
echo "  Epochs: $NUM_EPOCHS"
echo "  Random seed: $CURRENT_SEED"

# Run experiment based on method
echo "Starting experiment..."
if [ "$CURRENT_METHOD" = "dil" ]; then
    # Run our DIL method
    python train_multi_dataset.py \
        --dataset $CURRENT_DATASET \
        --sequence_length $CURRENT_SEQ_LEN \
        --prediction_length $CURRENT_PRED_LEN \
        --epochs $NUM_EPOCHS \
        --device cuda \
        --output_dir experiments/kdd/$CURRENT_METHOD/$CURRENT_DATASET \
        --log_dir logs \
        --save_every 20 \
        --eval_every 10
else
    # Run baseline method
    python baselines/run_baseline.py \
        --method $CURRENT_METHOD \
        --dataset $CURRENT_DATASET \
        --sequence_length $CURRENT_SEQ_LEN \
        --prediction_length $CURRENT_PRED_LEN \
        --epochs $NUM_EPOCHS \
        --seed $CURRENT_SEED \
        --fold 0 \
        --output_dir experiments/kdd/$CURRENT_METHOD/$CURRENT_DATASET \
        --device cuda
fi

# Check if experiment completed successfully
EXPERIMENT_EXIT_CODE=$?
if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results saved to: experiments/kdd/$CURRENT_METHOD/$CURRENT_DATASET"
else
    echo "Experiment failed with exit code $EXPERIMENT_EXIT_CODE"
    echo "Check logs for details: logs/kdd_dil_${LSB_JOBINDEX}_${LSB_JOBID}.err"
fi

echo "End time: $(date)"
echo "KDD Experiment completed."
echo "Configuration: $CURRENT_METHOD on $CURRENT_DATASET (seq=$CURRENT_SEQ_LEN, pred=$CURRENT_PRED_LEN, seed=$CURRENT_SEED)"
