#!/bin/bash
#BSUB -J cuda_debug
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 00:30
#BSUB -o logs/cuda_debug_%J.out
#BSUB -e logs/cuda_debug_%J.err
#BSUB -N
#BSUB -u xiuli@dtu.dk

# Job information
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Host: $(hostname)"
echo "Date: $(date)"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/zhome/bb/9/101964/xiuli/dynamic_info_lattices:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

# Change to project directory
cd /zhome/bb/9/101964/xiuli/dynamic_info_lattices

# Run CUDA debug test
echo "Running CUDA indexing debug test..."
python debug_cuda_indexing.py

echo "CUDA debug test completed."
