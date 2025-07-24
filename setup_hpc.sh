#!/bin/bash
# HPC Environment Setup for Dynamic Information Lattices
# This script sets up the proper environment for running DIL on HPC systems
# with conda Python and CUDA modules.

echo "Setting up Dynamic Information Lattices on HPC..."

# Load CUDA module
echo "Loading CUDA module..."
module load cuda/12.9.1

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please ensure conda is installed and in PATH."
    exit 1
fi

# Initialize conda for bash (if needed)
eval "$(conda shell.bash hook)"

# Activate conda environment (modify as needed)
echo "Activating conda environment..."
# Uncomment and modify the line below if you have a specific conda environment
# conda activate your_env_name

# Check Python version
echo "Python version:"
python --version

# Check PyTorch and CUDA availability
echo "Checking PyTorch and CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - will run on CPU')
"

# Install package in development mode
echo "Installing Dynamic Information Lattices..."
pip install -e .

echo "Setup complete!"
echo ""
echo "For interactive use:"
echo "  python examples/train_dil.py --dataset etth1 --device cuda"
echo "  python examples/evaluate_dil.py --dataset etth1 --device cuda"
echo "  python examples/simple_example.py"
echo ""
echo "For cluster submission (LSF):"
echo "  bsub < jobs/train_dil.lsf"
echo "  bsub < jobs/evaluate_dil.lsf"
