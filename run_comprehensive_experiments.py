#!/usr/bin/env python3
"""
Comprehensive Experimental Evaluation for Dynamic Information Lattices
Runs experiments on all 13 available datasets with multiple configurations
"""

import argparse
import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import our modules
from dynamic_info_lattices.data.processed_datasets import get_available_datasets, get_dataset_info


def create_experiment_config():
    """Create comprehensive experiment configuration"""
    
    # Get all available datasets
    datasets = get_available_datasets()
    
    # Define experiment configurations
    experiments = {
        # Small datasets - shorter sequences, more epochs
        'small_datasets': {
            'datasets': ['illness', 'gefcom2014', 'southern_china'],
            'sequence_lengths': [24, 48],
            'prediction_lengths': [6, 12, 24],
            'epochs': 200,
            'batch_size': 64
        },
        
        # Medium datasets - standard configurations
        'medium_datasets': {
            'datasets': ['exchange_rate', 'etth1', 'etth2', 'weather'],
            'sequence_lengths': [96, 192],
            'prediction_lengths': [24, 48, 96],
            'epochs': 150,
            'batch_size': 32
        },
        
        # Large datasets - longer sequences, fewer epochs
        'large_datasets': {
            'datasets': ['ettm1', 'ettm2', 'ecl', 'solar'],
            'sequence_lengths': [96, 192, 336],
            'prediction_lengths': [24, 48, 96, 192],
            'epochs': 100,
            'batch_size': 16
        },
        
        # Very large datasets - optimized for efficiency
        'very_large_datasets': {
            'datasets': ['electricity', 'traffic'],
            'sequence_lengths': [96, 192],
            'prediction_lengths': [24, 48, 96],
            'epochs': 80,
            'batch_size': 8
        }
    }
    
    return experiments


def generate_job_script(dataset, seq_len, pred_len, epochs, batch_size, exp_name, job_id):
    """Generate LSF job script for a single experiment"""
    
    script_content = f"""#!/bin/bash
#BSUB -J dil_{dataset}_{seq_len}_{pred_len}_{job_id}
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o logs/dil_{dataset}_{seq_len}_{pred_len}_{job_id}.out
#BSUB -e logs/dil_{dataset}_{seq_len}_{pred_len}_{job_id}.err

# Load modules
module load python3/3.11.3
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/zhome/bb/9/101964/xiuli/dynamic_info_lattices:$PYTHONPATH

# Change to project directory
cd /zhome/bb/9/101964/xiuli/dynamic_info_lattices

# Install package in development mode
pip install -e . --quiet

# Run experiment
echo "Starting experiment: {exp_name}"
echo "Dataset: {dataset}"
echo "Sequence Length: {seq_len}"
echo "Prediction Length: {pred_len}"
echo "Epochs: {epochs}"
echo "Batch Size: {batch_size}"
echo "Job ID: {job_id}"
echo "Timestamp: $(date)"

python train_multi_dataset.py \\
    --dataset {dataset} \\
    --sequence_length {seq_len} \\
    --prediction_length {pred_len} \\
    --epochs {epochs} \\
    --device cuda \\
    --output_dir experiments/{exp_name}/{dataset} \\
    --log_dir logs \\
    --save_every 20 \\
    --eval_every 10

echo "Experiment completed: $(date)"
"""
    
    return script_content


def submit_experiment_batch(experiments, dry_run=False, max_concurrent=10):
    """Submit all experiments as batch jobs"""
    
    # Create necessary directories
    os.makedirs("jobs/experiments", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    submitted_jobs = []
    job_counter = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for exp_name, config in experiments.items():
        print(f"\n=== {exp_name.upper()} ===")
        
        for dataset in config['datasets']:
            dataset_info = get_dataset_info(dataset)
            print(f"\nDataset: {dataset} ({dataset_info['shape']})")
            
            for seq_len in config['sequence_lengths']:
                for pred_len in config['prediction_lengths']:
                    # Skip if prediction length is too large relative to sequence length
                    if pred_len >= seq_len:
                        continue
                    
                    job_counter += 1
                    job_id = f"{timestamp}_{job_counter:03d}"
                    
                    # Generate job script
                    script_content = generate_job_script(
                        dataset, seq_len, pred_len, 
                        config['epochs'], config['batch_size'],
                        exp_name, job_id
                    )
                    
                    # Save job script
                    script_path = f"jobs/experiments/dil_{dataset}_{seq_len}_{pred_len}_{job_id}.sh"
                    with open(script_path, 'w') as f:
                        f.write(script_content)
                    
                    # Make executable
                    os.chmod(script_path, 0o755)
                    
                    if dry_run:
                        print(f"  [DRY RUN] Would submit: {script_path}")
                    else:
                        # Submit job
                        try:
                            result = subprocess.run(
                                ["bsub", "<", script_path],
                                shell=True,
                                capture_output=True,
                                text=True,
                                cwd="/zhome/bb/9/101964/xiuli/dynamic_info_lattices"
                            )
                            
                            if result.returncode == 0:
                                job_info = result.stdout.strip()
                                print(f"  ✅ Submitted: {job_info}")
                                submitted_jobs.append({
                                    'dataset': dataset,
                                    'seq_len': seq_len,
                                    'pred_len': pred_len,
                                    'job_id': job_id,
                                    'script_path': script_path,
                                    'submission_output': job_info
                                })
                            else:
                                print(f"  ❌ Failed to submit {script_path}: {result.stderr}")
                        
                        except Exception as e:
                            print(f"  ❌ Error submitting {script_path}: {e}")
                    
                    # Rate limiting
                    if len(submitted_jobs) >= max_concurrent and not dry_run:
                        print(f"\n⏸️  Reached max concurrent jobs ({max_concurrent}), waiting...")
                        time.sleep(60)  # Wait 1 minute before submitting more
    
    # Save job tracking information
    if not dry_run:
        job_tracking_file = f"experiments/job_tracking_{timestamp}.json"
        with open(job_tracking_file, 'w') as f:
            json.dump(submitted_jobs, f, indent=2)
        
        print(f"\n📊 Job tracking saved to: {job_tracking_file}")
    
    return submitted_jobs


def create_monitoring_script(job_tracking_file):
    """Create a script to monitor all submitted jobs"""
    
    monitor_script = f"""#!/bin/bash
# Monitor all experimental jobs

echo "=== Dynamic Information Lattices - Experiment Monitoring ==="
echo "Timestamp: $(date)"
echo

# Show job status
echo "=== Current Job Status ==="
bjobs -w | grep "dil_"

echo
echo "=== Job Summary ==="
bjobs | grep "dil_" | awk '{{print $3}}' | sort | uniq -c

echo
echo "=== Recent Log Outputs ==="
for log in logs/dil_*.out; do
    if [[ -f "$log" && $(find "$log" -mmin -30) ]]; then
        echo "--- $log (last 5 lines) ---"
        tail -5 "$log"
        echo
    fi
done

echo "=== Monitoring completed at $(date) ==="
"""
    
    with open("monitor_experiments.sh", 'w') as f:
        f.write(monitor_script)
    
    os.chmod("monitor_experiments.sh", 0o755)
    print("📊 Created monitoring script: monitor_experiments.sh")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive DIL experiments")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be submitted without actually submitting")
    parser.add_argument("--max-concurrent", type=int, default=10,
                       help="Maximum number of concurrent jobs")
    parser.add_argument("--experiment-groups", nargs="+", 
                       choices=["small_datasets", "medium_datasets", "large_datasets", "very_large_datasets"],
                       help="Which experiment groups to run (default: all)")
    
    args = parser.parse_args()
    
    print("🚀 Dynamic Information Lattices - Comprehensive Experimental Evaluation")
    print("=" * 80)
    
    # Get experiment configuration
    all_experiments = create_experiment_config()
    
    # Filter experiments if specified
    if args.experiment_groups:
        experiments = {k: v for k, v in all_experiments.items() if k in args.experiment_groups}
    else:
        experiments = all_experiments
    
    # Show experiment summary
    total_experiments = 0
    for exp_name, config in experiments.items():
        exp_count = 0
        for dataset in config['datasets']:
            for seq_len in config['sequence_lengths']:
                for pred_len in config['prediction_lengths']:
                    if pred_len < seq_len:
                        exp_count += 1
        total_experiments += exp_count
        print(f"{exp_name:20}: {exp_count:3d} experiments")
    
    print(f"{'TOTAL':20}: {total_experiments:3d} experiments")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No jobs will be submitted")
    else:
        print(f"📋 Will submit up to {args.max_concurrent} concurrent jobs")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Submit experiments
    submitted_jobs = submit_experiment_batch(
        experiments, 
        dry_run=args.dry_run, 
        max_concurrent=args.max_concurrent
    )
    
    if not args.dry_run and submitted_jobs:
        # Create monitoring script
        create_monitoring_script("job_tracking.json")
        
        print(f"\n🎉 Successfully submitted {len(submitted_jobs)} experiments!")
        print("\nNext steps:")
        print("1. Monitor jobs: ./monitor_experiments.sh")
        print("2. Check specific job: bjobs -l <job_id>")
        print("3. View logs: tail -f logs/dil_*.out")
        print("4. Kill all jobs: bjobs | grep dil_ | awk '{print $1}' | xargs bkill")


if __name__ == "__main__":
    main()
