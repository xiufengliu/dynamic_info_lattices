#!/usr/bin/env python3
"""
KDD-Quality Experimental Framework for Dynamic Information Lattices
Comprehensive evaluation with statistical significance testing and multiple baselines
"""

import argparse
import os
import json
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import itertools

# Import our modules
from dynamic_info_lattices.data.processed_datasets import get_available_datasets, get_dataset_info


class KDDExperimentalFramework:
    """KDD-quality experimental framework with comprehensive evaluation"""
    
    def __init__(self):
        self.datasets = get_available_datasets()
        self.baseline_methods = self._define_baseline_methods()
        self.evaluation_metrics = self._define_evaluation_metrics()
        self.experimental_configs = self._define_experimental_configs()
        
    def _define_baseline_methods(self) -> Dict[str, Dict]:
        """Define baseline methods for comparison"""
        return {
            # Simple baselines
            'naive_seasonal': {
                'description': 'Naive seasonal baseline (repeat last season)',
                'implementation': 'baseline_naive_seasonal.py'
            },
            'linear_trend': {
                'description': 'Linear trend extrapolation',
                'implementation': 'baseline_linear_trend.py'
            },
            'arima': {
                'description': 'Auto-ARIMA with seasonal decomposition',
                'implementation': 'baseline_arima.py'
            },
            
            # Deep learning baselines
            'lstm': {
                'description': 'LSTM with attention mechanism',
                'implementation': 'baseline_lstm.py'
            },
            'transformer': {
                'description': 'Vanilla Transformer for time series',
                'implementation': 'baseline_transformer.py'
            },
            'informer': {
                'description': 'Informer (AAAI 2021)',
                'implementation': 'baseline_informer.py'
            },
            'autoformer': {
                'description': 'Autoformer (NeurIPS 2021)',
                'implementation': 'baseline_autoformer.py'
            },
            'fedformer': {
                'description': 'FEDformer (ICML 2022)',
                'implementation': 'baseline_fedformer.py'
            },
            'patchtst': {
                'description': 'PatchTST (ICLR 2023)',
                'implementation': 'baseline_patchtst.py'
            },
            
            # Recent SOTA methods
            'timesnet': {
                'description': 'TimesNet (ICLR 2023)',
                'implementation': 'baseline_timesnet.py'
            },
            'dlinear': {
                'description': 'DLinear (AAAI 2023)',
                'implementation': 'baseline_dlinear.py'
            },
            'nbeats': {
                'description': 'N-BEATS (ICLR 2020)',
                'implementation': 'baseline_nbeats.py'
            },
            
            # Diffusion baselines
            'tsdiff': {
                'description': 'TSDiff (ICLR 2024)',
                'implementation': 'baseline_tsdiff.py'
            },
            'csdi': {
                'description': 'CSDI (NeurIPS 2021)',
                'implementation': 'baseline_csdi.py'
            },
            'timegrad': {
                'description': 'TimeGrad (ICML 2021)',
                'implementation': 'baseline_timegrad.py'
            }
        }
    
    def _define_evaluation_metrics(self) -> List[str]:
        """Define comprehensive evaluation metrics for KDD"""
        return [
            # Point forecast accuracy
            'mae', 'rmse', 'mape', 'smape', 'mase',
            
            # Probabilistic forecast quality
            'crps', 'quantile_loss', 'energy_score', 'coverage_probability',
            
            # Distribution-based metrics
            'wasserstein_distance', 'kl_divergence',
            
            # Efficiency metrics
            'training_time', 'inference_time', 'memory_usage', 'energy_consumption',
            
            # Robustness metrics
            'noise_robustness', 'missing_data_robustness'
        ]
    
    def _define_experimental_configs(self) -> Dict[str, Dict]:
        """Define experimental configurations for different dataset categories"""
        return {
            # Small datasets (< 1000 samples)
            'small_scale': {
                'datasets': ['illness', 'gefcom2014', 'southern_china'],
                'sequence_lengths': [24, 48, 96],
                'prediction_lengths': [6, 12, 24],
                'cross_validation_folds': 3,
                'random_seeds': [42, 123, 456],
                'epochs': 100,
                'early_stopping_patience': 15
            },
            
            # Medium datasets (1K-10K samples)
            'medium_scale': {
                'datasets': ['exchange_rate', 'etth1', 'etth2'],
                'sequence_lengths': [24, 48, 96, 168],
                'prediction_lengths': [6, 12, 24, 48, 96],
                'cross_validation_folds': 5,
                'random_seeds': [42, 123, 456, 789, 999],
                'epochs': 150,
                'early_stopping_patience': 15
            },
            
            # Large datasets (10K-100K samples)
            'large_scale': {
                'datasets': ['ettm1', 'ettm2', 'weather', 'solar'],
                'sequence_lengths': [96, 192, 336, 720],
                'prediction_lengths': [24, 48, 96, 192, 336],
                'cross_validation_folds': 3,
                'random_seeds': [42, 123, 456],
                'epochs': 100,
                'early_stopping_patience': 10
            },
            
            # Very large datasets (>100K samples)
            'very_large_scale': {
                'datasets': ['ecl', 'electricity', 'traffic'],
                'sequence_lengths': [96, 192, 336],
                'prediction_lengths': [24, 48, 96, 192],
                'cross_validation_folds': 3,
                'random_seeds': [42, 123, 456],
                'epochs': 80,
                'early_stopping_patience': 8
            }
        }
    
    def generate_comprehensive_experiments(self) -> List[Dict]:
        """Generate comprehensive experimental plan"""
        experiments = []
        
        for scale, config in self.experimental_configs.items():
            for dataset in config['datasets']:
                dataset_info = get_dataset_info(dataset)
                
                for seq_len in config['sequence_lengths']:
                    for pred_len in config['prediction_lengths']:
                        # Skip invalid combinations
                        if pred_len >= seq_len:
                            continue
                        
                        for seed in config['random_seeds']:
                            for fold in range(config['cross_validation_folds']):
                                # Our method
                                experiments.append({
                                    'method': 'dil',
                                    'dataset': dataset,
                                    'scale': scale,
                                    'sequence_length': seq_len,
                                    'prediction_length': pred_len,
                                    'seed': seed,
                                    'fold': fold,
                                    'epochs': config['epochs'],
                                    'early_stopping_patience': config['early_stopping_patience'],
                                    'dataset_info': dataset_info
                                })
                                
                                # Baseline methods (subset for efficiency)
                                key_baselines = ['dlinear', 'patchtst', 'timesnet', 'tsdiff']
                                for baseline in key_baselines:
                                    experiments.append({
                                        'method': baseline,
                                        'dataset': dataset,
                                        'scale': scale,
                                        'sequence_length': seq_len,
                                        'prediction_length': pred_len,
                                        'seed': seed,
                                        'fold': fold,
                                        'epochs': config['epochs'],
                                        'early_stopping_patience': config['early_stopping_patience'],
                                        'dataset_info': dataset_info
                                    })
        
        return experiments
    
    def create_job_script(self, experiment: Dict, job_id: str) -> str:
        """Create LSF job script for a single experiment"""
        
        method = experiment['method']
        dataset = experiment['dataset']
        seq_len = experiment['sequence_length']
        pred_len = experiment['prediction_length']
        seed = experiment['seed']
        fold = experiment['fold']
        
        # Determine resource requirements based on dataset size
        if experiment['scale'] == 'very_large_scale':
            queue = 'gpua100'
            mem = '32GB'
            time_limit = '24:00'
            gpu_mem = 'gpu40gb'
        elif experiment['scale'] == 'large_scale':
            queue = 'gpuv100'
            mem = '24GB'
            time_limit = '12:00'
            gpu_mem = 'gpu32gb'
        else:
            queue = 'gpuv100'
            mem = '16GB'
            time_limit = '8:00'
            gpu_mem = 'gpu16gb'
        
        script_content = f"""#!/bin/bash
#BSUB -J kdd_{method}_{dataset}_{seq_len}_{pred_len}_s{seed}_f{fold}
#BSUB -q {queue}
#BSUB -n 4
#BSUB -R "rusage[mem={mem}]"
#BSUB -R "select[{gpu_mem}]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W {time_limit}
#BSUB -o logs/kdd_{method}_{dataset}_{seq_len}_{pred_len}_s{seed}_f{fold}.out
#BSUB -e logs/kdd_{method}_{dataset}_{seq_len}_{pred_len}_s{seed}_f{fold}.err

# Load modules
module load python3/3.11.3
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/zhome/bb/9/101964/xiuli/dynamic_info_lattices:$PYTHONPATH
export PYTHONHASHSEED={seed}

# Change to project directory
cd /zhome/bb/9/101964/xiuli/dynamic_info_lattices

# Install package
pip install -e . --quiet

# Set random seeds for reproducibility
export RANDOM_SEED={seed}

echo "=== KDD Experiment ==="
echo "Method: {method}"
echo "Dataset: {dataset}"
echo "Sequence Length: {seq_len}"
echo "Prediction Length: {pred_len}"
echo "Random Seed: {seed}"
echo "CV Fold: {fold}"
echo "Job ID: {job_id}"
echo "Scale: {experiment['scale']}"
echo "Start Time: $(date)"
echo "========================"

# Run experiment based on method
if [ "{method}" = "dil" ]; then
    python train_multi_dataset.py \\
        --dataset {dataset} \\
        --sequence_length {seq_len} \\
        --prediction_length {pred_len} \\
        --epochs {experiment['epochs']} \\
        --device cuda \\
        --output_dir experiments/kdd/{method}/{dataset} \\
        --log_dir logs \\
        --save_every 20 \\
        --eval_every 5
else
    python baselines/run_baseline.py \\
        --method {method} \\
        --dataset {dataset} \\
        --sequence_length {seq_len} \\
        --prediction_length {pred_len} \\
        --epochs {experiment['epochs']} \\
        --seed {seed} \\
        --fold {fold} \\
        --output_dir experiments/kdd/{method}/{dataset}
fi

echo "Experiment completed: $(date)"
"""
        
        return script_content
    
    def submit_kdd_experiments(self, dry_run: bool = False, max_concurrent: int = 20) -> List[Dict]:
        """Submit all KDD experiments"""
        
        # Generate all experiments
        experiments = self.generate_comprehensive_experiments()
        
        print(f"Generated {len(experiments)} total experiments")
        
        # Group by method and scale for reporting
        method_counts = {}
        scale_counts = {}
        
        for exp in experiments:
            method = exp['method']
            scale = exp['scale']
            
            method_counts[method] = method_counts.get(method, 0) + 1
            scale_counts[scale] = scale_counts.get(scale, 0) + 1
        
        print("\nExperiment breakdown:")
        print("By method:")
        for method, count in sorted(method_counts.items()):
            print(f"  {method:15}: {count:4d} experiments")
        
        print("\nBy scale:")
        for scale, count in sorted(scale_counts.items()):
            print(f"  {scale:15}: {count:4d} experiments")
        
        if dry_run:
            print(f"\n[DRY RUN] Would submit {len(experiments)} experiments")
            return experiments
        
        # Create directories
        os.makedirs("jobs/kdd", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("experiments/kdd", exist_ok=True)
        
        # Submit experiments
        submitted_jobs = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, experiment in enumerate(experiments):
            job_id = f"kdd_{timestamp}_{i:04d}"
            
            # Create job script
            script_content = self.create_job_script(experiment, job_id)
            script_path = f"jobs/kdd/{job_id}.sh"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            os.chmod(script_path, 0o755)
            
            # Submit job
            try:
                result = subprocess.run(
                    ["bsub", "<", script_path],
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    job_info = result.stdout.strip()
                    print(f"✅ Submitted {i+1}/{len(experiments)}: {job_info}")
                    
                    experiment['job_id'] = job_id
                    experiment['script_path'] = script_path
                    experiment['submission_output'] = job_info
                    submitted_jobs.append(experiment)
                else:
                    print(f"❌ Failed to submit {script_path}: {result.stderr}")
            
            except Exception as e:
                print(f"❌ Error submitting {script_path}: {e}")
            
            # Rate limiting
            if len(submitted_jobs) % max_concurrent == 0:
                print(f"⏸️  Submitted {len(submitted_jobs)} jobs, waiting 60s...")
                time.sleep(60)
        
        # Save experiment tracking
        tracking_file = f"experiments/kdd_experiment_tracking_{timestamp}.json"
        with open(tracking_file, 'w') as f:
            json.dump(submitted_jobs, f, indent=2, default=str)
        
        print(f"\n🎉 Successfully submitted {len(submitted_jobs)} KDD experiments!")
        print(f"📊 Experiment tracking saved to: {tracking_file}")
        
        return submitted_jobs


def main():
    parser = argparse.ArgumentParser(description="KDD-quality experimental framework")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be submitted without actually submitting")
    parser.add_argument("--max-concurrent", type=int, default=20,
                       help="Maximum number of concurrent jobs")
    parser.add_argument("--scales", nargs="+",
                       choices=["small_scale", "medium_scale", "large_scale", "very_large_scale"],
                       help="Which scales to run (default: all)")
    
    args = parser.parse_args()
    
    print("🏆 KDD-Quality Experimental Framework for Dynamic Information Lattices")
    print("=" * 80)
    
    # Initialize framework
    framework = KDDExperimentalFramework()
    
    # Filter scales if specified
    if args.scales:
        original_configs = framework.experimental_configs.copy()
        framework.experimental_configs = {
            k: v for k, v in original_configs.items() if k in args.scales
        }
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No jobs will be submitted")
    else:
        print(f"📋 Will submit experiments with max {args.max_concurrent} concurrent jobs")
        print("\n⚠️  This will submit a large number of jobs for comprehensive evaluation!")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Submit experiments
    submitted_jobs = framework.submit_kdd_experiments(
        dry_run=args.dry_run,
        max_concurrent=args.max_concurrent
    )
    
    if not args.dry_run and submitted_jobs:
        print("\n📋 Next steps for KDD submission:")
        print("1. Monitor progress: bjobs | grep kdd_")
        print("2. Analyze results: python analyze_experimental_results.py --stats --plots --latex")
        print("3. Generate paper tables: python generate_kdd_tables.py")
        print("4. Statistical significance: python statistical_analysis.py")


if __name__ == "__main__":
    main()
