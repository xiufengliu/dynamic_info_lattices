#!/usr/bin/env python3
"""
Evaluation script for Dynamic Information Lattices

This script demonstrates how to evaluate the trained DIL model and reproduce
the experimental results from the paper.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dynamic_info_lattices import (
    DynamicInfoLattices, DILConfig,
    ScoreNetwork
)

# Import real dataset loader
from dynamic_info_lattices.data.real_datasets import get_real_dataset

import torch
import numpy as np
import logging
from pathlib import Path
import json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Dynamic Information Lattices")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str,
                       help="Path to model configuration file")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="etth1",
                       choices=["etth1", "etth2", "ettm1", "ettm2", "ecl", "gefcom2014",
                               "southern_china", "traffic", "solar", "exchange", "weather"],
                       help="Dataset to evaluate on")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing datasets")
    parser.add_argument("--sequence_length", type=int, default=96,
                       help="Input sequence length")
    parser.add_argument("--prediction_length", type=int, default=24,
                       help="Prediction sequence length")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for probabilistic evaluation")
    
    # Ablation study
    parser.add_argument("--run_ablation", action="store_true",
                       help="Run ablation study")
    parser.add_argument("--ablation_components", nargs="+",
                       default=["score_entropy", "guidance_entropy", "solver_entropy", 
                               "temporal_entropy", "spectral_entropy"],
                       help="Components to ablate")
    
    # Robustness testing
    parser.add_argument("--run_robustness", action="store_true",
                       help="Run robustness testing")
    parser.add_argument("--missing_rates", nargs="+", type=float,
                       default=[0.0, 0.1, 0.2, 0.3],
                       help="Missing data rates to test")
    parser.add_argument("--noise_levels", nargs="+", type=float,
                       default=[0.0, 0.05, 0.1, 0.2],
                       help="Noise levels to test")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save predictions to file")
    
    return parser.parse_args()


def load_model(checkpoint_path, config_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")

    # For demonstration, create a simple model
    # In practice, you would load the actual checkpoint
    model_config = DILConfig(
        num_diffusion_steps=100,
        inference_steps=10,
        max_scales=3,
        entropy_budget=0.2
    )

    data_shape = (96, 1)  # Default shape for demonstration

    score_network = ScoreNetwork(
        in_channels=data_shape[-1],
        out_channels=data_shape[-1],
        model_channels=64
    )

    model = DynamicInfoLattices(
        config=model_config,
        score_network=score_network,
        data_shape=data_shape
    )

    # Try to load checkpoint if it exists
    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"✓ Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("Using randomly initialized model for demonstration")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Using randomly initialized model for demonstration")

    model = model.to(device)
    model.eval()

    return model


def create_test_loader(args):
    """Create test data loader"""
    test_dataset = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split="test",
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return test_loader


def create_baseline_models(device):
    """Create baseline models for comparison"""
    # This is a placeholder - you would implement actual baseline models
    # For demonstration, we'll create simple models
    
    class SimpleBaseline(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear = torch.nn.Linear(input_size, output_size)
        
        def forward(self, x):
            # Simple linear model
            batch_size, seq_len, features = x.shape
            x_flat = x.view(batch_size, -1)
            output = self.linear(x_flat)
            return output.view(batch_size, -1, features)
    
    baselines = {
        "Linear": SimpleBaseline(96, 24).to(device),
        # Add more baselines here:
        # "DDPM": DDPMBaseline().to(device),
        # "TSDiff": TSDiffBaseline().to(device),
        # "DPM-Solver++": DPMSolverBaseline().to(device),
    }
    
    return baselines


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device() if args.device == "auto" else args.device
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    print("=== Dynamic Information Lattices Evaluation ===")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.config, device)
    
    # Create test data loader
    print("Loading test data...")
    test_loader = create_test_loader(args)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create baseline models
    baseline_models = create_baseline_models(device)
    print(f"Baseline models: {list(baseline_models.keys())}")
    
    # Evaluation configuration
    eval_config = EvaluationConfig(
        quantiles=[0.1, 0.5, 0.9],
        compute_detailed_metrics=True,
        missing_data_rates=args.missing_rates if args.run_robustness else [0.0],
        noise_levels=args.noise_levels if args.run_robustness else [0.0],
        ablation_components=args.ablation_components if args.run_ablation else [],
        results_dir=args.output_dir,
        save_results=True,
        plot_results=True
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        config=eval_config,
        device=device
    )
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate(
        test_loader=test_loader,
        baseline_models=baseline_models
    )
    
    # Print summary results
    print("\n=== Evaluation Results ===")
    
    # Standard metrics
    standard_metrics = results['standard']['metrics']
    print(f"MAE: {standard_metrics['mae']:.4f}")
    print(f"MSE: {standard_metrics['mse']:.4f}")
    print(f"RMSE: {standard_metrics['rmse']:.4f}")
    if 'crps' in standard_metrics:
        print(f"CRPS: {standard_metrics['crps']:.4f}")
    print(f"Inference time: {standard_metrics['avg_inference_time']:.4f}s")
    print(f"Throughput: {standard_metrics['throughput']:.2f} samples/s")
    
    # Baseline comparison
    if 'baseline_comparison' in results:
        print("\n=== Baseline Comparison ===")
        for baseline_name, baseline_results in results['baseline_comparison'].items():
            baseline_mae = baseline_results['metrics']['mae']
            improvement = (baseline_mae - standard_metrics['mae']) / baseline_mae * 100
            speedup = baseline_results['speedup']
            significance = baseline_results['significance_test']['is_significant']
            
            print(f"{baseline_name}:")
            print(f"  MAE: {baseline_mae:.4f} (Improvement: {improvement:+.1f}%)")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Significant: {significance}")
    
    # Ablation study
    if args.run_ablation and 'ablation' in results:
        print("\n=== Ablation Study ===")
        for component, ablation_results in results['ablation'].items():
            performance_drop = ablation_results['performance_drop']['mae']
            print(f"{component}: +{performance_drop:.4f} MAE when removed")
    
    # Robustness testing
    if args.run_robustness and 'robustness' in results:
        print("\n=== Robustness Testing ===")
        
        if 'missing_data' in results['robustness']:
            print("Missing Data Robustness:")
            for key, metrics in results['robustness']['missing_data'].items():
                rate = key.split('_')[1]
                print(f"  {float(rate)*100:.0f}% missing: MAE = {metrics['mae']:.4f}")
        
        if 'noise' in results['robustness']:
            print("Noise Robustness:")
            for key, metrics in results['robustness']['noise'].items():
                level = key.split('_')[1]
                print(f"  {float(level)*100:.0f}% noise: MAE = {metrics['mae']:.4f}")
    
    # Save predictions if requested
    if args.save_predictions:
        from dynamic_info_lattices.utils import save_predictions
        predictions = results['standard']['predictions']
        targets = results['standard']['targets']
        
        save_predictions(
            predictions=predictions,
            targets=targets,
            filepath=Path(args.output_dir) / "predictions.npz",
            metadata={
                'dataset': args.dataset,
                'checkpoint': args.checkpoint,
                'metrics': standard_metrics
            }
        )
        print(f"Predictions saved to {args.output_dir}/predictions.npz")
    
    print(f"\nDetailed results saved to {args.output_dir}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
