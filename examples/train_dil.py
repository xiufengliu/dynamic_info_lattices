#!/usr/bin/env python3
"""
Training script for Dynamic Information Lattices

This script demonstrates how to train the DIL model on time series forecasting tasks.
It reproduces the experimental setup from the paper.
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
    ScoreNetwork, EntropyWeightNetwork,
    get_dataset, DataPreprocessor,
    DILTrainer, TrainingConfig,
    set_seed, get_device, setup_logging, create_experiment_dir
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Dynamic Information Lattices")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="traffic", 
                       choices=["traffic", "solar", "exchange", "weather"],
                       help="Dataset to use for training")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing datasets")
    parser.add_argument("--sequence_length", type=int, default=96,
                       help="Input sequence length")
    parser.add_argument("--prediction_length", type=int, default=24,
                       help="Prediction sequence length")
    
    # Model arguments
    parser.add_argument("--model_channels", type=int, default=64,
                       help="Base number of model channels")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000,
                       help="Number of diffusion steps during training")
    parser.add_argument("--inference_steps", type=int, default=20,
                       help="Number of inference steps")
    parser.add_argument("--max_scales", type=int, default=4,
                       help="Maximum number of lattice scales")
    parser.add_argument("--entropy_budget", type=float, default=0.2,
                       help="Entropy budget (fraction of lattice nodes)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="dil_experiment",
                       help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="Output directory for experiments")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="dynamic-info-lattices",
                       help="Weights & Biases project name")
    
    return parser.parse_args()


def create_model(args, data_shape):
    """Create DIL model"""
    # Model configuration
    model_config = DILConfig(
        num_diffusion_steps=args.num_diffusion_steps,
        inference_steps=args.inference_steps,
        max_scales=args.max_scales,
        entropy_budget=args.entropy_budget,
        device=args.device
    )
    
    # Score network
    score_network = ScoreNetwork(
        in_channels=data_shape[-1],
        out_channels=data_shape[-1],
        model_channels=args.model_channels
    )
    
    # Create DIL model
    model = DynamicInfoLattices(
        config=model_config,
        score_network=score_network,
        data_shape=data_shape
    )
    
    return model


def create_data_loaders(args):
    """Create data loaders"""
    # Data preprocessing
    preprocessor = DataPreprocessor(
        scaler_type="standard",
        handle_missing="interpolate"
    )
    
    # Load datasets
    train_dataset = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split="train",
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length
    )
    
    val_dataset = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split="val",
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, preprocessor


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device() if args.device == "auto" else args.device
    args.device = device
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file=str(exp_dir / "logs" / "training.log")
    )
    
    print("=== Dynamic Information Lattices Training ===")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Experiment directory: {exp_dir}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, preprocessor = create_data_loaders(args)
    
    # Get data shape from first batch
    sample_batch = next(iter(train_loader))
    data_shape = sample_batch[0].shape[1:]  # (seq_len, features)
    print(f"Data shape: {data_shape}")
    
    # Create model
    print("Creating model...")
    model = create_model(args, data_shape)
    model = model.to(device)
    
    # Print model info
    from dynamic_info_lattices.utils import count_parameters, get_model_size
    param_info = count_parameters(model)
    size_info = get_model_size(model)
    
    print(f"Model parameters: {param_info['trainable_parameters']:,}")
    print(f"Model size: {size_info['total_mb']:.2f} MB")
    
    # Training configuration
    training_config = TrainingConfig(
        model_config=model.config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        checkpoint_dir=str(exp_dir / "checkpoints"),
        log_dir=str(exp_dir / "logs"),
        seed=args.seed
    )
    
    # Create trainer
    trainer = DILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    # Save configuration
    from dynamic_info_lattices.utils import save_config
    save_config(args, exp_dir / "config.json")
    save_config(training_config, exp_dir / "training_config.json")
    
    # Start training
    print("Starting training...")
    results = trainer.train()
    
    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Total training time: {results['total_time']:.2f} seconds")
    
    # Save final results
    import json
    with open(exp_dir / "results" / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
