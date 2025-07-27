#!/usr/bin/env python3
"""
Multi-dataset training script for Dynamic Information Lattices
Supports all processed datasets with unified interface
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Import our modules
from dynamic_info_lattices.core import DynamicInfoLattices, DILConfig
from dynamic_info_lattices.models import ScoreNetwork
from dynamic_info_lattices.data.processed_datasets import create_dataset, get_available_datasets, get_dataset_info
from dynamic_info_lattices.training.trainer import DILTrainer


def validate_tensor_dimensions(data_loader, dataset_name):
    """Validate tensor dimensions to prevent CUDA indexing issues"""
    try:
        # Get a sample batch to check dimensions
        sample_batch = next(iter(data_loader))
        if isinstance(sample_batch, (list, tuple)):
            x = sample_batch[0]
        else:
            x = sample_batch

        batch_size, seq_len = x.shape[:2]
        channels = x.shape[2] if len(x.shape) > 2 else 1

        print(f"Dataset {dataset_name} dimensions: batch={batch_size}, seq_len={seq_len}, channels={channels}")

        # Check for problematic dimensions that might cause CUDA issues
        if seq_len > 10000:
            print(f"WARNING: Very long sequence length ({seq_len}) may cause CUDA indexing issues")
        if batch_size > 1000:
            print(f"WARNING: Very large batch size ({batch_size}) may cause memory issues")
        if channels > 1000:
            print(f"WARNING: Very high dimensionality ({channels}) may cause issues")

        return True
    except Exception as e:
        print(f"ERROR: Failed to validate tensor dimensions for {dataset_name}: {e}")
        return False

def setup_logging(log_dir: str, dataset_name: str) -> logging.Logger:
    """Setup logging for the training run"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{dataset_name}_{timestamp}.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def get_dataset_config(dataset_name: str, base_config: DILConfig) -> DILConfig:
    """Get dataset-specific configuration adjustments"""
    dataset_info = get_dataset_info(dataset_name)
    
    # Create a copy of the base config with simplified settings for debugging
    config = DILConfig()

    # Update config based on dataset characteristics
    config.input_dim = dataset_info['num_series']
    config.sequence_length = base_config.sequence_length
    config.prediction_length = base_config.prediction_length

    # Disable problematic features for debugging
    config.max_scales = 1  # Reduced to avoid scale synchronization issues
    config.inference_steps = 5  # Reduced for faster debugging
    config.adaptive_guidance = False  # Disabled to avoid guidance computation issues
    config.cross_scale_sync = False  # Disabled to avoid interpolation issues
    
    # Adjust model size based on dataset complexity
    if dataset_info['num_series'] > 500:  # Large datasets like traffic
        config.hidden_dim = 256
        config.num_layers = 6
        config.num_heads = 8
    elif dataset_info['num_series'] > 100:  # Medium datasets like solar
        config.hidden_dim = 128
        config.num_layers = 4
        config.num_heads = 8
    else:  # Small datasets like exchange_rate
        config.hidden_dim = 64
        config.num_layers = 3
        config.num_heads = 4
    
    # Adjust training parameters based on dataset size
    if dataset_info['length'] > 50000:  # Large datasets
        config.batch_size = 32
        config.learning_rate = 1e-4
    elif dataset_info['length'] > 10000:  # Medium datasets
        config.batch_size = 64
        config.learning_rate = 5e-4
    else:  # Small datasets
        config.batch_size = 128
        config.learning_rate = 1e-3
    
    return config


def create_data_loaders(dataset_name: str, config: DILConfig, device: str):
    """Create train and test data loaders"""
    
    # Create datasets
    train_dataset = create_dataset(
        dataset_name=dataset_name,
        split="train",
        sequence_length=config.sequence_length,
        prediction_length=config.prediction_length,
        normalize=True
    )

    test_dataset = create_dataset(
        dataset_name=dataset_name,
        split="test",
        sequence_length=config.sequence_length,
        prediction_length=config.prediction_length,
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid CUDA issues
        pin_memory=False  # Disable pin_memory to avoid issues
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid CUDA issues
        pin_memory=False  # Disable pin_memory to avoid issues
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Dynamic Information Lattice on multiple datasets")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset name (use 'list' to see available datasets)")
    parser.add_argument("--sequence_length", type=int, default=96,
                       help="Input sequence length")
    parser.add_argument("--prediction_length", type=int, default=24,
                       help="Prediction sequence length")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--output_dir", type=str, default="experiments",
                       help="Output directory for results")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for log files")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save model every N epochs")
    parser.add_argument("--eval_every", type=int, default=5,
                       help="Evaluate model every N epochs")
    
    args = parser.parse_args()
    
    # Handle device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # List available datasets if requested
    if args.dataset == "list":
        print("\nAvailable datasets:")
        datasets = get_available_datasets()
        for dataset in datasets:
            info = get_dataset_info(dataset)
            print(f"  {dataset:15}: {info['shape']} - {info['description']}")
        return
    
    # Validate dataset
    available_datasets = get_available_datasets()
    if args.dataset not in available_datasets:
        print(f"Error: Dataset '{args.dataset}' not found.")
        print(f"Available datasets: {available_datasets}")
        return
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.dataset)
    logger.info(f"Starting training on dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    logger.info(f"Dataset info: {dataset_info}")
    
    # Create base config
    base_config = DILConfig()
    base_config.sequence_length = args.sequence_length
    base_config.prediction_length = args.prediction_length
    base_config.num_epochs = args.epochs
    
    # Get dataset-specific config
    config = get_dataset_config(args.dataset, base_config)
    logger.info(f"Model config: input_dim={config.input_dim}, hidden_dim={config.hidden_dim}, "
               f"num_layers={config.num_layers}, batch_size={config.batch_size}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
        args.dataset, config, device
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")

    # Create simplified score network with adaptive kernel size
    class SimpleScoreNetwork(nn.Module):
        def __init__(self, in_channels, out_channels, hidden_dim=64):
            super().__init__()
            self.time_embed = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Use kernel size 1 to avoid size issues with small inputs
            self.conv1 = nn.Conv1d(in_channels, hidden_dim, 1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.conv3 = nn.Conv1d(hidden_dim, out_channels, 1)

        def forward(self, x, timesteps):
            # x: [batch, channels, length]
            # timesteps: [batch]
            time_emb = self.time_embed(timesteps.float().unsqueeze(-1))  # [batch, hidden_dim]

            h = torch.relu(self.conv1(x))
            h = h + time_emb.unsqueeze(-1)  # Add time embedding
            h = torch.relu(self.conv2(h))
            h = self.conv3(h)
            return h

    score_network = SimpleScoreNetwork(
        in_channels=config.input_dim,
        out_channels=config.input_dim,
        hidden_dim=config.hidden_dim
    )

    # Data shape for the model
    data_shape = (config.sequence_length, config.input_dim)

    # Create the main model
    model = DynamicInfoLattices(config, score_network, data_shape).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config (skip for now due to serialization issues)
    logger.info("Skipping config save due to serialization issues")
    
    # Simple training function
    def train_epoch(model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device here (not in dataset)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass using the DIL model properly
            try:
                # Create a mask for the input (all observed for training)
                mask = torch.ones_like(inputs, dtype=torch.bool, device=device)

                # Use the DIL model's forward method which handles the score network correctly
                outputs = model(inputs, mask)

                # Ensure output shape matches target shape (preserve gradients)
                if outputs.shape[1] > targets.shape[1]:
                    outputs = outputs[:, -targets.shape[1]:, :]
                elif outputs.shape[1] < targets.shape[1]:
                    # Pad if necessary (preserve gradients)
                    pad_size = targets.shape[1] - outputs.shape[1]
                    outputs = torch.cat([outputs, outputs[:, -1:, :].expand(-1, pad_size, -1)], dim=1)

                loss = criterion(outputs, targets)
            except Exception as e:
                logger.warning(f"Error in forward pass: {e}, skipping batch")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate_epoch(model, test_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move data to device here (not in dataset)
                inputs, targets = inputs.to(device), targets.to(device)

                try:
                    # Create a mask for the input (all observed for evaluation)
                    mask = torch.ones_like(inputs, dtype=torch.bool, device=device)

                    # Use the DIL model's forward method
                    outputs = model(inputs, mask)

                    # Take only the prediction length portion
                    if outputs.shape[1] > targets.shape[1]:
                        outputs = outputs[:, -targets.shape[1]:, :]
                    elif outputs.shape[1] < targets.shape[1]:
                        # Pad if necessary
                        pad_size = targets.shape[1] - outputs.shape[1]
                        outputs = torch.cat([outputs, outputs[:, -1:, :].repeat(1, pad_size, 1)], dim=1)

                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    logger.warning(f"Error in evaluation: {e}, skipping batch")
                    continue

        return total_loss / max(num_batches, 1)

    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}")

        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            test_loss = evaluate_epoch(model, test_loader, criterion, device)
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Test Loss: {test_loss:.6f}")

            # Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_path = output_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'config': config.__dict__
                }, best_model_path)
                logger.info(f"Saved best model with test loss: {best_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config.__dict__
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    logger.info("Training completed!")
    logger.info(f"Best test loss: {best_loss:.6f}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
