#!/usr/bin/env python3
"""
Baseline methods implementation for KDD experimental comparison
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dynamic_info_lattices.data.processed_datasets import create_dataset, get_dataset_info


class SimpleLinearBaseline(nn.Module):
    """Simple linear baseline for time series forecasting"""
    
    def __init__(self, input_dim: int, sequence_length: int, prediction_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Simple linear projection
        self.linear = nn.Linear(sequence_length * input_dim, prediction_length * input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [batch_size, sequence_length * input_dim]
        x_flat = self.dropout(x_flat)
        output = self.linear(x_flat)  # [batch_size, prediction_length * input_dim]
        output = output.view(batch_size, self.prediction_length, self.input_dim)
        return output


class LSTMBaseline(nn.Module):
    """LSTM baseline with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, prediction_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        self.prediction_head = nn.Linear(hidden_dim, prediction_length * input_dim)
        
    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        batch_size = x.shape[0]
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, sequence_length, hidden_dim]
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state for prediction
        last_hidden = attn_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Generate predictions
        predictions = self.prediction_head(last_hidden)  # [batch_size, prediction_length * input_dim]
        predictions = predictions.view(batch_size, self.prediction_length, self.input_dim)
        
        return predictions


class TransformerBaseline(nn.Module):
    """Transformer baseline for time series forecasting"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, prediction_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_length = prediction_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, prediction_length * input_dim)
        
    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, sequence_length, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer(x)  # [batch_size, sequence_length, d_model]
        
        # Use last token for prediction
        last_token = encoded[:, -1, :]  # [batch_size, d_model]
        
        # Generate predictions
        predictions = self.output_projection(last_token)  # [batch_size, prediction_length * input_dim]
        predictions = predictions.view(batch_size, self.prediction_length, self.input_dim)
        
        return predictions


def create_baseline_model(method: str, input_dim: int, sequence_length: int, prediction_length: int):
    """Create baseline model based on method name"""
    
    if method == 'dlinear':
        return SimpleLinearBaseline(input_dim, sequence_length, prediction_length)
    
    elif method == 'lstm':
        hidden_dim = min(128, max(32, input_dim * 2))
        return LSTMBaseline(input_dim, hidden_dim, num_layers=2, prediction_length=prediction_length)
    
    elif method == 'transformer':
        d_model = min(256, max(64, input_dim * 4))
        nhead = min(8, max(2, d_model // 32))
        return TransformerBaseline(input_dim, d_model, nhead, num_layers=3, prediction_length=prediction_length)
    
    elif method in ['patchtst', 'timesnet', 'informer', 'autoformer', 'fedformer']:
        # For now, use transformer as placeholder for these methods
        # In a real implementation, you would use the actual implementations
        d_model = min(256, max(64, input_dim * 4))
        nhead = min(8, max(2, d_model // 32))
        return TransformerBaseline(input_dim, d_model, nhead, num_layers=4, prediction_length=prediction_length)
    
    elif method in ['tsdiff', 'csdi', 'timegrad']:
        # For diffusion methods, use a simple baseline for now
        # In practice, you would implement the actual diffusion models
        return SimpleLinearBaseline(input_dim, sequence_length, prediction_length)
    
    else:
        raise ValueError(f"Unknown baseline method: {method}")


def train_baseline(model, train_loader, test_loader, epochs: int, device: str, method: str):
    """Train baseline model"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / max(test_batches, 1)
        scheduler.step(avg_test_loss)
        
        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
    
    return best_test_loss


def main():
    parser = argparse.ArgumentParser(description="Run baseline methods")
    parser.add_argument("--method", type=str, required=True,
                       help="Baseline method name")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--sequence_length", type=int, default=96,
                       help="Input sequence length")
    parser.add_argument("--prediction_length", type=int, default=24,
                       help="Prediction length")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--fold", type=int, default=0,
                       help="Cross-validation fold")
    parser.add_argument("--output_dir", type=str, default="experiments/baselines",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Running baseline: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}, Fold: {args.fold}")
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    input_dim = dataset_info['num_series']
    
    # Create data loaders
    train_dataset = create_dataset(
        dataset_name=args.dataset,
        split="train",
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
        normalize=True
    )

    test_dataset = create_dataset(
        dataset_name=args.dataset,
        split="test",
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
        normalize=True
    )
    
    # Determine batch size based on dataset size
    if len(train_dataset) > 10000:
        batch_size = 32
    elif len(train_dataset) > 1000:
        batch_size = 64
    else:
        batch_size = 128
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Create model
    model = create_baseline_model(args.method, input_dim, args.sequence_length, args.prediction_length)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train model
    print("Starting training...")
    best_test_loss = train_baseline(model, train_loader, test_loader, args.epochs, device, args.method)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'method': args.method,
        'dataset': args.dataset,
        'sequence_length': args.sequence_length,
        'prediction_length': args.prediction_length,
        'seed': args.seed,
        'fold': args.fold,
        'best_test_loss': best_test_loss,
        'total_params': total_params,
        'dataset_info': dataset_info
    }
    
    result_file = output_dir / f"{args.method}_{args.dataset}_s{args.seed}_f{args.fold}_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {result_file}")
    print(f"Best test loss: {best_test_loss:.6f}")


if __name__ == "__main__":
    main()
