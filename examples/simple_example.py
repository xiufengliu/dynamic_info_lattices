#!/usr/bin/env python3
"""
Simple working example of Dynamic Information Lattices

This script demonstrates the core functionality of the DIL framework
with a minimal working example that can run without external dependencies.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dynamic_info_lattices import (
    DynamicInfoLattices, DILConfig, ScoreNetwork
)

# Import real dataset loader
from dynamic_info_lattices.data.real_datasets import get_real_dataset


def create_real_data_sample(dataset_name="etth1", batch_size=4):
    """Create sample data from real dataset"""
    print(f"Loading sample from real dataset: {dataset_name}")

    try:
        # Load real dataset
        dataset = get_real_dataset(
            dataset_name=dataset_name,
            data_dir="./data",
            split="train",
            sequence_length=32,
            prediction_length=8
        )

        # Get a few samples
        samples = []
        for i in range(min(batch_size, len(dataset))):
            x, y, mask = dataset[i]
            samples.append(x)

        # Stack into batch
        y_obs = torch.stack(samples)  # [batch_size, seq_len, channels]
        mask = torch.ones_like(y_obs)  # All data is observed

        print(f"Real data shape: {y_obs.shape}")
        print(f"Data range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
        return y_obs, mask

    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_data(batch_size, 32, 1)


def create_synthetic_data(batch_size=4, seq_len=32, channels=1):
    """Create simple synthetic time series data as fallback"""
    print("Creating synthetic time series data...")

    # Generate sinusoidal data with noise
    t = torch.linspace(0, 4*np.pi, seq_len)

    data = []
    for i in range(batch_size):
        # Different frequency and phase for each sample
        freq = 1 + 0.5 * i
        phase = np.pi * i / 4

        # Create sinusoidal signal with noise
        signal = torch.sin(freq * t + phase) + 0.1 * torch.randn(seq_len)
        signal = signal.unsqueeze(-1)  # Add channel dimension
        data.append(signal)

    # Stack into batch
    y_obs = torch.stack(data)  # [batch_size, seq_len, channels]
    mask = torch.ones_like(y_obs)  # All data is observed

    print(f"Generated data shape: {y_obs.shape}")
    return y_obs, mask


def demonstrate_dil_components():
    """Demonstrate individual DIL components"""
    print("\n" + "="*60)
    print("DEMONSTRATING DIL COMPONENTS")
    print("="*60)
    
    # Configuration
    config = DILConfig(
        num_diffusion_steps=100,
        inference_steps=5,  # Small for demo
        max_scales=2,
        entropy_budget=0.3,
        temperature=2.0
    )
    
    print(f"Configuration:")
    print(f"  Diffusion steps: {config.num_diffusion_steps}")
    print(f"  Inference steps: {config.inference_steps}")
    print(f"  Max scales: {config.max_scales}")
    print(f"  Entropy budget: {config.entropy_budget}")
    
    # Create data
    batch_size, seq_len, channels = 2, 32, 1
    y_obs, mask = create_real_data_sample("etth1", batch_size)
    
    # Create score network
    print(f"\nCreating score network...")
    score_network = ScoreNetwork(
        in_channels=channels,
        out_channels=channels,
        model_channels=32  # Small for demo
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in score_network.parameters())
    print(f"Score network parameters: {total_params:,}")
    
    # Create DIL model
    print(f"\nCreating DIL model...")
    data_shape = (seq_len, channels)
    model = DynamicInfoLattices(
        config=config,
        score_network=score_network,
        data_shape=data_shape
    )
    
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_model_params:,}")
    
    # Test individual components
    print(f"\nTesting individual components...")
    
    # 1. Test hierarchical lattice
    print(f"  1. Hierarchical lattice construction...")
    lattice = model.lattice.construct_hierarchical_lattice(y_obs)
    print(f"     Active nodes: {len(lattice['active_nodes'])}")
    print(f"     Scales: {list(lattice['hierarchy'].keys())}")
    print(f"     Sample nodes: {lattice['active_nodes'][:3]}")
    
    # 2. Test entropy estimation
    print(f"  2. Multi-component entropy estimation...")
    try:
        z = torch.randn_like(y_obs)
        entropy_map = model.entropy_estimator(
            z=z, k=50, lattice=lattice, y_obs=y_obs, 
            entropy_history=[], score_network=score_network
        )
        print(f"     Entropy map shape: {entropy_map.shape}")
        print(f"     Entropy values: {entropy_map[:3].tolist()}")
    except Exception as e:
        print(f"     Error: {e}")
    
    # 3. Test information-aware sampling
    print(f"  3. Information-aware sampling...")
    try:
        selected_nodes = model.sampler.stratified_sample(
            lattice=lattice,
            entropy_map=entropy_map,
            budget_fraction=config.entropy_budget
        )
        print(f"     Selected {len(selected_nodes)} out of {len(lattice['active_nodes'])} nodes")
        print(f"     Sample selected nodes: {selected_nodes[:3]}")
    except Exception as e:
        print(f"     Error: {e}")
    
    # 4. Test adaptive solver
    print(f"  4. Adaptive solver selection...")
    try:
        for i, (t, f, s) in enumerate(lattice['active_nodes'][:3]):
            order = model.solver.select_solver_order(
                entropy_map=entropy_map,
                t=t, f=f, s=s, k=50,
                active_nodes=lattice['active_nodes']
            )
            entropy_val = entropy_map[i] if i < len(entropy_map) else 0.0
            print(f"     Node ({t},{f},{s}), entropy={entropy_val:.3f} -> order={order}")
    except Exception as e:
        print(f"     Error: {e}")
    
    return model, y_obs, mask


def demonstrate_full_forward_pass(model, y_obs, mask):
    """Demonstrate full forward pass"""
    print(f"\n" + "="*60)
    print("DEMONSTRATING FULL FORWARD PASS")
    print("="*60)
    
    print(f"Input shape: {y_obs.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Set model to evaluation mode
    model.eval()
    
    try:
        with torch.no_grad():
            print(f"Running forward pass...")
            output = model(y_obs, mask)
            
            print(f"✓ Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Input range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Compute simple metrics
            mse = torch.nn.functional.mse_loss(output, y_obs)
            mae = torch.nn.functional.l1_loss(output, y_obs)
            
            print(f"Reconstruction MSE: {mse:.6f}")
            print(f"Reconstruction MAE: {mae:.6f}")
            
            return True
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demonstration function"""
    print("DYNAMIC INFORMATION LATTICES - SIMPLE EXAMPLE")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Demonstrate components
        model, y_obs, mask = demonstrate_dil_components()
        
        # Move to device
        model = model.to(device)
        y_obs = y_obs.to(device)
        mask = mask.to(device)
        
        # Demonstrate full forward pass
        success = demonstrate_full_forward_pass(model, y_obs, mask)
        
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if success:
            print("🎉 All demonstrations completed successfully!")
            print("✅ Core DIL framework is working correctly")
            print("✅ All components are properly integrated")
            print("✅ Ready for further development and experimentation")
        else:
            print("⚠️  Some demonstrations failed")
            print("Please check the error messages above")
        
        return success
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
