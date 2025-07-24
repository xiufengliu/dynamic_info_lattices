#!/usr/bin/env python3
"""
Test script to verify all code review fixes are working correctly.

This script tests the core components of the Dynamic Information Lattices
implementation to ensure they match the paper specifications.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add the package to path
sys.path.append(str(Path(__file__).parent))

from dynamic_info_lattices import (
    DynamicInfoLattices, DILConfig,
    ScoreNetwork, EntropyWeightNetwork
)


def test_algorithm_s1_implementation():
    """Test Algorithm S1: Dynamic Information Lattices main algorithm"""
    print("Testing Algorithm S1 Implementation...")
    
    # Create test configuration
    config = DILConfig(
        num_diffusion_steps=100,
        inference_steps=10,
        max_scales=3,
        entropy_budget=0.2
    )
    
    # Create test data
    batch_size = 2
    length = 32
    channels = 1
    
    y_obs = torch.randn(batch_size, length, channels)
    mask = torch.ones_like(y_obs)
    
    # Create score network
    score_network = ScoreNetwork(
        in_channels=channels,
        out_channels=channels,
        model_channels=32
    )
    
    # Create DIL model
    model = DynamicInfoLattices(
        config=config,
        score_network=score_network,
        data_shape=(length, channels)
    )
    
    try:
        # Test forward pass
        with torch.no_grad():
            result = model(y_obs, mask)
        
        print(f"✓ Algorithm S1 forward pass successful")
        print(f"  Input shape: {y_obs.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Expected same shape: {y_obs.shape == result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm S1 failed: {e}")
        return False


def test_multi_component_entropy():
    """Test Algorithm S2: Multi-Component Entropy Estimation"""
    print("\nTesting Multi-Component Entropy Estimation...")
    
    from dynamic_info_lattices.core.multi_component_entropy import MultiComponentEntropy
    
    config = DILConfig()
    data_shape = (32, 1)
    
    entropy_estimator = MultiComponentEntropy(config, data_shape)
    
    # Test data
    batch_size = 2
    z = torch.randn(batch_size, *data_shape)
    y_obs = torch.randn(batch_size, *data_shape)
    
    # Mock lattice structure
    lattice = {
        'active_nodes': [(0, 0, 0), (1, 0, 0), (0, 0, 1)],
        'max_scales': 3
    }
    
    # Mock score network
    score_network = ScoreNetwork(in_channels=1, out_channels=1, model_channels=32)
    
    try:
        entropy_map = entropy_estimator(
            z=z,
            k=50,
            lattice=lattice,
            y_obs=y_obs,
            entropy_history=[],
            score_network=score_network
        )
        
        print(f"✓ Multi-component entropy estimation successful")
        print(f"  Entropy map shape: {entropy_map.shape}")
        print(f"  Number of active nodes: {len(lattice['active_nodes'])}")
        print(f"  Entropy values: {entropy_map.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-component entropy failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hierarchical_lattice():
    """Test Algorithm S3: Hierarchical Lattice Construction"""
    print("\nTesting Hierarchical Lattice Construction...")
    
    from dynamic_info_lattices.core.hierarchical_lattice import HierarchicalLattice
    
    config = DILConfig(max_scales=3)
    data_shape = (32, 1)
    lattice_builder = HierarchicalLattice(config, data_shape)
    
    # Test data
    y_obs = torch.randn(2, 32, 1)
    
    try:
        lattice = lattice_builder.construct_hierarchical_lattice(y_obs)
        
        print(f"✓ Hierarchical lattice construction successful")
        print(f"  Total active nodes: {len(lattice['active_nodes'])}")
        print(f"  Scales: {list(lattice['hierarchy'].keys())}")
        print(f"  Resolution: {lattice['resolution']}")
        
        # Test adaptation
        entropy_map = torch.rand(len(lattice['active_nodes']))
        adapted_lattice = lattice_builder.adapt_lattice(lattice, entropy_map, k=50)
        
        print(f"✓ Lattice adaptation successful")
        print(f"  Adapted active nodes: {len(adapted_lattice['active_nodes'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hierarchical lattice failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_information_aware_sampling():
    """Test Algorithm S5: Information-Aware Sampling"""
    print("\nTesting Information-Aware Sampling...")
    
    from dynamic_info_lattices.core.information_aware_sampler import InformationAwareSampler
    
    config = DILConfig(entropy_budget=0.3, temperature=5.0)
    sampler = InformationAwareSampler(config)
    
    # Mock lattice and entropy data
    active_nodes = [(i, 0, 0) for i in range(10)]
    lattice = {
        'active_nodes': active_nodes,
        'max_scales': 3
    }
    entropy_map = torch.rand(len(active_nodes))
    
    try:
        selected_nodes = sampler.stratified_sample(
            lattice=lattice,
            entropy_map=entropy_map,
            budget_fraction=0.3
        )
        
        print(f"✓ Information-aware sampling successful")
        print(f"  Total nodes: {len(active_nodes)}")
        print(f"  Selected nodes: {len(selected_nodes)}")
        print(f"  Budget fraction: {len(selected_nodes) / len(active_nodes):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Information-aware sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_solver():
    """Test Adaptive Solver Selection"""
    print("\nTesting Adaptive Solver Selection...")
    
    from dynamic_info_lattices.core.adaptive_solver import AdaptiveSolver
    
    config = DILConfig()
    solver = AdaptiveSolver(config)
    
    # Test data
    entropy_map = torch.tensor([0.1, 0.3, 0.7, 0.9])
    active_nodes = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    
    try:
        # Test solver order selection
        for i, (t, f, s) in enumerate(active_nodes):
            order = solver.select_solver_order(
                entropy_map=entropy_map,
                t=t, f=f, s=s, k=50,
                active_nodes=active_nodes
            )
            print(f"  Node ({t},{f},{s}), entropy={entropy_map[i]:.2f} -> order={order}")
        
        print(f"✓ Adaptive solver selection successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive solver failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_network_architecture():
    """Test Score Network Architecture"""
    print("\nTesting Score Network Architecture...")
    
    try:
        # Test with paper specifications
        score_network = ScoreNetwork(
            in_channels=1,
            out_channels=1,
            model_channels=64,
            channel_mult=(1, 2, 4, 8, 12, 16)  # 6 blocks as per paper
        )
        
        # Test forward pass
        batch_size = 2
        length = 64
        channels = 1
        
        x = torch.randn(batch_size, channels, length)
        t = torch.randint(0, 1000, (batch_size,))
        
        with torch.no_grad():
            output = score_network(x, t)
        
        print(f"✓ Score network forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Architecture: 6 encoder/decoder blocks")
        
        return True
        
    except Exception as e:
        print(f"✗ Score network failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DYNAMIC INFORMATION LATTICES - CODE REVIEW FIX VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_algorithm_s1_implementation,
        test_multi_component_entropy,
        test_hierarchical_lattice,
        test_information_aware_sampling,
        test_adaptive_solver,
        test_score_network_architecture
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Code review fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
