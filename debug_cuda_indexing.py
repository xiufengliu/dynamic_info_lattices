#!/usr/bin/env python3
"""
Minimal test case to isolate CUDA indexing issue in Dynamic Information Lattices
"""

import torch
import torch.nn as nn
import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/zhome/bb/9/101964/xiuli/dynamic_info_lattices')

from dynamic_info_lattices.config import DILConfig
from dynamic_info_lattices.core.dynamic_info_lattices import DynamicInfoLattices
from dynamic_info_lattices.core.hierarchical_lattice import HierarchicalLattice

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinimalScoreNetwork(nn.Module):
    """Minimal score network for debugging"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)  # Kernel size 1 for safety
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        logger.debug(f"MinimalScoreNetwork input: x.shape={x.shape}, timesteps.shape={timesteps.shape}")
        result = self.conv(x)
        logger.debug(f"MinimalScoreNetwork output: result.shape={result.shape}")
        return result

def test_cuda_indexing():
    """Test CUDA indexing with minimal setup"""
    
    logger.info("Starting CUDA indexing test...")
    
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create minimal configuration
    config = DILConfig()
    config.input_dim = 2  # Small number of channels
    config.sequence_length = 32  # Small sequence length
    config.prediction_length = 8
    config.batch_size = 4  # Small batch size
    config.num_diffusion_steps = 10  # Few diffusion steps
    config.max_scales = 2  # Only 2 scales to reduce complexity
    config.entropy_threshold = 0.1
    config.refinement_threshold = 0.05
    
    logger.info(f"Config: input_dim={config.input_dim}, seq_len={config.sequence_length}, batch_size={config.batch_size}")
    
    # Create minimal score network
    score_network = MinimalScoreNetwork(
        in_channels=config.input_dim,
        out_channels=config.input_dim
    ).to(device)
    
    # Create data shape
    data_shape = (config.sequence_length, config.input_dim)
    
    # Create DIL model
    logger.info("Creating DIL model...")
    model = DynamicInfoLattices(config, score_network, data_shape).to(device)
    
    # Create minimal test data
    logger.info("Creating test data...")
    batch_size = config.batch_size
    x = torch.randn(batch_size, config.sequence_length, config.input_dim, device=device)
    mask = torch.ones(batch_size, config.sequence_length, device=device)
    
    logger.info(f"Test data shapes: x={x.shape}, mask={mask.shape}")
    
    # Test hierarchical lattice construction
    logger.info("Testing hierarchical lattice construction...")
    try:
        lattice = model.lattice_constructor.construct_lattice(
            data_shape, config.max_scales
        )
        logger.info(f"Lattice constructed successfully with {len(lattice['all_nodes'])} nodes")
        
        # Print first few nodes for debugging
        for i, node in enumerate(lattice['all_nodes'][:5]):
            logger.debug(f"Node {i}: {node}")
            
    except Exception as e:
        logger.error(f"Lattice construction failed: {e}")
        return False
    
    # Test local region extraction
    logger.info("Testing local region extraction...")
    try:
        # Test with first few nodes
        for i, (t, f, s) in enumerate(lattice['all_nodes'][:3]):
            logger.debug(f"Testing extraction for node {i}: t={t}, f={f}, s={s}")
            local_region = model._extract_local_region(x, t, f, s)
            logger.debug(f"Extracted region shape: {local_region.shape}")
            
    except Exception as e:
        logger.error(f"Local region extraction failed: {e}")
        return False
    
    # Test score network call
    logger.info("Testing score network call...")
    try:
        # Use a small local region
        test_region = x[:, :4, :]  # First 4 time steps
        logger.debug(f"Test region shape: {test_region.shape}")
        
        # Transpose for score network
        test_region_transposed = test_region.transpose(-2, -1)
        logger.debug(f"Transposed region shape: {test_region_transposed.shape}")
        
        # Create timesteps
        timesteps = torch.full((batch_size,), 5, device=device, dtype=torch.long)
        logger.debug(f"Timesteps shape: {timesteps.shape}")
        
        # Call score network
        with torch.no_grad():
            score_output = score_network(test_region_transposed, timesteps)
            logger.debug(f"Score output shape: {score_output.shape}")
            
    except Exception as e:
        logger.error(f"Score network call failed: {e}")
        return False
    
    # Test full forward pass with minimal data
    logger.info("Testing full forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            # Use very small input
            small_x = x[:2, :8, :]  # 2 samples, 8 time steps
            small_mask = mask[:2, :8]
            logger.info(f"Small input shapes: x={small_x.shape}, mask={small_mask.shape}")
            
            output = model(small_x, small_mask)
            logger.info(f"Forward pass successful! Output shape: {output.shape}")
            
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    logger.info("All tests passed!")
    return True

if __name__ == "__main__":
    success = test_cuda_indexing()
    if success:
        print("✅ CUDA indexing test passed")
        sys.exit(0)
    else:
        print("❌ CUDA indexing test failed")
        sys.exit(1)
