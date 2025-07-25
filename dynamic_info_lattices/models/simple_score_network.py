"""
Simple Score Network Implementation that works correctly with tensor dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time steps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleResidualBlock(nn.Module):
    """Simple residual block with group normalization"""
    
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        
        # Adjust groups to be compatible with channel count
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )
        
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_out = self.time_mlp(time_emb)
        h = h + time_out[:, :, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h  # Residual connection

class SimpleScoreNetwork(nn.Module):
    """
    Simple Score Network that works correctly with tensor dimensions
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        num_blocks: int = 4,
        time_emb_dim: Optional[int] = None
    ):
        super().__init__()
        
        if time_emb_dim is None:
            time_emb_dim = model_channels * 4
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, model_channels, 3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            SimpleResidualBlock(model_channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        groups = min(8, model_channels)
        while model_channels % groups != 0 and groups > 1:
            groups -= 1
        
        self.output_proj = nn.Sequential(
            nn.GroupNorm(groups, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the score network
        
        Args:
            x: Input tensor [batch_size, channels, length]
            timesteps: Diffusion timesteps [batch_size]
            
        Returns:
            score: Estimated score function [batch_size, channels, length]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input projection
        h = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            h = block(h, time_emb)
        
        # Output projection
        output = self.output_proj(h)
        
        return output
