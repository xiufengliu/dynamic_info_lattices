"""
Score Network Implementation

Implements the 1D U-Net architecture for score function estimation
as specified in the supplementary material.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


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


class ResidualBlock(nn.Module):
    """Residual block with group normalization"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Adjust groups to be compatible with channel count
        groups_1 = min(groups, in_channels)
        while in_channels % groups_1 != 0 and groups_1 > 1:
            groups_1 -= 1

        groups_2 = min(groups, out_channels)
        while out_channels % groups_2 != 0 and groups_2 > 1:
            groups_2 -= 1

        self.norm1 = nn.GroupNorm(groups_1, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(groups_2, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        # Adjust groups to be compatible with channel count
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv.unbind(1)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.contiguous().view(B, C, L)
        
        out = self.proj_out(out)
        
        return x + out


class DownBlock(nn.Module):
    """Downsampling block with residual connections"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        downsample: bool = True,
        attention: bool = False
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])
        
        self.attention = AttentionBlock(out_channels) if attention else None
        
        if downsample:
            self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = x
        
        for layer in self.layers:
            x = layer(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        if self.downsample is not None:
            skip = x
            x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        upsample: bool = True,
        attention: bool = False
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])
        
        self.attention = AttentionBlock(out_channels) if attention else None
        
        if upsample:
            self.upsample = nn.ConvTranspose1d(out_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.upsample = None
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Handle spatial dimension mismatch between x and skip
        if x.shape[-1] != skip.shape[-1]:
            # Interpolate skip to match x's spatial dimension
            skip = torch.nn.functional.interpolate(
                skip, size=x.shape[-1], mode='linear', align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        
        for layer in self.layers:
            x = layer(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x


class ScoreNetwork(nn.Module):
    """
    1D U-Net Score Network for Diffusion Models
    
    Architecture specifications from supplementary material:
    - Encoder: 6 downsampling blocks with channels [64, 128, 256, 512, 768, 1024]
    - Decoder: 6 upsampling blocks with skip connections and feature fusion
    - Attention: Multi-head self-attention at multiple scales
    - Activation: SiLU activation with learnable parameters
    - Normalization: Adaptive group normalization with 8 groups
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 12, 16),
        num_heads: int = 8,
        time_emb_dim: Optional[int] = None
    ):
        super().__init__()
        
        if time_emb_dim is None:
            time_emb_dim = model_channels * 4
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    DownBlock(
                        ch,
                        out_ch,
                        time_emb_dim,
                        num_layers=1,
                        downsample=False,
                        attention=level in attention_resolutions
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    DownBlock(
                        ch,
                        ch,
                        time_emb_dim,
                        num_layers=1,
                        downsample=True,
                        attention=False
                    )
                )
                input_block_chans.append(ch)
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_emb_dim)
        )
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()

                self.up_blocks.append(
                    UpBlock(
                        ch,
                        out_ch,
                        skip_ch,
                        time_emb_dim,
                        num_layers=1,
                        upsample=i == num_res_blocks and level != 0,
                        attention=level in attention_resolutions
                    )
                )
                ch = out_ch
        
        # Output projection with adaptive GroupNorm
        groups = min(8, ch)
        while ch % groups != 0 and groups > 1:
            groups -= 1
        self.output_proj = nn.Sequential(
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv1d(ch, out_channels, 3, padding=1)
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
        
        # Encoder with skip connections
        skips = []
        
        for block in self.down_blocks:
            h, skip = block(h, time_emb)
            skips.append(skip)
        
        # Middle block
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        # Decoder with skip connections
        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, time_emb)
        
        # Output projection
        output = self.output_proj(h)
        
        return output
