"""
Entropy Weight Network

Implements the adaptive weight network for entropy component combination
as specified in the supplementary material.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EntropyWeightNetwork(nn.Module):
    """
    Adaptive Weight Network for Entropy Component Combination
    
    Architecture from supplementary material:
    - Input: Concatenated features [z, k_embed, pos_enc, local_stats]
    - Architecture: 3-layer MLP with [256, 128, 64] hidden units
    - Output: 5-dimensional weight vector with softmax normalization
    - Regularization: Dropout (0.1) and weight decay (1e-5)
    """
    
    def __init__(
        self,
        data_shape: Tuple[int, ...],
        k_embed_dim: int = 64,
        pos_enc_dim: int = 5,
        local_stats_dim: int = 10,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        num_entropy_components: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.data_shape = data_shape
        self.k_embed_dim = k_embed_dim
        self.pos_enc_dim = pos_enc_dim
        self.local_stats_dim = local_stats_dim
        self.num_entropy_components = num_entropy_components
        
        # Calculate input dimension
        z_features_dim = np.prod(data_shape)
        self.input_dim = z_features_dim + k_embed_dim + pos_enc_dim + local_stats_dim
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_entropy_components))
        
        self.mlp = nn.Sequential(*layers)
        
        # Softmax for probability normalization
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        z_local: torch.Tensor,
        k: int,
        t: int,
        f: int,
        s: int,
        max_scales: int = 4
    ) -> torch.Tensor:
        """
        Compute adaptive weights for entropy components
        
        Args:
            z_local: Local latent region
            k: Current diffusion step
            t: Time coordinate
            f: Frequency coordinate
            s: Scale coordinate
            max_scales: Maximum number of scales
            
        Returns:
            weights: Adaptive weights for entropy components [num_entropy_components]
        """
        batch_size = z_local.shape[0]
        device = z_local.device
        
        # Extract features from local region
        z_features = self._extract_z_features(z_local)
        
        # Time step embedding
        k_embed = self._sinusoidal_embedding(k, self.k_embed_dim, device)
        
        # Positional encoding
        pos_enc = self._positional_encoding(t, f, s, max_scales, device)
        
        # Local statistics
        local_stats = self._compute_local_statistics(z_local)
        
        # Concatenate all features
        features = torch.cat([
            z_features,
            k_embed.unsqueeze(0).expand(batch_size, -1),
            pos_enc.unsqueeze(0).expand(batch_size, -1),
            local_stats
        ], dim=-1)
        
        # Forward through MLP
        logits = self.mlp(features)
        
        # Apply softmax to get normalized weights
        weights = self.softmax(logits)
        
        return weights
    
    def _extract_z_features(self, z_local: torch.Tensor) -> torch.Tensor:
        """Extract features from local latent region"""
        batch_size = z_local.shape[0]
        
        # Flatten spatial dimensions
        z_flat = z_local.view(batch_size, -1)
        
        # Pad or truncate to expected size
        expected_size = np.prod(self.data_shape)
        current_size = z_flat.shape[1]
        
        if current_size < expected_size:
            # Pad with zeros
            padding = torch.zeros(batch_size, expected_size - current_size, device=z_local.device)
            z_features = torch.cat([z_flat, padding], dim=1)
        elif current_size > expected_size:
            # Truncate
            z_features = z_flat[:, :expected_size]
        else:
            z_features = z_flat
        
        return z_features
    
    def _sinusoidal_embedding(self, k: int, dim: int, device: torch.device) -> torch.Tensor:
        """Create sinusoidal positional embedding for diffusion step"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = k * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=0)
        
        # Ensure correct dimension
        if emb.shape[0] < dim:
            padding = torch.zeros(dim - emb.shape[0], device=device)
            emb = torch.cat([emb, padding], dim=0)
        elif emb.shape[0] > dim:
            emb = emb[:dim]
        
        return emb
    
    def _positional_encoding(
        self,
        t: int,
        f: int,
        s: int,
        max_scales: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create positional encoding for lattice coordinates"""
        # Temporal encoding
        t_sin = np.sin(0.1 * t)
        t_cos = np.cos(0.1 * t)
        
        # Frequency encoding
        f_sin = np.sin(0.1 * f)
        f_cos = np.cos(0.1 * f)
        
        # Scale encoding (normalized)
        s_norm = s / max_scales
        
        pos_enc = torch.tensor([
            t_sin, t_cos, f_sin, f_cos, s_norm
        ], device=device, dtype=torch.float32)
        
        return pos_enc
    
    def _compute_local_statistics(self, z_local: torch.Tensor) -> torch.Tensor:
        """Compute local statistics of the latent region"""
        batch_size = z_local.shape[0]
        
        # Compute statistics along spatial dimensions
        stats_list = []
        
        for b in range(batch_size):
            z_b = z_local[b]
            
            # Basic statistics
            mean_val = torch.mean(z_b)
            std_val = torch.std(z_b)
            min_val = torch.min(z_b)
            max_val = torch.max(z_b)
            median_val = torch.median(z_b)
            var_val = torch.var(z_b)
            
            # Advanced statistics
            norm_val = torch.norm(z_b)
            l1_norm = torch.sum(torch.abs(z_b))
            pos_count = torch.sum(z_b > 0).float()
            neg_count = torch.sum(z_b < 0).float()
            
            stats = torch.stack([
                mean_val, std_val, min_val, max_val, median_val,
                var_val, norm_val, l1_norm, pos_count, neg_count
            ])
            
            stats_list.append(stats)
        
        local_stats = torch.stack(stats_list, dim=0)
        
        return local_stats


class AdaptiveWeightScheduler:
    """Scheduler for adaptive weight learning"""
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        decay_rate: float = 0.95
    ):
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self.current_temperature = initial_temperature
    
    def step(self, epoch: int) -> float:
        """Update temperature for weight learning"""
        self.current_temperature = max(
            self.min_temperature,
            self.initial_temperature * (self.decay_rate ** epoch)
        )
        return self.current_temperature
    
    def get_temperature(self) -> float:
        """Get current temperature"""
        return self.current_temperature


class WeightRegularizer:
    """Regularization for entropy weights"""
    
    def __init__(
        self,
        entropy_reg: float = 0.01,
        diversity_reg: float = 0.1,
        sparsity_reg: float = 0.05
    ):
        self.entropy_reg = entropy_reg
        self.diversity_reg = diversity_reg
        self.sparsity_reg = sparsity_reg
    
    def compute_regularization_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss for weights"""
        batch_size = weights.shape[0]
        
        # Entropy regularization (encourage diversity)
        entropy_loss = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))
        
        # Diversity regularization (encourage different weights across batch)
        mean_weights = torch.mean(weights, dim=0)
        diversity_loss = -torch.sum(mean_weights * torch.log(mean_weights + 1e-8))
        
        # Sparsity regularization (encourage some specialization)
        sparsity_loss = torch.mean(torch.sum(weights ** 2, dim=1))
        
        total_loss = (
            self.entropy_reg * entropy_loss +
            self.diversity_reg * diversity_loss +
            self.sparsity_reg * sparsity_loss
        )
        
        return total_loss


class WeightAnalyzer:
    """Analyzer for entropy weight patterns"""
    
    def __init__(self):
        self.weight_history = []
        self.component_usage = {i: [] for i in range(5)}
    
    def analyze_weights(self, weights: torch.Tensor, step: int) -> dict:
        """Analyze weight patterns"""
        # Average weights across batch
        avg_weights = torch.mean(weights, dim=0)
        
        # Record component usage
        for i in range(5):
            self.component_usage[i].append(avg_weights[i].item())
        
        # Compute statistics
        weight_entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-8))
        dominant_component = torch.argmax(avg_weights).item()
        weight_variance = torch.var(avg_weights)
        
        analysis = {
            'step': step,
            'avg_weights': avg_weights.tolist(),
            'weight_entropy': weight_entropy.item(),
            'dominant_component': dominant_component,
            'weight_variance': weight_variance.item()
        }
        
        self.weight_history.append(analysis)
        
        return analysis
    
    def get_component_trends(self) -> dict:
        """Get trends for each entropy component"""
        trends = {}
        
        for component, usage in self.component_usage.items():
            if usage:
                trends[component] = {
                    'mean_usage': np.mean(usage),
                    'std_usage': np.std(usage),
                    'trend': np.polyfit(range(len(usage)), usage, 1)[0] if len(usage) > 1 else 0.0
                }
        
        return trends
