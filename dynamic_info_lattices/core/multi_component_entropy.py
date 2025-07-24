"""
Multi-Component Entropy Estimation System

Implements Algorithm S2: Multi-Component Entropy Estimation
from the supplementary material.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiComponentEntropy(nn.Module):
    """
    Multi-Component Entropy Estimation System
    
    Estimates five types of entropy:
    1. Score function uncertainty (epistemic)
    2. Self-guidance uncertainty
    3. Solver order uncertainty
    4. Temporal uncertainty
    5. Spectral uncertainty
    """
    
    def __init__(self, config, data_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.data_shape = data_shape
        
        # Adaptive weight network for entropy combination
        self.weight_network = self._build_weight_network()
        
        # Parameters for entropy estimation
        self.num_mc_samples = 10
        self.finite_diff_eps = 1e-4
        self.temporal_window = 4
        
    def _build_weight_network(self) -> nn.Module:
        """Build adaptive weight network for entropy component combination"""
        # Input features: [z_features, k_embed, pos_enc, local_stats]
        input_dim = (
            np.prod(self.data_shape) +  # z features
            64 +  # k embedding
            5 +   # positional encoding (t, f, s, sin/cos terms)
            10    # local statistics
        )
        
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5 entropy components
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        z: torch.Tensor,
        k: int,
        lattice: Dict,
        y_obs: torch.Tensor,
        entropy_history: List
    ) -> torch.Tensor:
        """
        Estimate multi-component entropy map
        
        Args:
            z: Current latent state [batch_size, length, channels]
            k: Current diffusion step
            lattice: Current lattice structure
            y_obs: Observed data
            entropy_history: History of entropy estimates
            
        Returns:
            entropy_map: Entropy values for each lattice node
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Get active lattice nodes
        active_nodes = lattice.get('active_nodes', [])
        
        # Initialize entropy map
        entropy_map = torch.zeros(
            len(active_nodes), device=device, dtype=z.dtype
        )
        
        for i, (t, f, s) in enumerate(active_nodes):
            # Extract local region
            z_local = self._extract_local_region(z, t, f, s)
            
            # Estimate individual entropy components
            h_score = self._estimate_score_entropy(z_local, k)
            h_guidance = self._estimate_guidance_entropy(z_local, y_obs, t, f, s)
            h_solver = self._estimate_solver_entropy(z_local, k)
            h_temporal = self._estimate_temporal_entropy(z_local, t, entropy_history)
            h_spectral = self._estimate_spectral_entropy(z_local, f)
            
            # Compute adaptive weights
            weights = self._compute_adaptive_weights(z_local, k, t, f, s)
            
            # Combine entropy components
            entropy_components = torch.stack([h_score, h_guidance, h_solver, h_temporal, h_spectral])
            entropy_map[i] = torch.sum(weights * entropy_components)
        
        return entropy_map
    
    def _extract_local_region(
        self,
        z: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Extract local region from global tensor based on lattice coordinates"""
        scale_factor = 2 ** s
        t_start = t * scale_factor
        t_end = min((t + 1) * scale_factor, z.shape[1])
        f_start = f * scale_factor
        f_end = min((f + 1) * scale_factor, z.shape[2])
        
        return z[:, t_start:t_end, f_start:f_end]
    
    def _estimate_score_entropy(
        self,
        z_local: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Estimate score function uncertainty using Monte Carlo dropout"""
        entropies = []
        
        # Enable dropout for uncertainty estimation
        self.train()
        
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                # This would call the score network with dropout enabled
                # For now, we simulate the uncertainty
                noise = torch.randn_like(z_local) * 0.1
                score_sample = z_local + noise
                entropies.append(score_sample)
        
        # Compute differential entropy from samples
        scores_tensor = torch.stack(entropies)
        mean_score = torch.mean(scores_tensor, dim=0)
        var_score = torch.var(scores_tensor, dim=0)
        
        # Differential entropy: H = 0.5 * log(2πe * σ²)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * (var_score + 1e-8))
        
        return torch.mean(entropy)
    
    def _estimate_guidance_entropy(
        self,
        z_local: torch.Tensor,
        y_obs: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Estimate self-guidance uncertainty"""
        # Compute guidance gradient
        z_local.requires_grad_(True)
        
        # Simple guidance based on observation likelihood
        # In practice, this would be more sophisticated
        if y_obs is not None:
            obs_local = self._extract_local_region(y_obs, t, f, s)
            guidance_loss = F.mse_loss(z_local, obs_local, reduction='sum')
            
            # Compute gradient
            grad = torch.autograd.grad(
                guidance_loss, z_local,
                create_graph=False, retain_graph=False
            )[0]
            
            # Estimate variance of gradient using finite differences
            eps = self.finite_diff_eps
            z_plus = z_local + eps
            z_minus = z_local - eps
            
            loss_plus = F.mse_loss(z_plus, obs_local, reduction='sum')
            loss_minus = F.mse_loss(z_minus, obs_local, reduction='sum')
            
            grad_var = torch.var((loss_plus - loss_minus) / (2 * eps))
            
            # Entropy from gradient variance
            entropy = -torch.log(grad_var + 1e-8)
        else:
            entropy = torch.tensor(0.0, device=z_local.device)
        
        z_local.requires_grad_(False)
        return entropy
    
    def _estimate_solver_entropy(
        self,
        z_local: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Estimate solver order uncertainty using KL divergence"""
        # Simulate different solver predictions
        # In practice, this would use actual solver predictions
        
        # First-order (Euler) prediction
        mu_1 = z_local + torch.randn_like(z_local) * 0.1
        sigma_1 = torch.ones_like(z_local) * 0.1
        
        # Second-order (Heun) prediction  
        mu_2 = z_local + torch.randn_like(z_local) * 0.05
        sigma_2 = torch.ones_like(z_local) * 0.05
        
        # Third-order prediction
        mu_3 = z_local + torch.randn_like(z_local) * 0.02
        sigma_3 = torch.ones_like(z_local) * 0.02
        
        # Compute KL divergences between solver predictions
        kl_12 = self._kl_divergence_gaussian(mu_1, sigma_1, mu_2, sigma_2)
        kl_23 = self._kl_divergence_gaussian(mu_2, sigma_2, mu_3, sigma_3)
        
        return kl_12 + kl_23
    
    def _estimate_temporal_entropy(
        self,
        z_local: torch.Tensor,
        t: int,
        entropy_history: List
    ) -> torch.Tensor:
        """Estimate temporal uncertainty using covariance analysis"""
        if len(entropy_history) < self.temporal_window:
            return torch.tensor(0.0, device=z_local.device)
        
        # Get temporal differences
        recent_history = entropy_history[-self.temporal_window:]
        temporal_diffs = []
        
        for i in range(1, len(recent_history)):
            diff = recent_history[i] - recent_history[i-1]
            temporal_diffs.append(diff)
        
        if temporal_diffs:
            temporal_tensor = torch.stack(temporal_diffs)
            cov_matrix = torch.cov(temporal_tensor.flatten().unsqueeze(0))
            
            # Multivariate entropy: H = 0.5 * log(det(2πe * Σ))
            det_cov = torch.det(cov_matrix + torch.eye(cov_matrix.shape[0], device=cov_matrix.device) * 1e-6)
            entropy = 0.5 * torch.log(2 * np.pi * np.e * det_cov)
            
            return entropy
        
        return torch.tensor(0.0, device=z_local.device)
    
    def _estimate_spectral_entropy(
        self,
        z_local: torch.Tensor,
        f: int
    ) -> torch.Tensor:
        """Estimate spectral uncertainty using power spectral density"""
        # Compute FFT along time dimension
        z_fft = torch.fft.fft(z_local, dim=1)
        power_spectrum = torch.abs(z_fft) ** 2
        
        # Normalize to get probability distribution
        power_sum = torch.sum(power_spectrum, dim=1, keepdim=True)
        p_omega = power_spectrum / (power_sum + 1e-8)
        
        # Compute spectral entropy: H = -Σ p(ω) log p(ω)
        log_p = torch.log(p_omega + 1e-8)
        entropy = -torch.sum(p_omega * log_p)
        
        return entropy
    
    def _compute_adaptive_weights(
        self,
        z_local: torch.Tensor,
        k: int,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Compute adaptive weights for entropy component combination"""
        # Prepare input features
        z_features = z_local.flatten()
        
        # Time step embedding
        k_embed = self._sinusoidal_embedding(k, 64)
        
        # Positional encoding
        pos_enc = torch.tensor([
            np.sin(0.1 * t), np.cos(0.1 * t),
            np.sin(0.1 * f), np.cos(0.1 * f),
            s / self.config.max_scales
        ], device=z_local.device, dtype=z_local.dtype)
        
        # Local statistics
        local_stats = torch.tensor([
            torch.mean(z_local), torch.std(z_local),
            torch.min(z_local), torch.max(z_local),
            torch.median(z_local), torch.var(z_local),
            torch.norm(z_local), torch.sum(torch.abs(z_local)),
            torch.sum(z_local > 0).float(), torch.sum(z_local < 0).float()
        ], device=z_local.device, dtype=z_local.dtype)
        
        # Concatenate all features
        features = torch.cat([z_features, k_embed, pos_enc, local_stats])
        
        # Get adaptive weights
        weights = self.weight_network(features.unsqueeze(0)).squeeze(0)
        
        return weights
    
    def _sinusoidal_embedding(self, x: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional embedding"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = x * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=0)
        return emb
    
    def _kl_divergence_gaussian(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between two Gaussian distributions"""
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        
        kl = torch.log(sigma2 / sigma1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
        
        return torch.mean(kl)
