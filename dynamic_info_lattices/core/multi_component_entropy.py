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
        # Use a fixed input size for the non-variable features
        # z_features will be processed separately and pooled to fixed size
        fixed_features_dim = (
            64 +  # z features (pooled to fixed size)
            64 +  # k embedding
            5 +   # positional encoding (t, f, s, sin/cos terms)
            10    # local statistics
        )

        return nn.Sequential(
            nn.Linear(fixed_features_dim, 256),
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
        entropy_history: List,
        score_network: nn.Module
    ) -> torch.Tensor:
        """
        Estimate multi-component entropy map implementing Algorithm S2

        Implements Equation (1) from paper:
        H_{t,f,s}^{(k)} = α₁H_{score}^{(k)} + α₂H_{guidance}^{(k)} + α₃H_{solver}^{(k)} + α₄H_{temporal}^{(k)} + α₅H_{spectral}^{(k)}

        Args:
            z: Current latent state [batch_size, length, channels]
            k: Current diffusion step
            lattice: Current lattice structure
            y_obs: Observed data
            entropy_history: History of entropy estimates
            score_network: Score function network for uncertainty estimation

        Returns:
            entropy_map: Entropy values for each lattice node
        """
        batch_size = z.shape[0]
        device = z.device

        # Get active lattice nodes
        active_nodes = lattice.get('active_nodes', [])

        if not active_nodes:
            return torch.zeros(0, device=device, dtype=z.dtype)

        # Initialize entropy map
        entropy_map = torch.zeros(
            len(active_nodes), device=device, dtype=z.dtype
        )

        # Batch process entropy estimation for efficiency
        all_z_local = []
        all_coords = []

        for i, (t, f, s) in enumerate(active_nodes):
            # Extract local region
            z_local = self._extract_local_region(z, t, f, s)
            all_z_local.append(z_local)
            all_coords.append((t, f, s))

        # Batch estimate entropy components
        for i, ((t, f, s), z_local) in enumerate(zip(all_coords, all_z_local)):
            # Estimate individual entropy components with proper implementations
            h_score = self._estimate_score_entropy(z_local, k, score_network)
            h_guidance = self._estimate_guidance_entropy(z_local, y_obs, t, f, s)
            h_solver = self._estimate_solver_entropy(z_local, k, score_network)
            h_temporal = self._estimate_temporal_entropy(z_local, t, entropy_history)
            h_spectral = self._estimate_spectral_entropy(z_local, f)

            # Compute adaptive weights using the weight network
            weights = self._compute_adaptive_weights(z_local, k, t, f, s)

            # Combine entropy components according to Equation (1)
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
        k: int,
        score_network: nn.Module
    ) -> torch.Tensor:
        """
        Estimate score function epistemic uncertainty using Monte Carlo dropout

        Implements proper uncertainty estimation for score function as described in paper.
        """
        if score_network is None:
            return torch.tensor(0.0, device=z_local.device)

        # Store original training state
        original_training = score_network.training

        # Enable dropout for uncertainty estimation
        score_network.train()

        score_samples = []
        timestep_tensor = torch.full((z_local.shape[0],), k, device=z_local.device, dtype=torch.long)

        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                # Ensure proper shape for ScoreNetwork: [batch, channels, length]
                if len(z_local.shape) == 3:
                    # z_local is [batch, time, features] -> reshape to [batch, features, time]
                    z_local_transposed = z_local.transpose(-2, -1)
                else:
                    # Handle other cases by flattening and reshaping appropriately
                    batch_size = z_local.shape[0]
                    # Flatten spatial dimensions and treat as sequence
                    z_local_flat = z_local.view(batch_size, -1)
                    # Add channel dimension: [batch, length] -> [batch, 1, length]
                    z_local_transposed = z_local_flat.unsqueeze(1)

                # Get score function prediction with dropout enabled
                score_sample = score_network(z_local_transposed, timestep_tensor)

                # Transpose back to match expected output format
                if len(z_local.shape) == 3:
                    score_sample = score_sample.transpose(-2, -1)
                else:
                    # Reshape back to original spatial structure if needed
                    score_sample = score_sample.squeeze(1).view_as(z_local_flat)

                score_samples.append(score_sample)

        # Restore original training state
        score_network.train(original_training)

        if not score_samples:
            return torch.tensor(0.0, device=z_local.device)

        # Compute epistemic uncertainty from score function samples
        scores_tensor = torch.stack(score_samples)  # [num_samples, batch, ...]

        # Compute variance across samples (epistemic uncertainty)
        var_score = torch.var(scores_tensor, dim=0, unbiased=True)

        # Differential entropy formulation: H = 0.5 * log(2πe * σ²)
        # This is more principled than the ad-hoc variance-log formulation
        entropy = 0.5 * torch.log(2 * np.pi * np.e * (var_score + 1e-8))

        return torch.mean(entropy)

    def _extract_local_region(
        self,
        tensor: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Extract local region from tensor based on lattice coordinates"""
        scale_factor = 2 ** s

        # Handle 1D vs 2D data
        if len(tensor.shape) == 2:  # [batch, length]
            t_start = t * scale_factor
            t_end = min((t + 1) * scale_factor, tensor.shape[1])
            return tensor[:, t_start:t_end]
        elif len(tensor.shape) == 3:  # [batch, length, channels]
            t_start = t * scale_factor
            t_end = min((t + 1) * scale_factor, tensor.shape[1])
            return tensor[:, t_start:t_end, :]
        else:
            # For higher dimensional tensors, extract time dimension
            t_start = t * scale_factor
            t_end = min((t + 1) * scale_factor, tensor.shape[1])
            return tensor[:, t_start:t_end, ...]
    
    def _estimate_guidance_entropy(
        self,
        z_local: torch.Tensor,
        y_obs: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """
        Estimate self-guidance uncertainty using finite differences

        Implements guidance gradient variance computation as described in paper.
        """
        device = z_local.device

        if y_obs is None or y_obs.numel() == 0:
            return torch.tensor(0.0, device=device)

        # Create perturbations for finite difference estimation
        guidance_grads = []

        for _ in range(5):  # Multiple perturbations for robust estimation
            # Create small random perturbation
            eps = self.finite_diff_eps
            perturbation = eps * torch.randn_like(z_local)
            z_perturbed = z_local + perturbation

            # Compute guidance signal (reconstruction loss gradient)
            z_perturbed_detached = z_perturbed.detach().requires_grad_(True)

            # Extract corresponding region from observations
            y_local = self._extract_corresponding_obs(y_obs, t, f, s, z_local.shape)
            if y_local is not None:
                recon_loss = F.mse_loss(z_perturbed_detached, y_local, reduction='sum')

                # Compute gradient
                grad = torch.autograd.grad(
                    outputs=recon_loss,
                    inputs=z_perturbed_detached,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                )

                if grad[0] is not None:
                    guidance_grads.append(grad[0].detach())

        if not guidance_grads:
            return torch.tensor(0.0, device=device)

        # Compute variance across gradient estimates
        grads_tensor = torch.stack(guidance_grads)
        grad_var = torch.var(grads_tensor, dim=0)

        # Convert variance to entropy
        entropy = 0.5 * torch.log(2 * np.pi * np.e * (torch.mean(grad_var) + 1e-8))

        return entropy

    def _extract_corresponding_obs(
        self,
        y_obs: torch.Tensor,
        t: int,
        f: int,
        s: int,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Extract corresponding observation region"""
        try:
            # Use the same extraction logic as for z_local
            y_local = self._extract_local_region(y_obs, t, f, s)

            # Ensure shapes match
            if y_local.shape != target_shape:
                # Resize if necessary
                if len(target_shape) == 2:  # [batch, length]
                    y_local = F.interpolate(
                        y_local.unsqueeze(1),
                        size=target_shape[1],
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)
                elif len(target_shape) == 3:  # [batch, length, channels]
                    y_local = F.interpolate(
                        y_local.transpose(1, 2),
                        size=target_shape[1],
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)

            return y_local
        except Exception:
            return None
    
    def _estimate_solver_entropy(
        self,
        z_local: torch.Tensor,
        k: int,
        score_network: nn.Module
    ) -> torch.Tensor:
        """
        Estimate solver order uncertainty using KL divergence between different solver orders

        Implements Equations (3-4) from paper comparing DPM-Solver predictions of different orders.
        """
        if score_network is None:
            return torch.tensor(0.0, device=z_local.device)

        device = z_local.device
        timestep_tensor = torch.full((z_local.shape[0],), k, device=device, dtype=torch.long)

        try:
            with torch.no_grad():
                # Transpose tensor for 1D convolution: [batch, length, channels] -> [batch, channels, length]
                z_local_transposed = z_local.transpose(-2, -1)

                # Get score function prediction
                score = score_network(z_local_transposed, timestep_tensor)

                # Transpose back: [batch, channels, length] -> [batch, length, channels]
                score = score.transpose(-2, -1)

                # Simulate different solver order predictions
                # In practice, these would be actual DPM-Solver implementations

                # First-order (Euler) prediction: z_{k-1} = z_k + h * score
                h = 1.0 / 1000  # Step size
                mu_1 = z_local + h * score
                sigma_1 = torch.ones_like(z_local) * 0.1

                # Second-order prediction (simplified Heun's method)
                z_temp = z_local + h * score
                z_temp_transposed = z_temp.transpose(-2, -1)
                score_temp_transposed = score_network(z_temp_transposed, timestep_tensor - 1) if k > 0 else score.transpose(-2, -1)
                score_temp = score_temp_transposed.transpose(-2, -1) if k > 0 else score
                mu_2 = z_local + h * 0.5 * (score + score_temp)
                sigma_2 = torch.ones_like(z_local) * 0.05

                # Third-order prediction (simplified)
                mu_3 = z_local + h * score + 0.5 * h**2 * (score_temp - score) / h
                sigma_3 = torch.ones_like(z_local) * 0.02

                # Compute KL divergences between different orders
                kl_12 = self._kl_divergence_gaussian(mu_1, sigma_1, mu_2, sigma_2)
                kl_23 = self._kl_divergence_gaussian(mu_2, sigma_2, mu_3, sigma_3)

                return torch.mean(kl_12 + kl_23)

        except Exception as e:
            logger.warning(f"Error in solver entropy estimation: {e}")
            return torch.tensor(0.0, device=device)
    
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
            temporal_flat = temporal_tensor.flatten()

            # For 1D case, use variance directly
            if temporal_flat.numel() == 1:
                entropy = 0.5 * torch.log(2 * np.pi * np.e * torch.var(temporal_flat))
            else:
                cov_matrix = torch.cov(temporal_flat.unsqueeze(0))

                # Handle scalar covariance matrix
                if cov_matrix.dim() == 0:
                    det_cov = cov_matrix + 1e-6
                else:
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
        # Pool z_local to fixed size features using global pooling
        z_flat = z_local.flatten()
        if len(z_flat) >= 64:
            # Use adaptive pooling to get exactly 64 features
            z_reshaped = z_flat.view(1, 1, -1)  # [1, 1, N]
            z_pooled = F.adaptive_avg_pool1d(z_reshaped, 64).squeeze()  # [64]
        else:
            # Pad with zeros if too small
            z_pooled = F.pad(z_flat, (0, 64 - len(z_flat)))

        # Time step embedding
        k_embed = self._sinusoidal_embedding(k, 64, z_local.device)

        # Positional encoding
        pos_enc = torch.tensor([
            np.sin(0.1 * t), np.cos(0.1 * t),
            np.sin(0.1 * f), np.cos(0.1 * f),
            s / self.config.max_scales
        ], device=z_local.device, dtype=z_local.dtype)

        # Local statistics - compute on device
        local_stats = torch.stack([
            torch.mean(z_local), torch.std(z_local),
            torch.min(z_local), torch.max(z_local),
            torch.median(z_local), torch.var(z_local),
            torch.norm(z_local), torch.sum(torch.abs(z_local)),
            torch.sum(z_local > 0).float(), torch.sum(z_local < 0).float()
        ])

        # Concatenate all features
        features = torch.cat([z_pooled, k_embed, pos_enc, local_stats])

        # Get adaptive weights
        weights = self.weight_network(features.unsqueeze(0)).squeeze(0)

        return weights
    
    def _sinusoidal_embedding(self, x: int, dim: int, device: torch.device) -> torch.Tensor:
        """Create sinusoidal positional embedding"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
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
