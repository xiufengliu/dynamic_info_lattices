"""
Core implementation of the Dynamic Information Lattices algorithm.

This module implements Algorithm S1 from the supplementary material:
Dynamic Information Lattices - Complete Algorithm
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

from .multi_component_entropy import MultiComponentEntropy
from .hierarchical_lattice import HierarchicalLattice
from .information_aware_sampler import InformationAwareSampler
from .adaptive_solver import AdaptiveSolver
from ..models.score_network import ScoreNetwork

logger = logging.getLogger(__name__)


@dataclass
class DILConfig:
    """Configuration for Dynamic Information Lattices"""
    # Diffusion parameters
    num_diffusion_steps: int = 1000
    inference_steps: int = 20
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    # Lattice parameters
    max_scales: int = 4
    base_resolution: Tuple[int, int] = (64, 64)  # (time, frequency)
    refinement_threshold: float = 0.3
    coarsening_threshold: float = 0.1
    
    # Entropy parameters
    entropy_budget: float = 0.2  # 20% of lattice nodes
    temperature: float = 5.0
    adaptive_temperature: bool = True
    
    # Solver parameters
    max_solver_order: int = 3
    stability_threshold_low: float = 0.1
    stability_threshold_high: float = 0.5
    
    # Training parameters
    guidance_strength: float = 2.0
    adaptive_guidance: bool = True
    
    # Device and precision
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


class DynamicInfoLattices(nn.Module):
    """
    Dynamic Information Lattices: Main framework implementation
    
    Implements Algorithm S1: Dynamic Information Lattices - Complete Algorithm
    """
    
    def __init__(
        self,
        config: DILConfig,
        score_network: ScoreNetwork,
        data_shape: Tuple[int, ...],
    ):
        super().__init__()
        self.config = config
        self.data_shape = data_shape
        
        # Core components
        self.score_network = score_network
        self.entropy_estimator = MultiComponentEntropy(config, data_shape)
        self.lattice = HierarchicalLattice(config, data_shape)
        self.sampler = InformationAwareSampler(config)
        self.solver = AdaptiveSolver(config)
        
        # Diffusion schedule
        self.register_buffer("betas", self._get_beta_schedule())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        
        # History tracking for temporal entropy
        self.entropy_history = []
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule for diffusion process"""
        if self.config.beta_schedule == "linear":
            return torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.num_diffusion_steps,
                dtype=self.config.dtype
            )
        elif self.config.beta_schedule == "cosine":
            steps = self.config.num_diffusion_steps
            s = 0.008
            x = torch.linspace(0, steps, steps + 1, dtype=self.config.dtype)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
    
    def forward(
        self,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass implementing Algorithm S1: Dynamic Information Lattices
        
        Args:
            y_obs: Observed time series data [batch_size, length, channels]
            mask: Observation mask [batch_size, length, channels]
            return_trajectory: Whether to return full sampling trajectory
            
        Returns:
            Generated time series or (generated, trajectory) if return_trajectory=True
        """
        batch_size = y_obs.shape[0]
        device = y_obs.device
        
        # Phase 1: Initialize Hierarchical Lattice
        lattice_k = self.lattice.construct_hierarchical_lattice(y_obs)
        
        # Sample initial noise
        z_k = torch.randn(
            batch_size, *self.data_shape,
            device=device, dtype=self.config.dtype
        )
        
        # Initialize entropy history
        self.entropy_history = []
        trajectory = [] if return_trajectory else None
        
        # Inference loop
        inference_steps = self.config.inference_steps
        step_size = self.config.num_diffusion_steps // inference_steps
        
        for i, k in enumerate(range(self.config.num_diffusion_steps - 1, -1, -step_size)):
            k_prev = max(k - step_size, 0)
            
            # Phase 2: Multi-Component Entropy Estimation
            entropy_map = self._estimate_entropy_map(z_k, k, lattice_k, y_obs)
            
            # Phase 3: Dynamic Lattice Adaptation
            lattice_k_prev = self.lattice.adapt_lattice(
                lattice_k, entropy_map, k
            )
            
            # Phase 4: Information-Aware Sampling
            selected_nodes = self.sampler.stratified_sample(
                lattice_k, entropy_map, self.config.entropy_budget
            )
            
            # Phase 5: Multi-Scale Updates with Adaptive Solvers
            z_k_prev = self._multi_scale_update(
                z_k, k, k_prev, selected_nodes, y_obs, mask, entropy_map
            )
            
            # Phase 6: Cross-Scale Synchronization
            z_k_prev = self._synchronize_scales(z_k_prev, lattice_k_prev)
            
            # Update for next iteration
            z_k = z_k_prev
            lattice_k = lattice_k_prev
            
            if return_trajectory:
                trajectory.append(z_k.clone())
                
            logger.debug(f"Inference step {i+1}/{inference_steps}, k={k}")
        
        # Decode final result
        y_hat = self._decode_from_lattice(z_k, lattice_k)
        
        if return_trajectory:
            return y_hat, trajectory
        return y_hat
    
    def _estimate_entropy_map(
        self,
        z: torch.Tensor,
        k: int,
        lattice: Dict,
        y_obs: torch.Tensor
    ) -> torch.Tensor:
        """Estimate entropy map across all lattice nodes"""
        return self.entropy_estimator(z, k, lattice, y_obs, self.entropy_history)
    
    def _multi_scale_update(
        self,
        z_k: torch.Tensor,
        k: int,
        k_prev: int,
        selected_nodes: List[Tuple[int, int, int]],
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        entropy_map: torch.Tensor
    ) -> torch.Tensor:
        """Perform multi-scale updates with adaptive solvers"""
        z_k_prev = z_k.clone()
        
        for t, f, s in selected_nodes:
            # Select solver order based on entropy and stability
            solver_order = self.solver.select_solver_order(
                entropy_map, t, f, s, k
            )
            
            # Apply DPM solver step
            z_local = self.solver.dpm_solver_step(
                z_k, k, k_prev, solver_order, self.score_network,
                t, f, s
            )
            
            # Apply adaptive guidance
            if self.config.adaptive_guidance:
                guidance_strength = self.solver.adaptive_guidance_strength(
                    entropy_map[t, f, s], k
                )
                guidance = self._compute_guidance(z_k, y_obs, mask, t, f, s)
                z_local = z_local + guidance_strength * guidance
            
            # Update local region
            z_k_prev = self._update_local_region(z_k_prev, z_local, t, f, s)
        
        return z_k_prev
    
    def _compute_guidance(
        self,
        z: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Compute guidance signal for self-guidance"""
        # Implement guidance computation based on observation likelihood
        # This is a simplified version - full implementation would include
        # proper gradient computation through the observation model
        guidance = torch.zeros_like(z)
        
        # Simple guidance based on observation error
        if mask is not None:
            obs_error = (z - y_obs) * mask
            guidance = -obs_error / (torch.var(obs_error) + 1e-8)
        
        return guidance
    
    def _update_local_region(
        self,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Update local region in global tensor"""
        # This is a simplified version - full implementation would handle
        # proper multi-scale updates based on lattice structure
        z_updated = z_global.clone()
        
        # Simple local update (would be more sophisticated in practice)
        scale_factor = 2 ** s
        t_start = t * scale_factor
        t_end = min((t + 1) * scale_factor, z_global.shape[1])
        f_start = f * scale_factor
        f_end = min((f + 1) * scale_factor, z_global.shape[2])
        
        if t_end > t_start and f_end > f_start:
            z_updated[:, t_start:t_end, f_start:f_end] = z_local[:, t_start:t_end, f_start:f_end]
        
        return z_updated
    
    def _synchronize_scales(
        self,
        z: torch.Tensor,
        lattice: Dict
    ) -> torch.Tensor:
        """Synchronize across scales to maintain consistency"""
        # Implement cross-scale synchronization
        # This ensures consistency across different resolution levels
        return self.lattice.synchronize_scales(z, lattice)
    
    def _decode_from_lattice(
        self,
        z: torch.Tensor,
        lattice: Dict
    ) -> torch.Tensor:
        """Decode final result from lattice representation"""
        # Simple decoding - in practice this might involve more sophisticated
        # reconstruction from the hierarchical lattice structure
        return z
