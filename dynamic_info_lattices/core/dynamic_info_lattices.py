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

        Implements the complete 5-phase algorithm as specified in the paper:
        1. Multi-component entropy estimation across all lattice nodes
        2. Dynamic lattice adaptation based on entropy patterns
        3. Information-aware sampling using entropy-weighted probabilities
        4. Multi-scale updates with adaptive solvers
        5. Cross-scale synchronization

        Args:
            y_obs: Observed time series data [batch_size, length, channels]
            mask: Observation mask [batch_size, length, channels]
            return_trajectory: Whether to return full sampling trajectory

        Returns:
            Generated time series or (generated, trajectory) if return_trajectory=True
        """
        batch_size = y_obs.shape[0]
        device = y_obs.device

        # Validate input dimensions
        if len(y_obs.shape) != 3:
            raise ValueError(f"Expected 3D input [batch, length, channels], got {y_obs.shape}")

        # Initialize hierarchical lattice (Algorithm S3)
        lattice_k = self.lattice.construct_hierarchical_lattice(y_obs)

        # Sample initial noise from standard Gaussian
        z_k = torch.randn(
            batch_size, *self.data_shape,
            device=device, dtype=self.config.dtype
        )

        # Initialize entropy history for temporal entropy computation
        self.entropy_history = []
        trajectory = [] if return_trajectory else None

        # Compute inference schedule
        inference_steps = self.config.inference_steps
        timesteps = self._get_inference_schedule(inference_steps)

        # Main inference loop implementing Algorithm S1
        for i, (k, k_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):

            # Phase 1: Multi-Component Entropy Estimation (Algorithm S2)
            entropy_map = self._estimate_entropy_map(z_k, k, lattice_k, y_obs)

            # Store entropy for temporal analysis
            self.entropy_history.append(entropy_map.clone())

            # Phase 2: Dynamic Lattice Adaptation (Algorithm S4)
            lattice_k_prev = self.lattice.adapt_lattice(
                lattice_k, entropy_map, k, self._compute_entropy_gradients(entropy_map, lattice_k)
            )

            # Phase 3: Information-Aware Sampling (Algorithm S5)
            selected_nodes = self.sampler.stratified_sample(
                lattice_k_prev, entropy_map, self.config.entropy_budget
            )

            # Phase 4: Multi-Scale Updates with Adaptive Solvers
            z_k_prev = self._multi_scale_update(
                z_k, k, k_prev, selected_nodes, y_obs, mask, entropy_map, lattice_k_prev
            )

            # Phase 5: Cross-Scale Synchronization (disabled for debugging)
            if self.config.cross_scale_sync:
                z_k_prev = self._synchronize_scales(z_k_prev, lattice_k_prev)

            # Update state for next iteration
            z_k = z_k_prev
            lattice_k = lattice_k_prev

            # Store trajectory if requested
            if return_trajectory:
                trajectory.append(z_k.clone().detach())

            # Check convergence criteria
            if self._check_convergence(z_k, k, entropy_map):
                logger.info(f"Converged early at step {i+1}/{inference_steps}")
                break

            logger.debug(f"Inference step {i+1}/{inference_steps}, k={k}, active_nodes={len(lattice_k['active_nodes'])}")

        # Decode final result from lattice representation
        y_hat = self._decode_from_lattice(z_k, lattice_k)

        if return_trajectory:
            return y_hat, trajectory
        return y_hat
    
    def _get_inference_schedule(self, inference_steps: int) -> List[int]:
        """Get inference timestep schedule"""
        if inference_steps >= self.config.num_diffusion_steps:
            return list(range(self.config.num_diffusion_steps - 1, -1, -1))

        # Use uniform spacing for simplicity (could implement DDIM scheduling)
        step_size = self.config.num_diffusion_steps // inference_steps
        timesteps = list(range(self.config.num_diffusion_steps - 1, -1, -step_size))
        timesteps.append(0)  # Ensure we end at 0
        return timesteps

    def _estimate_entropy_map(
        self,
        z: torch.Tensor,
        k: int,
        lattice: Dict,
        y_obs: torch.Tensor
    ) -> torch.Tensor:
        """Estimate entropy map across all lattice nodes using Algorithm S2"""
        return self.entropy_estimator(z, k, lattice, y_obs, self.entropy_history, self.score_network)

    def _compute_entropy_gradients(
        self,
        entropy_map: torch.Tensor,
        lattice: Dict
    ) -> torch.Tensor:
        """Compute spatial gradients of entropy for refinement decisions"""
        active_nodes = lattice['active_nodes']
        gradients = torch.zeros_like(entropy_map)

        for i, (t, f, s) in enumerate(active_nodes):
            if i >= len(entropy_map):
                continue

            # Find neighboring nodes at the same scale
            neighbors = self._find_spatial_neighbors(t, f, s, active_nodes)

            if neighbors:
                neighbor_entropies = []
                for nt, nf, ns in neighbors:
                    if (nt, nf, ns) in active_nodes:
                        neighbor_idx = active_nodes.index((nt, nf, ns))
                        if neighbor_idx < len(entropy_map):
                            neighbor_entropies.append(entropy_map[neighbor_idx])

                if neighbor_entropies:
                    neighbor_tensor = torch.stack(neighbor_entropies)
                    current_entropy = entropy_map[i]

                    # Compute gradient magnitude using finite differences
                    gradient_mag = torch.std(neighbor_tensor - current_entropy)
                    gradients[i] = gradient_mag

        return gradients

    def _find_spatial_neighbors(
        self,
        t: int,
        f: int,
        s: int,
        active_nodes: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Find spatial neighbors at the same scale"""
        neighbors = []

        # Check 8-connected neighbors (or 2-connected for 1D)
        if len(self.data_shape) == 1:  # 1D time series
            for dt in [-1, 1]:
                neighbor = (t + dt, 0, s)  # f=0 for 1D
                if neighbor in active_nodes:
                    neighbors.append(neighbor)
        else:  # 2D case
            for dt in [-1, 0, 1]:
                for df in [-1, 0, 1]:
                    if dt == 0 and df == 0:
                        continue
                    neighbor = (t + dt, f + df, s)
                    if neighbor in active_nodes:
                        neighbors.append(neighbor)

        return neighbors

    def _check_convergence(
        self,
        z: torch.Tensor,
        k: int,
        entropy_map: torch.Tensor
    ) -> bool:
        """Check convergence criteria"""
        # Simple convergence check based on entropy
        if len(self.entropy_history) < 2:
            return False

        # Check if entropy change is below threshold
        current_entropy = torch.mean(entropy_map)
        prev_entropy = torch.mean(self.entropy_history[-2])
        entropy_change = torch.abs(current_entropy - prev_entropy)

        convergence_threshold = 1e-4
        return entropy_change < convergence_threshold
    
    def _multi_scale_update(
        self,
        z_k: torch.Tensor,
        k: int,
        k_prev: int,
        selected_nodes: List[Tuple[int, int, int]],
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        entropy_map: torch.Tensor,
        lattice: Dict
    ) -> torch.Tensor:
        """Perform multi-scale updates with adaptive solvers"""
        z_k_prev = z_k.clone()
        active_nodes = lattice['active_nodes']

        # Process selected nodes in order of decreasing entropy (most uncertain first)
        node_entropies = []
        for t, f, s in selected_nodes:
            if (t, f, s) in active_nodes:
                node_idx = active_nodes.index((t, f, s))
                if node_idx < len(entropy_map):
                    node_entropies.append((entropy_map[node_idx], t, f, s))

        # Sort by entropy (descending)
        node_entropies.sort(key=lambda x: x[0], reverse=True)

        for entropy_val, t, f, s in node_entropies:
            # Debug logging for lattice node processing
            logger.debug(f"Processing node: entropy={entropy_val:.6f}, t={t}, f={f}, s={s}")

            # Extract local region
            z_local = self._extract_local_region(z_k, t, f, s)
            logger.debug(f"Extracted local region shape: {z_local.shape}")

            # Select solver order based on entropy and stability
            solver_order = self.solver.select_solver_order(
                entropy_map, t, f, s, k, active_nodes
            )
            logger.debug(f"Selected solver order: {solver_order}")

            # Apply DPM solver step
            z_local_updated = self.solver.dpm_solver_step(
                z_local, k, k_prev, solver_order, self.score_network,
                t, f, s
            )
            logger.debug(f"Updated local region shape: {z_local_updated.shape}")

            # Apply adaptive guidance if enabled
            if self.config.adaptive_guidance:
                guidance_strength = self.solver.adaptive_guidance_strength(
                    entropy_val, k
                )
                guidance = self._compute_guidance(z_local, y_obs, mask, t, f, s)
                z_local_updated = z_local_updated + guidance_strength * guidance

            # Update local region in global tensor
            z_k_prev = self._update_local_region(z_k_prev, z_local_updated, t, f, s)

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
        """Compute guidance signal for self-guidance

        FIXED: Extract corresponding local regions from y_obs and mask
        """
        guidance = torch.zeros_like(z)

        # Extract corresponding local regions from observations and mask
        y_obs_local = self._extract_local_region(y_obs, t, f, s)
        mask_local = self._extract_local_region(mask, t, f, s)

        # Simple guidance based on observation error
        if mask_local is not None:
            obs_error = (z - y_obs_local) * mask_local
            guidance = -obs_error / (torch.var(obs_error) + 1e-8)

        return guidance
    
    def _extract_local_region(
        self,
        z: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Extract local region from global tensor based on lattice coordinates

        FIXED: Use consistent coordinate system with proper bounds checking and CUDA safety
        """
        scale_factor = 2 ** s
        seq_len = z.shape[1]

        # Debug logging for CUDA index investigation
        logger.debug(f"_extract_local_region: z.shape={z.shape}, t={t}, f={f}, s={s}, scale_factor={scale_factor}")

        # CUDA-safe bounds checking with additional safety margins
        t_orig = t
        t = max(0, min(t, seq_len - 1))

        # Extract temporal region based on scale with conservative bounds checking
        t_start = t
        t_end = min(t + scale_factor, seq_len)

        # Ensure we have at least one time step but don't exceed bounds
        if t_end <= t_start:
            t_end = min(t_start + 1, seq_len)

        # Additional safety: ensure indices are well within bounds
        t_start = max(0, min(t_start, seq_len - 1))
        t_end = max(t_start + 1, min(t_end, seq_len))

        # Additional debug logging
        logger.debug(f"_extract_local_region: t_orig={t_orig}, t_adjusted={t}, t_start={t_start}, t_end={t_end}, seq_len={seq_len}")

        # Validate indices before slicing with strict checks
        if t_start < 0 or t_end > seq_len or t_start >= t_end or t_start >= seq_len or t_end <= 0:
            logger.error(f"Invalid slice indices: t_start={t_start}, t_end={t_end}, seq_len={seq_len}")
            # Return a safe fallback instead of crashing
            if len(z.shape) == 3:
                return z[:, :1, :].clone()  # Return first time step as fallback
            else:
                return z[:, :1].clone()

        # CUDA-safe tensor slicing with explicit bounds
        try:
            if len(z.shape) == 3:  # [batch, length, channels]
                # Use explicit indexing to avoid CUDA index issues
                result = z[:, t_start:t_end, :].clone()
                logger.debug(f"_extract_local_region result shape: {result.shape}")
                return result
            else:  # [batch, length]
                result = z[:, t_start:t_end].clone()
                logger.debug(f"_extract_local_region result shape: {result.shape}")
                return result
        except Exception as e:
            logger.error(f"CUDA indexing error in _extract_local_region: {e}")
            logger.error(f"Tensor shape: {z.shape}, indices: [{t_start}:{t_end}]")
            # Return safe fallback
            if len(z.shape) == 3:
                return z[:, :1, :].clone()
            else:
                return z[:, :1].clone()

    def _update_local_region(
        self,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Update local region in global tensor"""
        z_updated = z_global.clone()
        scale_factor = 2 ** s
        seq_len = z_global.shape[1]

        # Ensure t coordinate is within bounds
        t = max(0, min(t, seq_len - 1))

        # Extract temporal region based on scale with proper bounds checking
        t_start = t
        t_end = min(t + scale_factor, seq_len)

        # Ensure we have at least one time step
        if t_end <= t_start:
            t_end = min(t_start + 1, seq_len)

        # Handle tensor updates with proper bounds checking
        # Always use the bounds-checked t_start and t_end from above
        if len(z_global.shape) == 3:  # [batch, length, channels]
            # Ensure z_local has enough data to fill the region
            available_length = z_local.shape[1]
            required_length = t_end - t_start

            if available_length >= required_length and required_length > 0:
                # Safe assignment with proper bounds
                z_updated[:, t_start:t_end, :] = z_local[:, :required_length, :]
        elif len(z_global.shape) == 2:  # [batch, length]
            # Ensure z_local has enough data to fill the region
            available_length = z_local.shape[1]
            required_length = t_end - t_start

            if available_length >= required_length and required_length > 0:
                # Safe assignment with proper bounds
                z_updated[:, t_start:t_end] = z_local[:, :required_length]

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
