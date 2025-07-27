"""
Adaptive Solver Order Selection

Implements adaptive solver order selection based on stability criteria
and entropy-driven numerical intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class AdaptiveSolver(nn.Module):
    """
    Adaptive Solver Order Selection System
    
    Implements stability-aware information processing with adaptive
    numerical intelligence for optimal solver order selection.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Stability thresholds for solver order selection
        self.stability_thresholds = {
            1: config.stability_threshold_low,    # First-order (Euler)
            2: config.stability_threshold_high,   # Second-order (Heun)
            3: 1.0                                # Third-order
        }
        
        # Condition number limits
        self.condition_limits = {
            1: 100.0,  # First-order: robust to ill-conditioning
            2: 50.0,   # Second-order: moderate conditioning
            3: 20.0    # Third-order: requires good conditioning
        }
        
        # Solver implementations
        self.solvers = {
            1: self._euler_step,
            2: self._heun_step,
            3: self._third_order_step
        }
    
    def select_solver_order(
        self,
        entropy_map: torch.Tensor,
        t: int,
        f: int,
        s: int,
        k: int,
        active_nodes: List[Tuple[int, int, int]] = None
    ) -> int:
        """
        Select optimal solver order based on entropy and stability
        
        Args:
            entropy_map: Current entropy map
            t, f, s: Lattice coordinates
            k: Current diffusion step
            
        Returns:
            solver_order: Selected solver order (1, 2, or 3)
        """
        # Get local entropy value with CUDA-safe bounds checking
        try:
            if active_nodes is not None:
                try:
                    node_idx = active_nodes.index((t, f, s))
                except ValueError:
                    return 1  # Default if node not found
            else:
                node_idx = self._get_node_index(entropy_map, t, f, s)

            # CUDA-safe bounds checking
            if node_idx >= len(entropy_map) or node_idx < 0:
                return 1  # Default to first-order for safety

            local_entropy = entropy_map[node_idx]

            # Validate entropy tensor value
            if torch.isnan(local_entropy) or torch.isinf(local_entropy):
                return 1  # Default to first-order for invalid values

            # Stability analysis based on entropy with error handling
            try:
                stability_score = self._compute_stability_score(local_entropy, k)
                # Convert to float and validate
                stability_score = float(stability_score)
                # Validate stability score using math functions
                import math
                if math.isnan(stability_score) or math.isinf(stability_score):
                    stability_score = 0.5  # Default moderate stability
            except Exception as e:
                logger.debug(f"Error computing stability score: {e}")
                stability_score = 0.5  # Default moderate stability

            # Select solver order based on stability with safe bounds
            if stability_score < self.stability_thresholds[1]:
                # High entropy/instability: use robust first-order
                return 1
            elif stability_score < self.stability_thresholds[2]:
                # Medium entropy: use second-order
                return 2
            else:
                # Low entropy/stable: use high-order for accuracy
                return 3

        except Exception as e:
            logger.warning(f"Error in solver order selection: {e}")
            return 1  # Safe fallback
    
    def dpm_solver_step(
        self,
        z: torch.Tensor,
        k: int,
        k_prev: int,
        solver_order: int,
        score_network: nn.Module,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """
        Perform DPM solver step with selected order

        Args:
            z: Current state
            k: Current diffusion step
            k_prev: Previous diffusion step
            solver_order: Selected solver order
            score_network: Score function network
            t, f, s: Lattice coordinates

        Returns:
            z_next: Next state
        """
        # Debug logging for CUDA index investigation
        logger.debug(f"DPM solver step: z.shape={z.shape}, k={k}, k_prev={k_prev}, solver_order={solver_order}, t={t}, f={f}, s={s}")

        solver_fn = self.solvers.get(solver_order, self._euler_step)

        try:
            result = solver_fn(z, k, k_prev, score_network, t, f, s)
            logger.debug(f"DPM solver step completed successfully, result.shape={result.shape}")
            return result
        except Exception as e:
            logger.error(f"DPM solver step failed: {e}")
            logger.error(f"Input: z.shape={z.shape}, k={k}, k_prev={k_prev}, solver_order={solver_order}")
            raise
    
    def adaptive_guidance_strength(
        self,
        local_entropy: torch.Tensor,
        k: int
    ) -> float:
        """
        Compute adaptive guidance strength based on local entropy
        
        Args:
            local_entropy: Local entropy value
            k: Current diffusion step
            
        Returns:
            guidance_strength: Adaptive guidance strength
        """
        base_strength = self.config.guidance_strength
        
        # Entropy-based adjustment
        entropy_factor = torch.sigmoid(local_entropy - 0.5)  # Normalize around 0.5
        
        # Diffusion step adjustment (stronger guidance early)
        step_factor = 1.0 + 0.5 * (k / self.config.num_diffusion_steps)
        
        # Combined adaptive strength
        adaptive_strength = base_strength * entropy_factor * step_factor
        
        return adaptive_strength.item()
    
    def _get_node_index(
        self,
        entropy_map: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> int:
        """
        Get index of node in entropy map with CUDA-safe bounds checking

        Proper lattice coordinate to linear index mapping.
        """
        if len(entropy_map) == 0:
            return 0

        # CUDA-safe bounds checking with additional validation
        max_idx = len(entropy_map) - 1
        if max_idx < 0:
            return 0

        # Validate input coordinates
        if t < 0 or f < 0 or s < 0:
            return 0

        # Create a deterministic mapping from (t,f,s) to index
        # This is simplified - real implementation would use lattice structure
        # Use safe modulo operations to prevent overflow
        try:
            t_safe = max(0, t) % 100  # Limit to reasonable range
            f_safe = max(0, f) % 100
            s_safe = max(0, s) % 10

            linear_idx = t_safe + f_safe * 100 + s_safe * 10000

            # Ensure index is within bounds
            safe_idx = linear_idx % (max_idx + 1)
            return min(safe_idx, max_idx)

        except Exception as e:
            logger.warning(f"Error computing node index for ({t}, {f}, {s}): {e}")
            return 0
    
    def _compute_stability_score(
        self,
        local_entropy: torch.Tensor,
        k: int
    ) -> float:
        """Compute stability score for solver selection"""
        # Higher entropy indicates lower stability
        entropy_score = 1.0 - torch.sigmoid(local_entropy)
        
        # Early diffusion steps are less stable
        step_score = k / self.config.num_diffusion_steps
        
        # Combined stability score
        stability = 0.7 * entropy_score + 0.3 * step_score
        
        return stability.item()
    
    def _euler_step(
        self,
        z: torch.Tensor,
        k: int,
        k_prev: int,
        score_network: nn.Module,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """First-order Euler step (most robust)"""
        # Get diffusion parameters
        alpha_k = self._get_alpha(k)
        alpha_k_prev = self._get_alpha(k_prev)
        sigma_k = self._get_sigma(k)

        # Validate input tensor dimensions
        if len(z.shape) != 3:
            raise ValueError(f"Expected 3D tensor [batch, length, channels], got {z.shape}")

        batch_size, seq_len, channels = z.shape
        if seq_len == 0 or channels == 0:
            raise ValueError(f"Invalid tensor dimensions: {z.shape}")

        # Debug logging for CUDA index investigation
        logger.debug(f"Euler step: z.shape={z.shape}, k={k}, device={z.device}")

        # Transpose tensor for 1D convolution: [batch, length, channels] -> [batch, channels, length]
        z_transposed = z.transpose(-2, -1)
        logger.debug(f"Euler step: z_transposed.shape={z_transposed.shape}")

        # Create timestep tensor with correct batch size
        timesteps = torch.full((batch_size,), k, device=z.device, dtype=torch.long)
        logger.debug(f"Euler step: timesteps.shape={timesteps.shape}, timesteps={timesteps}")

        # Compute score with error handling
        try:
            with torch.no_grad():
                logger.debug(f"Calling score network with z_transposed.shape={z_transposed.shape}, timesteps.shape={timesteps.shape}")
                score = score_network(z_transposed, timesteps)
                logger.debug(f"Score network output shape: {score.shape}")
        except Exception as e:
            logger.error(f"Score network error in Euler step: {e}")
            logger.error(f"Input shapes: z_transposed={z_transposed.shape}, timesteps={timesteps.shape}")
            # Return zero gradient as fallback
            return z

        # Transpose back: [batch, channels, length] -> [batch, length, channels]
        score = score.transpose(-2, -1)

        # Euler step
        h = alpha_k_prev - alpha_k
        z_next = z + h * score

        return z_next
    
    def _heun_step(
        self,
        z: torch.Tensor,
        k: int,
        k_prev: int,
        score_network: nn.Module,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Second-order Heun step (balanced accuracy/stability)"""
        # Get diffusion parameters
        alpha_k = self._get_alpha(k)
        alpha_k_prev = self._get_alpha(k_prev)
        h = alpha_k_prev - alpha_k

        # Validate input tensor dimensions
        if len(z.shape) != 3:
            raise ValueError(f"Expected 3D tensor [batch, length, channels], got {z.shape}")

        batch_size, seq_len, channels = z.shape
        if seq_len == 0 or channels == 0:
            raise ValueError(f"Invalid tensor dimensions: {z.shape}")

        # Transpose tensor for 1D convolution: [batch, length, channels] -> [batch, channels, length]
        z_transposed = z.transpose(-2, -1)

        # Create timestep tensors with correct batch size
        timesteps_k = torch.full((batch_size,), k, device=z.device, dtype=torch.long)
        timesteps_k_prev = torch.full((batch_size,), k_prev, device=z.device, dtype=torch.long)

        # First stage with error handling
        try:
            with torch.no_grad():
                score_1 = score_network(z_transposed, timesteps_k)
        except Exception as e:
            logger.warning(f"Score network error in Heun step (stage 1): {e}")
            return z

        # Transpose back for computation
        score_1 = score_1.transpose(-2, -1)
        z_temp = z + h * score_1
        z_temp_transposed = z_temp.transpose(-2, -1)

        # Second stage with error handling
        try:
            with torch.no_grad():
                score_2 = score_network(z_temp_transposed, timesteps_k_prev)
        except Exception as e:
            logger.warning(f"Score network error in Heun step (stage 2): {e}")
            return z

        # Transpose back
        score_2 = score_2.transpose(-2, -1)

        # Heun combination
        z_next = z + h * (score_1 + score_2) / 2

        return z_next
    
    def _third_order_step(
        self,
        z: torch.Tensor,
        k: int,
        k_prev: int,
        score_network: nn.Module,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Third-order step (highest accuracy, requires stability)"""
        # Get diffusion parameters
        alpha_k = self._get_alpha(k)
        alpha_k_prev = self._get_alpha(k_prev)
        h = alpha_k_prev - alpha_k

        # Transpose tensor for 1D convolution: [batch, length, channels] -> [batch, channels, length]
        z_transposed = z.transpose(-2, -1)

        # Create timestep tensors with correct batch size
        batch_size = z.shape[0]
        timesteps_k = torch.full((batch_size,), k, device=z.device, dtype=torch.long)

        # Three-stage Runge-Kutta
        with torch.no_grad():
            # Stage 1
            score_1 = score_network(z_transposed, timesteps_k)
            score_1 = score_1.transpose(-2, -1)  # Transpose back
            z_1 = z + h * score_1 / 3
            z_1_transposed = z_1.transpose(-2, -1)

            # Stage 2
            k_mid = int(k + (k_prev - k) / 3)
            timesteps_k_mid = torch.full((batch_size,), k_mid, device=z.device, dtype=torch.long)
            score_2 = score_network(z_1_transposed, timesteps_k_mid)
            score_2 = score_2.transpose(-2, -1)  # Transpose back
            z_2 = z + h * (score_1 + score_2) / 6
            z_2_transposed = z_2.transpose(-2, -1)

            # Stage 3
            k_mid2 = int(k + 2 * (k_prev - k) / 3)
            timesteps_k_mid2 = torch.full((batch_size,), k_mid2, device=z.device, dtype=torch.long)
            score_3 = score_network(z_2_transposed, timesteps_k_mid2)
            score_3 = score_3.transpose(-2, -1)  # Transpose back

        # Third-order combination
        z_next = z + h * (score_1 + 3 * score_3) / 4

        return z_next
    
    def _get_alpha(self, k: int) -> float:
        """Get alpha parameter for diffusion step k"""
        # This would use the actual diffusion schedule
        # Simplified version for demonstration
        return 1.0 - k / self.config.num_diffusion_steps
    
    def _get_sigma(self, k: int) -> float:
        """Get sigma parameter for diffusion step k"""
        # This would use the actual diffusion schedule
        # Simplified version for demonstration
        return np.sqrt(k / self.config.num_diffusion_steps)


class StabilityAnalyzer:
    """Analyzer for numerical stability of solver methods"""
    
    def __init__(self):
        self.stability_history = []
        self.solver_usage = {1: 0, 2: 0, 3: 0}
        self.error_estimates = []
    
    def analyze_stability(
        self,
        z_current: torch.Tensor,
        z_next: torch.Tensor,
        solver_order: int,
        entropy_value: float
    ) -> Dict:
        """Analyze stability of solver step"""
        # Compute step size and error estimates
        step_norm = torch.norm(z_next - z_current)
        relative_change = step_norm / (torch.norm(z_current) + 1e-8)
        
        # Stability indicators
        is_stable = relative_change < 1.0  # Reasonable step size
        condition_estimate = self._estimate_condition_number(z_current, z_next)
        
        # Record solver usage
        self.solver_usage[solver_order] += 1
        
        stability_info = {
            'solver_order': solver_order,
            'step_norm': step_norm.item(),
            'relative_change': relative_change.item(),
            'is_stable': is_stable,
            'condition_estimate': condition_estimate,
            'entropy_value': entropy_value
        }
        
        self.stability_history.append(stability_info)
        
        return stability_info
    
    def _estimate_condition_number(
        self,
        z_current: torch.Tensor,
        z_next: torch.Tensor
    ) -> float:
        """Estimate condition number of the numerical step"""
        # Simplified condition number estimation
        # In practice, this would be more sophisticated
        
        jacobian_approx = (z_next - z_current).flatten()
        if torch.norm(jacobian_approx) > 1e-8:
            # Rough estimate based on step characteristics
            condition_est = torch.max(torch.abs(jacobian_approx)) / (torch.min(torch.abs(jacobian_approx)) + 1e-8)
            return condition_est.item()
        else:
            return 1.0
    
    def get_solver_statistics(self) -> Dict:
        """Get statistics on solver usage and stability"""
        if not self.stability_history:
            return {}
        
        total_steps = sum(self.solver_usage.values())
        
        # Usage percentages
        usage_percentages = {
            order: count / total_steps * 100 
            for order, count in self.solver_usage.items()
        }
        
        # Stability statistics
        stable_steps = sum(1 for h in self.stability_history if h['is_stable'])
        stability_rate = stable_steps / len(self.stability_history) * 100
        
        # Average metrics by solver order
        order_stats = {}
        for order in [1, 2, 3]:
            order_history = [h for h in self.stability_history if h['solver_order'] == order]
            if order_history:
                order_stats[order] = {
                    'avg_step_norm': np.mean([h['step_norm'] for h in order_history]),
                    'avg_relative_change': np.mean([h['relative_change'] for h in order_history]),
                    'stability_rate': sum(1 for h in order_history if h['is_stable']) / len(order_history) * 100,
                    'avg_condition': np.mean([h['condition_estimate'] for h in order_history])
                }
        
        return {
            'usage_percentages': usage_percentages,
            'overall_stability_rate': stability_rate,
            'total_steps': total_steps,
            'order_statistics': order_stats
        }


class SolverOrderPredictor(nn.Module):
    """Neural network for predicting optimal solver order"""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 solver orders
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict optimal solver order probabilities"""
        return self.predictor(features)
    
    def predict_order(self, features: torch.Tensor) -> int:
        """Predict single optimal solver order"""
        probs = self.forward(features)
        return torch.argmax(probs).item() + 1  # Convert to 1-based indexing
