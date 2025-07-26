"""
Information-Aware Sampling Strategy

Implements Algorithm S5: Stratified Information-Aware Sampling
from the supplementary material.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class InformationAwareSampler(nn.Module):
    """
    Information-Aware Sampling Strategy
    
    Implements Algorithm S5: Stratified Information-Aware Sampling
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_strata = 10
        
    def stratified_sample(
        self,
        lattice: Dict,
        entropy_map: torch.Tensor,
        budget_fraction: float
    ) -> List[Tuple[int, int, int]]:
        """
        Perform stratified information-aware sampling
        
        Implements Algorithm S5: Stratified Information-Aware Sampling
        
        Args:
            lattice: Current lattice structure
            entropy_map: Entropy values for each lattice node
            budget_fraction: Fraction of nodes to sample (e.g., 0.2 for 20%)
            
        Returns:
            selected_nodes: List of selected lattice nodes
        """
        active_nodes = lattice['active_nodes']
        
        if len(active_nodes) == 0:
            return []
        
        # Compute total budget
        total_budget = max(1, int(budget_fraction * len(active_nodes)))
        
        # Compute sampling probabilities
        probabilities = self._compute_sampling_probabilities(
            entropy_map, active_nodes, lattice
        )
        
        # Partition nodes into strata by entropy
        strata = self._partition_by_entropy(
            active_nodes, entropy_map, self.num_strata
        )
        
        # Perform stratified sampling
        selected_nodes = []
        
        for stratum_nodes, stratum_entropies in strata:
            if len(stratum_nodes) == 0:
                continue
                
            # Allocate budget to this stratum
            stratum_budget = max(1, int(total_budget * len(stratum_nodes) / len(active_nodes)))
            
            # Normalize probabilities within stratum
            stratum_indices = [active_nodes.index(node) for node in stratum_nodes]
            stratum_probs = probabilities[stratum_indices]

            # Ensure stratum probabilities are valid
            stratum_probs = torch.clamp(stratum_probs, min=1e-8)
            stratum_sum = torch.sum(stratum_probs)

            if stratum_sum <= 1e-8:
                # Fallback to uniform distribution within stratum
                stratum_probs = torch.ones_like(stratum_probs) / len(stratum_probs)
            else:
                stratum_probs = stratum_probs / stratum_sum
            
            # Sample from stratum
            stratum_selected = self._multinomial_sample(
                stratum_nodes, stratum_probs, stratum_budget
            )
            
            selected_nodes.extend(stratum_selected)
        
        # Fill remaining budget with highest entropy nodes
        while len(selected_nodes) < total_budget:
            # Find highest entropy node not yet selected
            remaining_nodes = [
                node for node in active_nodes 
                if node not in selected_nodes
            ]
            
            if not remaining_nodes:
                break
                
            remaining_indices = [active_nodes.index(node) for node in remaining_nodes]
            remaining_entropies = entropy_map[remaining_indices]
            
            best_idx = torch.argmax(remaining_entropies)
            best_node = remaining_nodes[best_idx]
            selected_nodes.append(best_node)
        
        logger.debug(f"Sampled {len(selected_nodes)}/{len(active_nodes)} nodes "
                    f"(budget: {total_budget})")
        
        return selected_nodes
    
    def _compute_sampling_probabilities(
        self,
        entropy_map: torch.Tensor,
        active_nodes: List[Tuple[int, int, int]],
        lattice: Dict
    ) -> torch.Tensor:
        """
        Compute sampling probabilities based on entropy and scale

        Implements Equation (4) from paper:
        π_k(t,f,s) = exp(β_s · H_{t,f,s}^{(k)}) / Σ exp(β_{s'} · H_{t',f',s'}^{(k)})
        """
        if len(entropy_map) == 0:
            return torch.tensor([])

        device = entropy_map.device

        # Scale-dependent temperature adjustment (β_s from paper)
        scale_adjusted_probs = torch.zeros_like(entropy_map)

        for i, (t, f, s) in enumerate(active_nodes):
            if i >= len(entropy_map):
                continue

            # Scale-dependent temperature: β_s = β_0 · (1 + δ_s · s/S)
            max_scale = lattice.get('max_scales', 4)
            delta_s = 0.1  # Scale adjustment factor
            beta_s = self.config.temperature * (1 + delta_s * s / max_scale)

            # Clamp entropy values to prevent overflow/underflow
            entropy_val = torch.clamp(entropy_map[i], min=-10.0, max=10.0)

            # Compute probability according to Equation (4)
            scale_adjusted_probs[i] = torch.exp(beta_s * entropy_val)

        # Ensure probabilities are valid (positive and finite)
        scale_adjusted_probs = torch.clamp(scale_adjusted_probs, min=1e-8)
        scale_adjusted_probs = torch.where(
            torch.isfinite(scale_adjusted_probs),
            scale_adjusted_probs,
            torch.ones_like(scale_adjusted_probs) * 1e-8
        )

        # Normalize to probability distribution (denominator in Equation 4)
        prob_sum = torch.sum(scale_adjusted_probs)
        if prob_sum <= 1e-8:
            # Fallback to uniform distribution if all probabilities are too small
            probabilities = torch.ones_like(scale_adjusted_probs) / len(scale_adjusted_probs)
        else:
            probabilities = scale_adjusted_probs / prob_sum

        return probabilities
    
    def _partition_by_entropy(
        self,
        active_nodes: List[Tuple[int, int, int]],
        entropy_map: torch.Tensor,
        num_strata: int
    ) -> List[Tuple[List[Tuple[int, int, int]], torch.Tensor]]:
        """Partition nodes into strata based on entropy values"""
        if len(active_nodes) == 0 or len(entropy_map) == 0:
            return []
        
        # Determine entropy quantiles for stratification
        valid_entropies = entropy_map[:len(active_nodes)]
        quantiles = torch.quantile(
            valid_entropies,
            torch.linspace(0, 1, num_strata + 1, device=valid_entropies.device)
        )
        
        strata = []
        
        for i in range(num_strata):
            lower_bound = quantiles[i] if i > 0 else float('-inf')
            upper_bound = quantiles[i + 1] if i < num_strata - 1 else float('inf')
            
            # Find nodes in this stratum
            stratum_nodes = []
            stratum_entropies = []
            
            for j, node in enumerate(active_nodes):
                if j >= len(entropy_map):
                    continue
                    
                entropy_val = entropy_map[j]
                
                if lower_bound <= entropy_val < upper_bound or \
                   (i == num_strata - 1 and entropy_val == upper_bound):
                    stratum_nodes.append(node)
                    stratum_entropies.append(entropy_val)
            
            if stratum_nodes:
                strata.append((stratum_nodes, torch.tensor(stratum_entropies)))
        
        return strata
    
    def _multinomial_sample(
        self,
        nodes: List[Tuple[int, int, int]],
        probabilities: torch.Tensor,
        budget: int
    ) -> List[Tuple[int, int, int]]:
        """Sample nodes using multinomial distribution with robust validation"""
        if len(nodes) == 0 or budget <= 0:
            return []

        budget = min(budget, len(nodes))

        # Validate probabilities before multinomial sampling
        probabilities = torch.clamp(probabilities, min=1e-8)
        prob_sum = torch.sum(probabilities)

        if prob_sum <= 1e-8 or not torch.isfinite(prob_sum):
            # Fallback to uniform sampling if probabilities are invalid
            indices = torch.randperm(len(nodes))[:budget]
            selected_nodes = [nodes[idx] for idx in indices]
        else:
            # Normalize probabilities
            probabilities = probabilities / prob_sum

            # Sample without replacement
            try:
                selected_indices = torch.multinomial(
                    probabilities,
                    num_samples=budget,
                    replacement=False
                )
                selected_nodes = [nodes[idx] for idx in selected_indices]
            except RuntimeError:
                # Fallback to uniform sampling if multinomial fails
                indices = torch.randperm(len(nodes))[:budget]
                selected_nodes = [nodes[idx] for idx in indices]
        
        return selected_nodes


class TemperatureScheduler:
    """Temperature scheduler for adaptive sampling"""
    
    def __init__(self, initial_temp: float = 5.0, decay_rate: float = 0.95):
        self.initial_temp = initial_temp
        self.decay_rate = decay_rate
        self.current_temp = initial_temp
    
    def step(self, diffusion_step: int, total_steps: int) -> float:
        """Update temperature based on diffusion progress"""
        progress = diffusion_step / total_steps
        
        # Higher temperature early in diffusion (more exploration)
        # Lower temperature later (more exploitation)
        self.current_temp = self.initial_temp * (1 - progress * 0.5)
        
        return self.current_temp
    
    def get_temperature(self) -> float:
        """Get current temperature"""
        return self.current_temp


class AdaptiveBudgetScheduler:
    """Adaptive budget scheduler based on entropy dynamics"""
    
    def __init__(self, base_budget: float = 0.2, min_budget: float = 0.1, max_budget: float = 0.4):
        self.base_budget = base_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.entropy_history = []
    
    def update_budget(
        self,
        current_entropy: torch.Tensor,
        diffusion_step: int,
        total_steps: int
    ) -> float:
        """Update sampling budget based on entropy dynamics"""
        # Track entropy statistics
        mean_entropy = torch.mean(current_entropy)
        self.entropy_history.append(mean_entropy.item())
        
        # Compute entropy change rate
        if len(self.entropy_history) > 1:
            entropy_change = self.entropy_history[-1] - self.entropy_history[-2]
            
            # Increase budget when entropy is changing rapidly
            if abs(entropy_change) > 0.1:
                budget_multiplier = 1.2
            else:
                budget_multiplier = 0.9
        else:
            budget_multiplier = 1.0
        
        # Adjust based on diffusion progress
        progress = diffusion_step / total_steps
        
        # Higher budget early in diffusion
        progress_multiplier = 1.5 - 0.5 * progress
        
        # Compute final budget
        adaptive_budget = self.base_budget * budget_multiplier * progress_multiplier
        adaptive_budget = np.clip(adaptive_budget, self.min_budget, self.max_budget)
        
        return adaptive_budget


class SamplingAnalyzer:
    """Analyzer for sampling strategy performance"""
    
    def __init__(self):
        self.sampling_history = []
        self.entropy_coverage = []
        self.efficiency_metrics = []
    
    def analyze_sampling(
        self,
        selected_nodes: List[Tuple[int, int, int]],
        all_nodes: List[Tuple[int, int, int]],
        entropy_map: torch.Tensor
    ) -> Dict:
        """Analyze sampling strategy effectiveness"""
        if len(selected_nodes) == 0 or len(entropy_map) == 0:
            return {}
        
        # Compute coverage metrics
        coverage_ratio = len(selected_nodes) / len(all_nodes)
        
        # Compute entropy coverage
        selected_indices = [all_nodes.index(node) for node in selected_nodes if node in all_nodes]
        if selected_indices:
            selected_entropies = entropy_map[selected_indices]
            total_entropy = torch.sum(entropy_map)
            covered_entropy = torch.sum(selected_entropies)
            entropy_coverage = covered_entropy / (total_entropy + 1e-8)
        else:
            entropy_coverage = 0.0
        
        # Compute efficiency (entropy per node)
        if selected_indices:
            efficiency = torch.mean(selected_entropies).item()
        else:
            efficiency = 0.0
        
        # Scale distribution analysis
        scale_distribution = {}
        for t, f, s in selected_nodes:
            scale_distribution[s] = scale_distribution.get(s, 0) + 1
        
        analysis = {
            'coverage_ratio': coverage_ratio,
            'entropy_coverage': entropy_coverage.item() if torch.is_tensor(entropy_coverage) else entropy_coverage,
            'efficiency': efficiency,
            'scale_distribution': scale_distribution,
            'num_selected': len(selected_nodes),
            'num_total': len(all_nodes)
        }
        
        self.sampling_history.append(analysis)
        
        return analysis
    
    def get_sampling_statistics(self) -> Dict:
        """Get overall sampling statistics"""
        if not self.sampling_history:
            return {}
        
        # Aggregate statistics
        avg_coverage = np.mean([h['coverage_ratio'] for h in self.sampling_history])
        avg_entropy_coverage = np.mean([h['entropy_coverage'] for h in self.sampling_history])
        avg_efficiency = np.mean([h['efficiency'] for h in self.sampling_history])
        
        return {
            'average_coverage_ratio': avg_coverage,
            'average_entropy_coverage': avg_entropy_coverage,
            'average_efficiency': avg_efficiency,
            'num_samples': len(self.sampling_history)
        }
