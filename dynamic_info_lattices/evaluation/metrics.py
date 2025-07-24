"""
Evaluation metrics for time series forecasting

Implements all metrics mentioned in the paper:
CRPS, MAE, MSE, QL, Energy Score, etc.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    return_detailed: bool = False
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        predictions: Predicted values [batch_size, seq_len, features]
        targets: Target values [batch_size, seq_len, features]
        quantiles: Quantiles for quantile loss computation
        return_detailed: Whether to return detailed per-feature metrics
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = mean_absolute_error(predictions, targets)
    metrics['mse'] = mean_squared_error(predictions, targets)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mape'] = mean_absolute_percentage_error(predictions, targets)
    
    # Probabilistic metrics (assuming predictions are samples)
    if predictions.ndim > targets.ndim:
        # Multiple samples case
        metrics['crps'] = continuous_ranked_probability_score(predictions, targets)
        metrics['energy_score'] = energy_score(predictions, targets)
        metrics['coverage'] = coverage_probability(predictions, targets, quantiles)
        
        # Quantile losses
        for q in quantiles:
            pred_quantile = np.quantile(predictions, q, axis=0)
            metrics[f'ql_{q}'] = quantile_loss(pred_quantile, targets, q)
    
    # Additional metrics
    metrics['r2'] = r_squared(predictions, targets)
    metrics['correlation'] = correlation_coefficient(predictions, targets)
    
    if return_detailed:
        # Per-feature metrics
        detailed_metrics = {}
        for i in range(targets.shape[-1]):
            feature_metrics = compute_metrics(
                predictions[..., i:i+1], 
                targets[..., i:i+1], 
                quantiles, 
                return_detailed=False
            )
            detailed_metrics[f'feature_{i}'] = feature_metrics
        
        metrics['detailed'] = detailed_metrics
    
    return metrics


def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error"""
    return np.mean(np.abs(predictions - targets))


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Squared Error"""
    return np.mean((predictions - targets) ** 2)


def mean_absolute_percentage_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = np.abs(targets) > 1e-8
    if not mask.any():
        return 0.0
    
    return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100


def r_squared(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute R-squared coefficient"""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def correlation_coefficient(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Pearson correlation coefficient"""
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    if len(predictions_flat) < 2:
        return 0.0
    
    correlation_matrix = np.corrcoef(predictions_flat, targets_flat)
    return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0


def continuous_ranked_probability_score(
    predictions: np.ndarray, 
    targets: np.ndarray
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS)
    
    Args:
        predictions: Prediction samples [num_samples, batch_size, seq_len, features]
        targets: Target values [batch_size, seq_len, features]
        
    Returns:
        crps: CRPS value
    """
    if predictions.ndim == targets.ndim:
        # Single prediction case - use absolute error as approximation
        return mean_absolute_error(predictions, targets)
    
    num_samples = predictions.shape[0]
    crps_values = []
    
    # Flatten spatial dimensions for easier computation
    predictions_flat = predictions.reshape(num_samples, -1)
    targets_flat = targets.flatten()
    
    for i in range(predictions_flat.shape[1]):
        pred_samples = predictions_flat[:, i]
        target_val = targets_flat[i]
        
        # Sort prediction samples
        pred_sorted = np.sort(pred_samples)
        
        # Compute CRPS using the formula:
        # CRPS = E[|X - Y|] - 0.5 * E[|X - X'|]
        # where X, X' are independent samples from prediction distribution
        # and Y is the observation
        
        # First term: E[|X - Y|]
        term1 = np.mean(np.abs(pred_samples - target_val))
        
        # Second term: 0.5 * E[|X - X'|]
        term2 = 0.0
        for j in range(num_samples):
            for k in range(j + 1, num_samples):
                term2 += np.abs(pred_samples[j] - pred_samples[k])
        
        term2 = term2 / (num_samples * (num_samples - 1) / 2)
        
        crps_val = term1 - 0.5 * term2
        crps_values.append(crps_val)
    
    return np.mean(crps_values)


def energy_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Energy Score
    
    Args:
        predictions: Prediction samples [num_samples, batch_size, seq_len, features]
        targets: Target values [batch_size, seq_len, features]
        
    Returns:
        energy_score: Energy score value
    """
    if predictions.ndim == targets.ndim:
        # Single prediction case
        return mean_absolute_error(predictions, targets)
    
    num_samples = predictions.shape[0]
    
    # Flatten spatial dimensions
    predictions_flat = predictions.reshape(num_samples, -1)
    targets_flat = targets.flatten()
    
    energy_values = []
    
    for i in range(predictions_flat.shape[1]):
        pred_samples = predictions_flat[:, i]
        target_val = targets_flat[i]
        
        # Energy score formula:
        # ES = E[||X - Y||] - 0.5 * E[||X - X'||]
        
        # First term
        term1 = np.mean(np.abs(pred_samples - target_val))
        
        # Second term
        term2 = 0.0
        for j in range(num_samples):
            for k in range(j + 1, num_samples):
                term2 += np.abs(pred_samples[j] - pred_samples[k])
        
        term2 = term2 / (num_samples * (num_samples - 1) / 2)
        
        energy_val = term1 - 0.5 * term2
        energy_values.append(energy_val)
    
    return np.mean(energy_values)


def quantile_loss(predictions: np.ndarray, targets: np.ndarray, quantile: float) -> float:
    """
    Compute Quantile Loss
    
    Args:
        predictions: Predicted quantile values
        targets: Target values
        quantile: Quantile level (e.g., 0.5 for median)
        
    Returns:
        ql: Quantile loss value
    """
    errors = targets - predictions
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def coverage_probability(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    quantiles: List[float]
) -> Dict[str, float]:
    """
    Compute coverage probability for prediction intervals
    
    Args:
        predictions: Prediction samples [num_samples, batch_size, seq_len, features]
        targets: Target values [batch_size, seq_len, features]
        quantiles: List of quantiles to compute coverage for
        
    Returns:
        coverage: Dictionary of coverage probabilities
    """
    coverage = {}
    
    if predictions.ndim == targets.ndim:
        # Single prediction case - cannot compute coverage
        return {f'coverage_{q}': 0.0 for q in quantiles}
    
    targets_flat = targets.flatten()
    
    for q in quantiles:
        if q <= 0.5:
            # Lower quantile
            lower_q = q
            upper_q = 1 - q
            
            pred_lower = np.quantile(predictions, lower_q, axis=0).flatten()
            pred_upper = np.quantile(predictions, upper_q, axis=0).flatten()
            
            # Check if targets fall within prediction interval
            within_interval = (targets_flat >= pred_lower) & (targets_flat <= pred_upper)
            coverage[f'coverage_{lower_q}_{upper_q}'] = np.mean(within_interval)
    
    return coverage


def directional_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute directional accuracy (for time series with trends)
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        accuracy: Directional accuracy
    """
    if predictions.shape[1] < 2:  # Need at least 2 time steps
        return 0.0
    
    # Compute differences (trends)
    pred_diff = np.diff(predictions, axis=1)
    target_diff = np.diff(targets, axis=1)
    
    # Check if directions match
    same_direction = np.sign(pred_diff) == np.sign(target_diff)
    
    return np.mean(same_direction)


def normalized_deviation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Normalized Deviation
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        nd: Normalized deviation
    """
    numerator = np.sum(np.abs(targets - predictions))
    denominator = np.sum(np.abs(targets))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def root_relative_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Root Relative Squared Error
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        rrse: Root relative squared error
    """
    numerator = np.sum((targets - predictions) ** 2)
    denominator = np.sum((targets - np.mean(targets)) ** 2)
    
    if denominator == 0:
        return 0.0
    
    return np.sqrt(numerator / denominator)


class MetricsCalculator:
    """Comprehensive metrics calculator with caching and batch processing"""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.cached_metrics = {}
    
    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        cache_key: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute all available metrics"""
        if cache_key and cache_key in self.cached_metrics:
            return self.cached_metrics[cache_key]
        
        metrics = {}
        
        # Basic metrics
        metrics.update(compute_metrics(predictions, targets, self.quantiles))
        
        # Additional metrics
        metrics['directional_accuracy'] = directional_accuracy(predictions, targets)
        metrics['normalized_deviation'] = normalized_deviation(predictions, targets)
        metrics['rrse'] = root_relative_squared_error(predictions, targets)
        
        if cache_key:
            self.cached_metrics[cache_key] = metrics
        
        return metrics
    
    def clear_cache(self):
        """Clear cached metrics"""
        self.cached_metrics.clear()


def statistical_significance_test(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    targets: np.ndarray,
    metric_fn: callable = mean_absolute_error,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool]]:
    """
    Test statistical significance between two prediction methods
    
    Args:
        predictions1: Predictions from method 1
        predictions2: Predictions from method 2
        targets: Target values
        metric_fn: Metric function to use for comparison
        alpha: Significance level
        
    Returns:
        test_results: Dictionary with test statistics and p-value
    """
    # Compute metrics for both methods
    errors1 = np.abs(predictions1 - targets).flatten()
    errors2 = np.abs(predictions2 - targets).flatten()
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(errors1) + np.var(errors2)) / 2)
    cohens_d = (np.mean(errors1) - np.mean(errors2)) / pooled_std if pooled_std > 0 else 0.0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'cohens_d': cohens_d,
        'mean_error_1': np.mean(errors1),
        'mean_error_2': np.mean(errors2)
    }
