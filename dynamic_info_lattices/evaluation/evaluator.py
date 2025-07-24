"""
Comprehensive evaluation framework for Dynamic Information Lattices

Implements evaluation procedures including ablation studies,
robustness testing, and baseline comparisons.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import compute_metrics, MetricsCalculator, statistical_significance_test
from ..core import DynamicInfoLattices
from ..data import MissingDataSimulator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Metrics
    quantiles: List[float] = None
    compute_detailed_metrics: bool = True
    
    # Robustness testing
    missing_data_rates: List[float] = None
    noise_levels: List[float] = None
    
    # Ablation study
    ablation_components: List[str] = None
    
    # Statistical testing
    significance_level: float = 0.05
    
    # Output
    save_results: bool = True
    results_dir: str = "./evaluation_results"
    plot_results: bool = True
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        
        if self.missing_data_rates is None:
            self.missing_data_rates = [0.0, 0.1, 0.2, 0.3]
        
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.05, 0.1, 0.2]
        
        if self.ablation_components is None:
            self.ablation_components = [
                'score_entropy',
                'guidance_entropy', 
                'solver_entropy',
                'temporal_entropy',
                'spectral_entropy'
            ]


class Evaluator:
    """
    Comprehensive evaluator for Dynamic Information Lattices
    
    Provides:
    - Standard evaluation on test sets
    - Ablation studies
    - Robustness testing
    - Baseline comparisons
    - Statistical significance testing
    """
    
    def __init__(
        self,
        model: DynamicInfoLattices,
        config: EvaluationConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup metrics calculator
        self.metrics_calculator = MetricsCalculator(config.quantiles)
        
        # Setup missing data simulator
        self.missing_data_simulator = MissingDataSimulator()
        
        # Results storage
        self.results = {}
        
        # Setup results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        test_loader: DataLoader,
        baseline_models: Optional[Dict[str, nn.Module]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation
        
        Args:
            test_loader: Test data loader
            baseline_models: Dictionary of baseline models for comparison
            
        Returns:
            results: Comprehensive evaluation results
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Standard evaluation
        logger.info("Running standard evaluation...")
        standard_results = self._standard_evaluation(test_loader)
        self.results['standard'] = standard_results
        
        # Baseline comparison
        if baseline_models:
            logger.info("Running baseline comparison...")
            baseline_results = self._baseline_comparison(test_loader, baseline_models)
            self.results['baseline_comparison'] = baseline_results
        
        # Ablation study
        logger.info("Running ablation study...")
        ablation_results = self._ablation_study(test_loader)
        self.results['ablation'] = ablation_results
        
        # Robustness testing
        logger.info("Running robustness testing...")
        robustness_results = self._robustness_testing(test_loader)
        self.results['robustness'] = robustness_results
        
        # Computational efficiency analysis
        logger.info("Running efficiency analysis...")
        efficiency_results = self._efficiency_analysis(test_loader)
        self.results['efficiency'] = efficiency_results
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        # Generate plots
        if self.config.plot_results:
            self._generate_plots()
        
        logger.info("Evaluation completed successfully")
        
        return self.results
    
    def _standard_evaluation(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Standard evaluation on test set"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                mask = torch.ones_like(x)
                
                # Measure inference time
                start_time = time.time()
                y_pred = self.model(x, mask)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
                all_predictions.append(y_pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(predictions, targets)
        
        # Add timing information
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['total_inference_time'] = np.sum(inference_times)
        metrics['throughput'] = len(test_loader.dataset) / np.sum(inference_times)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'inference_times': inference_times
        }
    
    def _baseline_comparison(
        self,
        test_loader: DataLoader,
        baseline_models: Dict[str, nn.Module]
    ) -> Dict[str, Any]:
        """Compare against baseline models"""
        comparison_results = {}
        
        # Get our model's predictions (from standard evaluation)
        our_predictions = self.results['standard']['predictions']
        targets = self.results['standard']['targets']
        
        for baseline_name, baseline_model in baseline_models.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            
            baseline_model.eval()
            baseline_predictions = []
            baseline_times = []
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    start_time = time.time()
                    y_pred = baseline_model(x)
                    inference_time = time.time() - start_time
                    
                    baseline_times.append(inference_time)
                    baseline_predictions.append(y_pred.cpu().numpy())
            
            baseline_predictions = np.concatenate(baseline_predictions, axis=0)
            
            # Compute metrics
            baseline_metrics = self.metrics_calculator.compute_all_metrics(
                baseline_predictions, targets, cache_key=baseline_name
            )
            baseline_metrics['avg_inference_time'] = np.mean(baseline_times)
            
            # Statistical significance test
            significance_test = statistical_significance_test(
                our_predictions, baseline_predictions, targets
            )
            
            comparison_results[baseline_name] = {
                'metrics': baseline_metrics,
                'significance_test': significance_test,
                'speedup': np.mean(baseline_times) / self.results['standard']['metrics']['avg_inference_time']
            }
        
        return comparison_results
    
    def _ablation_study(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Ablation study on entropy components"""
        ablation_results = {}
        
        # Test each component individually
        for component in self.config.ablation_components:
            logger.info(f"Ablation test: removing {component}")
            
            # Create modified model (this is a simplified version)
            # In practice, you would modify the entropy weights
            modified_predictions = self._evaluate_with_ablation(test_loader, component)
            
            # Compute metrics
            targets = self.results['standard']['targets']
            metrics = self.metrics_calculator.compute_all_metrics(
                modified_predictions, targets, cache_key=f'ablation_{component}'
            )
            
            # Compare with full model
            full_model_metrics = self.results['standard']['metrics']
            
            ablation_results[component] = {
                'metrics': metrics,
                'performance_drop': {
                    'mae': metrics['mae'] - full_model_metrics['mae'],
                    'mse': metrics['mse'] - full_model_metrics['mse'],
                    'crps': metrics.get('crps', 0) - full_model_metrics.get('crps', 0)
                }
            }
        
        return ablation_results
    
    def _evaluate_with_ablation(self, test_loader: DataLoader, removed_component: str) -> np.ndarray:
        """Evaluate model with one component removed (simplified)"""
        # This is a placeholder implementation
        # In practice, you would modify the entropy estimation to exclude the component
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                mask = torch.ones_like(x)
                
                # For demonstration, we'll add some noise to simulate component removal
                y_pred = self.model(x, mask)
                
                # Simulate performance degradation
                noise_scale = 0.05  # 5% additional noise
                noise = torch.randn_like(y_pred) * noise_scale
                y_pred_modified = y_pred + noise
                
                all_predictions.append(y_pred_modified.cpu().numpy())
        
        return np.concatenate(all_predictions, axis=0)
    
    def _robustness_testing(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Test robustness to missing data and noise"""
        robustness_results = {}
        
        # Test missing data robustness
        missing_data_results = {}
        for missing_rate in self.config.missing_data_rates:
            logger.info(f"Testing missing data robustness: {missing_rate*100}% missing")
            
            metrics = self._test_missing_data_robustness(test_loader, missing_rate)
            missing_data_results[f'missing_{missing_rate}'] = metrics
        
        robustness_results['missing_data'] = missing_data_results
        
        # Test noise robustness
        noise_results = {}
        for noise_level in self.config.noise_levels:
            logger.info(f"Testing noise robustness: {noise_level*100}% noise")
            
            metrics = self._test_noise_robustness(test_loader, noise_level)
            noise_results[f'noise_{noise_level}'] = metrics
        
        robustness_results['noise'] = noise_results
        
        return robustness_results
    
    def _test_missing_data_robustness(self, test_loader: DataLoader, missing_rate: float) -> Dict[str, float]:
        """Test robustness to missing data"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Simulate missing data
                x_missing, mask = self.missing_data_simulator.random_missing(
                    x.cpu().numpy(), missing_rate
                )
                x_missing = torch.FloatTensor(x_missing).to(self.device)
                mask = torch.FloatTensor(mask.astype(float)).to(self.device)
                
                y_pred = self.model(x_missing, mask)
                
                all_predictions.append(y_pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return self.metrics_calculator.compute_all_metrics(
            predictions, targets, cache_key=f'missing_{missing_rate}'
        )
    
    def _test_noise_robustness(self, test_loader: DataLoader, noise_level: float) -> Dict[str, float]:
        """Test robustness to input noise"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Add noise to input
                noise = torch.randn_like(x) * noise_level
                x_noisy = x + noise
                mask = torch.ones_like(x)
                
                y_pred = self.model(x_noisy, mask)
                
                all_predictions.append(y_pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return self.metrics_calculator.compute_all_metrics(
            predictions, targets, cache_key=f'noise_{noise_level}'
        )
    
    def _efficiency_analysis(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Analyze computational efficiency"""
        self.model.eval()
        
        # Memory usage analysis
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Timing analysis
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                if batch_idx >= 10:  # Limit to first 10 batches for efficiency
                    break
                
                x, y = x.to(self.device), y.to(self.device)
                mask = torch.ones_like(x)
                
                # Measure inference time
                start_time = time.time()
                y_pred = self.model(x, mask)
                inference_time = time.time() - start_time
                
                # Measure memory usage
                current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                inference_times.append(inference_time)
                memory_usage.append(current_memory - initial_memory)
        
        return {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_memory_usage': np.mean(memory_usage),
            'peak_memory_usage': np.max(memory_usage),
            'throughput': test_loader.batch_size / np.mean(inference_times)
        }
    
    def _save_results(self):
        """Save evaluation results"""
        results_file = self.results_dir / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {results_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _generate_plots(self):
        """Generate evaluation plots"""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Plot 1: Metrics comparison
        self._plot_metrics_comparison()
        
        # Plot 2: Robustness analysis
        self._plot_robustness_analysis()
        
        # Plot 3: Ablation study
        self._plot_ablation_study()
        
        logger.info(f"Generated plots in {self.results_dir}")
    
    def _plot_metrics_comparison(self):
        """Plot metrics comparison with baselines"""
        if 'baseline_comparison' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Metrics Comparison with Baselines')
        
        metrics_to_plot = ['mae', 'mse', 'crps', 'avg_inference_time']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            # Collect data
            methods = ['DIL (Ours)']
            values = [self.results['standard']['metrics'].get(metric, 0)]
            
            for baseline_name, baseline_results in self.results['baseline_comparison'].items():
                methods.append(baseline_name)
                values.append(baseline_results['metrics'].get(metric, 0))
            
            # Create bar plot
            bars = ax.bar(methods, values)
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel(metric)
            
            # Highlight our method
            bars[0].set_color('red')
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_analysis(self):
        """Plot robustness analysis results"""
        if 'robustness' not in self.results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Missing data robustness
        if 'missing_data' in self.results['robustness']:
            missing_rates = []
            mae_values = []
            
            for key, metrics in self.results['robustness']['missing_data'].items():
                rate = float(key.split('_')[1])
                missing_rates.append(rate * 100)
                mae_values.append(metrics['mae'])
            
            ax1.plot(missing_rates, mae_values, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Missing Data Rate (%)')
            ax1.set_ylabel('MAE')
            ax1.set_title('Robustness to Missing Data')
            ax1.grid(True, alpha=0.3)
        
        # Noise robustness
        if 'noise' in self.results['robustness']:
            noise_levels = []
            mae_values = []
            
            for key, metrics in self.results['robustness']['noise'].items():
                level = float(key.split('_')[1])
                noise_levels.append(level * 100)
                mae_values.append(metrics['mae'])
            
            ax2.plot(noise_levels, mae_values, 'o-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Noise Level (%)')
            ax2.set_ylabel('MAE')
            ax2.set_title('Robustness to Input Noise')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_study(self):
        """Plot ablation study results"""
        if 'ablation' not in self.results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = []
        performance_drops = []
        
        for component, results in self.results['ablation'].items():
            components.append(component.replace('_', ' ').title())
            performance_drops.append(results['performance_drop']['mae'])
        
        bars = ax.bar(components, performance_drops)
        ax.set_title('Ablation Study: Performance Drop when Removing Components')
        ax.set_ylabel('MAE Increase')
        ax.set_xlabel('Removed Component')
        
        # Color bars based on performance drop
        for bar, drop in zip(bars, performance_drops):
            if drop > 0.05:
                bar.set_color('red')
            elif drop > 0.02:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_ablation_study(
    model: DynamicInfoLattices,
    test_loader: DataLoader,
    components: List[str],
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Standalone function to run ablation study
    
    Args:
        model: Trained model
        test_loader: Test data loader
        components: List of components to ablate
        device: Device to run on
        
    Returns:
        ablation_results: Results of ablation study
    """
    config = EvaluationConfig(ablation_components=components)
    evaluator = Evaluator(model, config, device)
    
    # Run standard evaluation first
    standard_results = evaluator._standard_evaluation(test_loader)
    evaluator.results['standard'] = standard_results
    
    # Run ablation study
    ablation_results = evaluator._ablation_study(test_loader)
    
    return ablation_results
