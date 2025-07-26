#!/usr/bin/env python3
"""
Statistical significance testing for KDD experimental results
Implements proper statistical validation with multiple comparison correction
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Statistical analysis for time series forecasting experiments"""
    
    def __init__(self, results_dir: str = "experiments/kdd"):
        self.results_dir = Path(results_dir)
        self.alpha = 0.05  # Significance level
        self.results_df = None
        
    def load_experimental_results(self) -> pd.DataFrame:
        """Load all experimental results from JSON files"""
        
        results = []
        
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return pd.DataFrame()
        
        # Recursively find all result JSON files
        for json_file in self.results_dir.rglob("*_results.json"):
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        if not results:
            print("No experimental results found")
            return pd.DataFrame()
        
        self.results_df = pd.DataFrame(results)
        print(f"Loaded {len(self.results_df)} experimental results")
        
        return self.results_df
    
    def perform_wilcoxon_tests(self, metric: str = 'best_test_loss') -> pd.DataFrame:
        """Perform pairwise Wilcoxon signed-rank tests between methods"""
        
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()
        
        # Get unique methods and datasets
        methods = self.results_df['method'].unique()
        datasets = self.results_df['dataset'].unique()
        
        # Prepare results for pairwise comparisons
        pairwise_results = []
        
        for dataset in datasets:
            dataset_results = self.results_df[self.results_df['dataset'] == dataset]
            
            # Get results for each method on this dataset
            method_results = {}
            for method in methods:
                method_data = dataset_results[dataset_results['method'] == method]
                if len(method_data) > 0:
                    method_results[method] = method_data[metric].values
            
            # Perform pairwise tests
            method_list = list(method_results.keys())
            for i, method1 in enumerate(method_list):
                for j, method2 in enumerate(method_list):
                    if i < j:  # Avoid duplicate comparisons
                        data1 = method_results[method1]
                        data2 = method_results[method2]
                        
                        if len(data1) > 1 and len(data2) > 1:
                            try:
                                # Wilcoxon signed-rank test
                                statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
                                
                                # Effect size (rank-biserial correlation)
                                n = len(data1)
                                effect_size = 1 - (2 * statistic) / (n * (n + 1))
                                
                                pairwise_results.append({
                                    'dataset': dataset,
                                    'method1': method1,
                                    'method2': method2,
                                    'statistic': statistic,
                                    'p_value': p_value,
                                    'effect_size': effect_size,
                                    'mean1': np.mean(data1),
                                    'mean2': np.mean(data2),
                                    'std1': np.std(data1),
                                    'std2': np.std(data2),
                                    'n1': len(data1),
                                    'n2': len(data2)
                                })
                            except Exception as e:
                                print(f"Error in Wilcoxon test for {method1} vs {method2} on {dataset}: {e}")
        
        pairwise_df = pd.DataFrame(pairwise_results)
        
        if not pairwise_df.empty:
            # Apply Bonferroni correction
            pairwise_df['p_value_corrected'] = pairwise_df['p_value'] * len(pairwise_df)
            pairwise_df['p_value_corrected'] = pairwise_df['p_value_corrected'].clip(upper=1.0)
            pairwise_df['significant'] = pairwise_df['p_value_corrected'] < self.alpha
        
        return pairwise_df
    
    def perform_friedman_test(self, metric: str = 'best_test_loss') -> Dict:
        """Perform Friedman test for multiple method comparison"""
        
        if self.results_df is None or self.results_df.empty:
            return {}
        
        # Prepare data for Friedman test
        datasets = self.results_df['dataset'].unique()
        methods = self.results_df['method'].unique()
        
        # Create matrix: rows = datasets, columns = methods
        friedman_data = []
        valid_datasets = []
        
        for dataset in datasets:
            dataset_results = self.results_df[self.results_df['dataset'] == dataset]
            
            method_means = []
            all_methods_present = True
            
            for method in methods:
                method_data = dataset_results[dataset_results['method'] == method]
                if len(method_data) > 0:
                    method_means.append(method_data[metric].mean())
                else:
                    all_methods_present = False
                    break
            
            if all_methods_present:
                friedman_data.append(method_means)
                valid_datasets.append(dataset)
        
        if len(friedman_data) < 2:
            return {'error': 'Insufficient data for Friedman test'}
        
        # Perform Friedman test
        friedman_data = np.array(friedman_data)
        try:
            statistic, p_value = friedmanchisquare(*friedman_data.T)
            
            # Calculate average ranks
            ranks = np.argsort(np.argsort(friedman_data, axis=1), axis=1) + 1
            average_ranks = np.mean(ranks, axis=0)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'methods': list(methods),
                'average_ranks': average_ranks.tolist(),
                'valid_datasets': valid_datasets,
                'n_datasets': len(valid_datasets)
            }
        except Exception as e:
            return {'error': f'Friedman test failed: {e}'}
    
    def calculate_effect_sizes(self, metric: str = 'best_test_loss') -> pd.DataFrame:
        """Calculate effect sizes (Cohen's d) between methods"""
        
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()
        
        methods = self.results_df['method'].unique()
        datasets = self.results_df['dataset'].unique()
        
        effect_sizes = []
        
        for dataset in datasets:
            dataset_results = self.results_df[self.results_df['dataset'] == dataset]
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:
                        data1 = dataset_results[dataset_results['method'] == method1][metric]
                        data2 = dataset_results[dataset_results['method'] == method2][metric]
                        
                        if len(data1) > 1 and len(data2) > 1:
                            # Cohen's d
                            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                                (len(data1) + len(data2) - 2))
                            
                            if pooled_std > 0:
                                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                                
                                # Interpret effect size
                                if abs(cohens_d) < 0.2:
                                    interpretation = 'negligible'
                                elif abs(cohens_d) < 0.5:
                                    interpretation = 'small'
                                elif abs(cohens_d) < 0.8:
                                    interpretation = 'medium'
                                else:
                                    interpretation = 'large'
                                
                                effect_sizes.append({
                                    'dataset': dataset,
                                    'method1': method1,
                                    'method2': method2,
                                    'cohens_d': cohens_d,
                                    'interpretation': interpretation,
                                    'mean1': np.mean(data1),
                                    'mean2': np.mean(data2)
                                })
        
        return pd.DataFrame(effect_sizes)
    
    def generate_statistical_report(self, output_file: str = "statistical_analysis_report.txt"):
        """Generate comprehensive statistical analysis report"""
        
        if self.results_df is None:
            self.load_experimental_results()
        
        if self.results_df.empty:
            print("No data available for statistical analysis")
            return
        
        with open(output_file, 'w') as f:
            f.write("STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write("1. DESCRIPTIVE STATISTICS\n")
            f.write("-" * 30 + "\n")
            
            summary_stats = self.results_df.groupby(['method', 'dataset'])['best_test_loss'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(6)
            
            f.write(summary_stats.to_string())
            f.write("\n\n")
            
            # Friedman test
            f.write("2. FRIEDMAN TEST (Multiple Method Comparison)\n")
            f.write("-" * 45 + "\n")
            
            friedman_result = self.perform_friedman_test()
            if 'error' not in friedman_result:
                f.write(f"Chi-square statistic: {friedman_result['statistic']:.4f}\n")
                f.write(f"p-value: {friedman_result['p_value']:.6f}\n")
                f.write(f"Significant: {friedman_result['significant']}\n")
                f.write(f"Number of datasets: {friedman_result['n_datasets']}\n\n")
                
                f.write("Average ranks by method:\n")
                for method, rank in zip(friedman_result['methods'], friedman_result['average_ranks']):
                    f.write(f"  {method:15}: {rank:.2f}\n")
            else:
                f.write(f"Error: {friedman_result['error']}\n")
            
            f.write("\n")
            
            # Pairwise comparisons
            f.write("3. PAIRWISE WILCOXON TESTS (with Bonferroni correction)\n")
            f.write("-" * 55 + "\n")
            
            pairwise_results = self.perform_wilcoxon_tests()
            if not pairwise_results.empty:
                significant_pairs = pairwise_results[pairwise_results['significant']]
                
                f.write(f"Total comparisons: {len(pairwise_results)}\n")
                f.write(f"Significant comparisons: {len(significant_pairs)}\n\n")
                
                if len(significant_pairs) > 0:
                    f.write("Significant differences (p < 0.05 after Bonferroni correction):\n")
                    for _, row in significant_pairs.iterrows():
                        f.write(f"  {row['dataset']}: {row['method1']} vs {row['method2']} "
                               f"(p = {row['p_value_corrected']:.6f})\n")
                else:
                    f.write("No significant differences found after correction.\n")
            
            f.write("\n")
            
            # Effect sizes
            f.write("4. EFFECT SIZES (Cohen's d)\n")
            f.write("-" * 25 + "\n")
            
            effect_sizes = self.calculate_effect_sizes()
            if not effect_sizes.empty:
                large_effects = effect_sizes[effect_sizes['interpretation'] == 'large']
                medium_effects = effect_sizes[effect_sizes['interpretation'] == 'medium']
                
                f.write(f"Large effects (|d| > 0.8): {len(large_effects)}\n")
                f.write(f"Medium effects (0.5 < |d| < 0.8): {len(medium_effects)}\n\n")
                
                if len(large_effects) > 0:
                    f.write("Large effect sizes:\n")
                    for _, row in large_effects.iterrows():
                        f.write(f"  {row['dataset']}: {row['method1']} vs {row['method2']} "
                               f"(d = {row['cohens_d']:.3f})\n")
        
        print(f"Statistical analysis report saved to: {output_file}")
    
    def create_statistical_plots(self, output_dir: str = "statistical_plots"):
        """Create statistical visualization plots"""
        
        if self.results_df is None:
            self.load_experimental_results()
        
        if self.results_df.empty:
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Box plot comparison
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=self.results_df, x='dataset', y='best_test_loss', hue='method')
        plt.xticks(rotation=45, ha='right')
        plt.title('Test Loss Distribution by Method and Dataset')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/method_comparison_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Statistical significance heatmap
        pairwise_results = self.perform_wilcoxon_tests()
        if not pairwise_results.empty:
            # Create significance matrix
            methods = self.results_df['method'].unique()
            datasets = self.results_df['dataset'].unique()
            
            for dataset in datasets:
                dataset_pairs = pairwise_results[pairwise_results['dataset'] == dataset]
                
                if len(dataset_pairs) > 0:
                    # Create matrix
                    sig_matrix = np.ones((len(methods), len(methods)))
                    
                    for _, row in dataset_pairs.iterrows():
                        i = list(methods).index(row['method1'])
                        j = list(methods).index(row['method2'])
                        sig_matrix[i, j] = row['p_value_corrected']
                        sig_matrix[j, i] = row['p_value_corrected']
                    
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(sig_matrix, annot=True, fmt='.3f', 
                               xticklabels=methods, yticklabels=methods,
                               cmap='RdYlBu_r', vmin=0, vmax=0.05)
                    plt.title(f'Statistical Significance Matrix - {dataset}\n(p-values after Bonferroni correction)')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/significance_matrix_{dataset}.png", dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"Statistical plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of experimental results")
    parser.add_argument("--results-dir", default="experiments/kdd",
                       help="Directory containing experimental results")
    parser.add_argument("--output-dir", default="statistical_analysis",
                       help="Output directory for analysis")
    parser.add_argument("--metric", default="best_test_loss",
                       help="Metric to analyze")
    
    args = parser.parse_args()
    
    print("📊 Statistical Significance Analysis for KDD Experiments")
    print("=" * 60)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(args.results_dir)
    
    # Load results
    results_df = analyzer.load_experimental_results()
    
    if results_df.empty:
        print("No experimental results found for analysis")
        return
    
    # Generate comprehensive report
    report_file = f"{args.output_dir}/statistical_analysis_report.txt"
    analyzer.generate_statistical_report(report_file)
    
    # Create plots
    plot_dir = f"{args.output_dir}/plots"
    analyzer.create_statistical_plots(plot_dir)
    
    # Save detailed results
    pairwise_results = analyzer.perform_wilcoxon_tests(args.metric)
    if not pairwise_results.empty:
        pairwise_results.to_csv(f"{args.output_dir}/pairwise_comparisons.csv", index=False)
    
    effect_sizes = analyzer.calculate_effect_sizes(args.metric)
    if not effect_sizes.empty:
        effect_sizes.to_csv(f"{args.output_dir}/effect_sizes.csv", index=False)
    
    friedman_result = analyzer.perform_friedman_test(args.metric)
    with open(f"{args.output_dir}/friedman_test.json", 'w') as f:
        json.dump(friedman_result, f, indent=2, default=str)
    
    print(f"\n✅ Statistical analysis completed!")
    print(f"📁 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
