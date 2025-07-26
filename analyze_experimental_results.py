#!/usr/bin/env python3
"""
Analyze and compile experimental results from all datasets
Generate tables and plots for the paper
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from typing import Dict, List, Tuple
import argparse

# Import our modules
from dynamic_info_lattices.data.processed_datasets import get_available_datasets, get_dataset_info


def collect_experimental_results(experiments_dir: str = "experiments") -> pd.DataFrame:
    """Collect all experimental results into a DataFrame"""
    
    results = []
    experiments_path = Path(experiments_dir)
    
    if not experiments_path.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return pd.DataFrame()
    
    # Iterate through all experiment groups
    for exp_group_dir in experiments_path.iterdir():
        if not exp_group_dir.is_dir():
            continue
        
        exp_group = exp_group_dir.name
        print(f"Processing experiment group: {exp_group}")
        
        # Iterate through datasets in this group
        for dataset_dir in exp_group_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name
            print(f"  Processing dataset: {dataset}")
            
            # Look for model checkpoints and logs
            best_model_path = dataset_dir / "best_model.pth"
            
            if best_model_path.exists():
                try:
                    # Load model checkpoint
                    checkpoint = torch.load(best_model_path, map_location='cpu')
                    
                    # Extract metrics
                    train_loss = checkpoint.get('train_loss', np.nan)
                    test_loss = checkpoint.get('test_loss', np.nan)
                    epoch = checkpoint.get('epoch', np.nan)
                    
                    # Get dataset info
                    dataset_info = get_dataset_info(dataset)
                    
                    result = {
                        'experiment_group': exp_group,
                        'dataset': dataset,
                        'num_series': dataset_info['num_series'],
                        'dataset_length': dataset_info['length'],
                        'frequency': dataset_info.get('frequency', 'unknown'),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'final_epoch': epoch,
                        'model_path': str(best_model_path)
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"    Error loading {best_model_path}: {e}")
            else:
                print(f"    No best model found for {dataset}")
    
    return pd.DataFrame(results)


def generate_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table for the paper"""
    
    if results_df.empty:
        return pd.DataFrame()
    
    # Group by dataset and calculate statistics
    summary = results_df.groupby('dataset').agg({
        'num_series': 'first',
        'dataset_length': 'first',
        'frequency': 'first',
        'train_loss': ['mean', 'std'],
        'test_loss': ['mean', 'std'],
        'final_epoch': 'mean'
    }).round(6)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns]
    
    # Sort by test loss
    summary = summary.sort_values('test_loss_mean')
    
    return summary


def create_performance_plots(results_df: pd.DataFrame, output_dir: str = "plots"):
    """Create performance visualization plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df.empty:
        print("No results to plot")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Test Loss by Dataset
    plt.figure(figsize=(12, 8))
    datasets = results_df['dataset'].unique()
    test_losses = [results_df[results_df['dataset'] == d]['test_loss'].values for d in datasets]
    
    plt.boxplot(test_losses, labels=datasets)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Distribution by Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/test_loss_by_dataset.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance vs Dataset Size
    plt.figure(figsize=(10, 6))
    summary = results_df.groupby('dataset').agg({
        'num_series': 'first',
        'dataset_length': 'first',
        'test_loss': 'mean'
    })
    
    plt.scatter(summary['num_series'], summary['test_loss'], 
               s=summary['dataset_length']/1000, alpha=0.7)
    
    for i, dataset in enumerate(summary.index):
        plt.annotate(dataset, (summary.iloc[i]['num_series'], summary.iloc[i]['test_loss']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Number of Series')
    plt.ylabel('Mean Test Loss')
    plt.title('Performance vs Dataset Complexity\n(Bubble size = Dataset Length / 1000)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_vs_complexity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Training Progress Heatmap
    plt.figure(figsize=(12, 8))
    pivot_data = results_df.pivot_table(
        values='test_loss', 
        index='dataset', 
        columns='experiment_group', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis_r')
    plt.title('Test Loss by Dataset and Experiment Group')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


def generate_latex_table(summary_df: pd.DataFrame, output_file: str = "results_table.tex"):
    """Generate LaTeX table for the paper"""
    
    if summary_df.empty:
        print("No results to generate table")
        return
    
    # Prepare data for LaTeX
    latex_data = []
    
    for dataset, row in summary_df.iterrows():
        latex_row = {
            'Dataset': dataset.replace('_', '\\_'),
            'Series': int(row['num_series_first']),
            'Length': int(row['dataset_length_first']),
            'Frequency': row['frequency_first'],
            'Train Loss': f"{row['train_loss_mean']:.4f} ± {row['train_loss_std']:.4f}",
            'Test Loss': f"{row['test_loss_mean']:.4f} ± {row['test_loss_std']:.4f}",
            'Epochs': int(row['final_epoch_mean'])
        }
        latex_data.append(latex_row)
    
    # Create DataFrame for LaTeX
    latex_df = pd.DataFrame(latex_data)
    
    # Generate LaTeX table
    latex_table = latex_df.to_latex(
        index=False,
        escape=False,
        column_format='l|r|r|l|c|c|r',
        caption='Experimental Results on All Datasets',
        label='tab:all_results'
    )
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")


def generate_paper_statistics(results_df: pd.DataFrame):
    """Generate key statistics for the paper"""
    
    if results_df.empty:
        print("No results for statistics")
        return
    
    stats = {
        'total_datasets': results_df['dataset'].nunique(),
        'total_experiments': len(results_df),
        'total_series': results_df.groupby('dataset')['num_series'].first().sum(),
        'total_datapoints': results_df.groupby('dataset')['dataset_length'].first().sum(),
        'best_dataset': results_df.loc[results_df['test_loss'].idxmin(), 'dataset'],
        'best_test_loss': results_df['test_loss'].min(),
        'mean_test_loss': results_df['test_loss'].mean(),
        'std_test_loss': results_df['test_loss'].std(),
        'datasets_by_size': results_df.groupby('dataset')['num_series'].first().sort_values(ascending=False).to_dict()
    }
    
    print("\n" + "="*60)
    print("EXPERIMENTAL STATISTICS FOR PAPER")
    print("="*60)
    print(f"Total Datasets: {stats['total_datasets']}")
    print(f"Total Experiments: {stats['total_experiments']}")
    print(f"Total Time Series: {stats['total_series']:,}")
    print(f"Total Data Points: {stats['total_datapoints']:,}")
    print(f"Best Dataset: {stats['best_dataset']} (Test Loss: {stats['best_test_loss']:.6f})")
    print(f"Mean Test Loss: {stats['mean_test_loss']:.6f} ± {stats['std_test_loss']:.6f}")
    print("\nDatasets by Size (Number of Series):")
    for dataset, size in stats['datasets_by_size'].items():
        print(f"  {dataset:15}: {size:4d} series")
    
    # Save statistics to JSON
    with open("experimental_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\nStatistics saved to experimental_statistics.json")


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument("--experiments-dir", default="experiments",
                       help="Directory containing experimental results")
    parser.add_argument("--output-dir", default="analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--plots", action="store_true",
                       help="Generate performance plots")
    parser.add_argument("--latex", action="store_true",
                       help="Generate LaTeX table")
    parser.add_argument("--stats", action="store_true",
                       help="Generate paper statistics")
    
    args = parser.parse_args()
    
    print("🔍 Dynamic Information Lattices - Results Analysis")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect results
    print("Collecting experimental results...")
    results_df = collect_experimental_results(args.experiments_dir)
    
    if results_df.empty:
        print("No experimental results found!")
        return
    
    print(f"Found {len(results_df)} experimental results")
    
    # Generate summary
    print("Generating summary table...")
    summary_df = generate_summary_table(results_df)
    
    # Save raw results
    results_df.to_csv(f"{args.output_dir}/raw_results.csv", index=False)
    summary_df.to_csv(f"{args.output_dir}/summary_results.csv")
    
    print(f"Results saved to {args.output_dir}/")
    
    # Generate plots
    if args.plots:
        print("Generating performance plots...")
        create_performance_plots(results_df, f"{args.output_dir}/plots")
    
    # Generate LaTeX table
    if args.latex:
        print("Generating LaTeX table...")
        generate_latex_table(summary_df, f"{args.output_dir}/results_table.tex")
    
    # Generate statistics
    if args.stats:
        generate_paper_statistics(results_df)
    
    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
