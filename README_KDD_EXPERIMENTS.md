# KDD-Quality Experimental Framework for Dynamic Information Lattices

This framework provides comprehensive experimental evaluation suitable for top-tier conference submission (KDD). It includes all 13 processed datasets, multiple baseline comparisons, statistical significance testing, and reproducibility features.

## 📊 Datasets (13 Total)

### Large-Scale Datasets
- **electricity**: 26,304 × 320 series (hourly electricity consumption)
- **traffic**: 17,544 × 861 series (hourly traffic occupancy)
- **ecl**: 140,256 × 13 series (electricity consuming load)

### Medium-Scale Datasets  
- **ettm1**: 69,680 × 17 series (15-minute electricity transformer)
- **ettm2**: 69,680 × 17 series (15-minute electricity transformer)
- **weather**: 52,696 × 20 series (10-minute weather measurements)
- **solar**: 52,560 × 137 series (10-minute photovoltaic power)

### Small-Scale Datasets
- **etth1**: 17,420 × 17 series (hourly electricity transformer)
- **etth2**: 17,420 × 17 series (hourly electricity transformer)
- **exchange_rate**: 7,588 × 7 series (daily exchange rates)
- **southern_china**: 679 × 441 series (regional data)
- **gefcom2014**: 528 × 139 series (energy forecasting competition)
- **illness**: 966 × 6 series (weekly influenza surveillance)

## 🔬 Experimental Design

### Baseline Methods (15 Total)
1. **Simple Baselines**: Naive seasonal, Linear trend, ARIMA
2. **Deep Learning**: LSTM, Transformer, Informer, Autoformer, FEDformer, PatchTST
3. **Recent SOTA**: TimesNet, DLinear, N-BEATS
4. **Diffusion Methods**: TSDiff, CSDI, TimeGrad

### Evaluation Metrics
- **Point Forecast**: MAE, RMSE, MAPE, SMAPE, MASE
- **Probabilistic**: CRPS, Quantile Loss, Energy Score, Coverage Probability
- **Distribution**: Wasserstein Distance, KL Divergence
- **Efficiency**: Training Time, Inference Time, Memory Usage, Energy Consumption
- **Robustness**: Noise Robustness, Missing Data Robustness

### Statistical Validation
- **5-fold cross-validation** with multiple random seeds
- **Wilcoxon signed-rank tests** for pairwise comparisons
- **Friedman test** for multiple method comparison
- **Bonferroni correction** for multiple testing
- **Effect size analysis** (Cohen's d)

## 🚀 Running Experiments

### 1. Quick Test (Single Dataset)
```bash
# Test on small dataset
python train_multi_dataset.py --dataset illness --epochs 10 --sequence_length 24 --prediction_length 6

# Test baseline
python baselines/run_baseline.py --method dlinear --dataset illness --epochs 10
```

### 2. Full KDD Experimental Suite
```bash
# Dry run to see what would be submitted
python kdd_experimental_framework.py --dry-run

# Submit all experiments (WARNING: This submits ~1000+ jobs)
python kdd_experimental_framework.py --max-concurrent 20

# Submit specific scales only
python kdd_experimental_framework.py --scales small_scale medium_scale
```

### 3. Monitor Progress
```bash
# Check job status
bjobs | grep kdd_

# Monitor specific experiment
tail -f logs/kdd_dil_electricity_96_24_s42_f0.out

# Kill all experiments if needed
bjobs | grep kdd_ | awk '{print $1}' | xargs bkill
```

## 📈 Analysis and Results

### 1. Collect and Analyze Results
```bash
# Analyze experimental results
python analyze_experimental_results.py --experiments-dir experiments/kdd --stats --plots --latex

# Statistical significance testing
python statistical_analysis.py --results-dir experiments/kdd --output-dir statistical_analysis
```

### 2. Generate KDD Tables
```bash
# Generate LaTeX tables for paper
python generate_kdd_tables.py --results-dir experiments/kdd --output-dir paper_tables
```

## 📋 Experimental Configuration

### Resource Requirements
- **Small datasets**: 8-16GB GPU memory, 8 hours
- **Medium datasets**: 16-24GB GPU memory, 12 hours  
- **Large datasets**: 24-32GB GPU memory, 24 hours
- **Very large datasets**: 32-40GB GPU memory, 24+ hours

### Reproducibility Features
- **Fixed random seeds**: 42, 123, 456, 789, 999
- **Deterministic operations**: CUDA deterministic mode
- **Version control**: All dependencies pinned
- **Environment**: Conda/Docker containers available
- **Hardware specification**: Documented GPU/CPU requirements

## 📊 Expected Results Structure

```
experiments/kdd/
├── small_scale/
│   ├── illness/
│   │   ├── dil/
│   │   │   ├── best_model.pth
│   │   │   └── training_log.json
│   │   ├── dlinear/
│   │   └── patchtst/
│   └── ...
├── medium_scale/
├── large_scale/
└── very_large_scale/

statistical_analysis/
├── statistical_analysis_report.txt
├── pairwise_comparisons.csv
├── effect_sizes.csv
├── friedman_test.json
└── plots/
    ├── method_comparison_boxplot.png
    └── significance_matrix_*.png

paper_tables/
├── main_results_table.tex
├── ablation_study_table.tex
├── scalability_analysis_table.tex
└── statistical_significance_table.tex
```

## 🎯 KDD Submission Checklist

### Experimental Rigor
- ✅ **Comprehensive baselines**: 15 state-of-the-art methods
- ✅ **Multiple datasets**: 13 diverse real-world datasets
- ✅ **Statistical validation**: Proper significance testing
- ✅ **Cross-validation**: 5-fold CV with multiple seeds
- ✅ **Effect size analysis**: Cohen's d for practical significance
- ✅ **Multiple metrics**: Beyond simple MSE/MAE

### Reproducibility
- ✅ **Code availability**: Complete implementation provided
- ✅ **Data preprocessing**: Standardized pipeline
- ✅ **Hyperparameters**: All settings documented
- ✅ **Random seeds**: Fixed for reproducibility
- ✅ **Environment**: Docker/Conda specifications
- ✅ **Hardware requirements**: Clearly specified

### Presentation Quality
- ✅ **Professional tables**: LaTeX formatted results
- ✅ **Statistical significance**: Properly marked in tables
- ✅ **Visualization**: High-quality plots and figures
- ✅ **Error analysis**: Confidence intervals and error bars
- ✅ **Scalability analysis**: Performance vs dataset size
- ✅ **Ablation studies**: Component contribution analysis

## 🔧 Troubleshooting

### Common Issues
1. **Out of memory**: Reduce batch size or use gradient accumulation
2. **Job failures**: Check GPU availability and resource limits
3. **Missing results**: Verify job completion and output directories
4. **Statistical tests fail**: Ensure sufficient data points per method

### Performance Optimization
- Use mixed precision training for large models
- Implement gradient checkpointing for memory efficiency
- Use data parallel training for very large datasets
- Cache preprocessed data to reduce I/O overhead

## 📚 References

This experimental framework follows best practices from:
- NeurIPS/ICML experimental guidelines
- Time series forecasting benchmarking standards
- Statistical significance testing in ML conferences
- Reproducibility checklists for top-tier venues

For questions or issues, please refer to the detailed documentation in each script or contact the development team.
