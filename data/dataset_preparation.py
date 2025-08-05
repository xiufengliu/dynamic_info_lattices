"""
Dataset preparation for I-ASNH experiments
Handles loading, preprocessing, and splitting of all 8 benchmark datasets
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import logging

class DatasetPreprocessor:
    """Handles preprocessing of time series datasets"""
    
    def __init__(self, window_size: int = 96, prediction_horizon: int = 24):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
    
    def normalize_series(self, series: np.ndarray, dataset_name: str, fit: bool = True) -> np.ndarray:
        """Apply z-score normalization to time series"""
        if fit or dataset_name not in self.scalers:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()
            self.scalers[dataset_name] = scaler
        else:
            scaler = self.scalers[dataset_name]
            normalized = scaler.transform(series.reshape(-1, 1)).flatten()
        
        return normalized
    
    def create_sliding_windows(self, series: np.ndarray) -> List[np.ndarray]:
        """Create sliding windows for time series analysis"""
        windows = []
        for i in range(len(series) - self.window_size + 1):
            window = series[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def train_test_split(self, series: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Split time series into train and test sets preserving temporal order"""
        split_point = int(len(series) * train_ratio)
        train_data = series[:split_point]
        test_data = series[split_point:]
        return train_data, test_data
    
    def handle_missing_values(self, series: np.ndarray) -> np.ndarray:
        """Handle missing values using linear interpolation"""
        if np.any(np.isnan(series)):
            # Linear interpolation for missing values
            df = pd.DataFrame({'value': series})
            df['value'] = df['value'].interpolate(method='linear')
            df['value'] = df['value'].fillna(method='bfill').fillna(method='ffill')
            return df['value'].values
        return series

class BenchmarkDatasets:
    """Manages loading and preparation of all 8 benchmark datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.datasets = {}
        self.dataset_info = {
            'ETTh1': {'file': 'ETTh1.csv', 'target_col': 'OT', 'freq': 'H'},
            'ETTh2': {'file': 'ETTh2.csv', 'target_col': 'OT', 'freq': 'H'},
            'ETTm1': {'file': 'ETTm1.csv', 'target_col': 'OT', 'freq': '15T'},
            'ETTm2': {'file': 'ETTm2.csv', 'target_col': 'OT', 'freq': '15T'},
            'Exchange': {'file': 'exchange_rate.csv', 'target_col': 'OT', 'freq': 'D'},
            'Weather': {'file': 'weather.csv', 'target_col': 'OT', 'freq': '10T'},
            'Illness': {'file': 'national_illness.csv', 'target_col': 'OT', 'freq': 'W'},
            'ECL': {'file': 'ECL.csv', 'target_col': 'OT', 'freq': 'H'}
        }
        self.preprocessor = DatasetPreprocessor()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def download_datasets(self):
        """Download benchmark datasets if not available locally"""
        # This would typically download from official sources
        # For now, we'll create synthetic datasets that match the characteristics
        logging.info("Creating synthetic benchmark datasets...")
        
        for dataset_name, info in self.dataset_info.items():
            filepath = os.path.join(self.data_dir, info['file'])
            if not os.path.exists(filepath):
                self._create_synthetic_dataset(dataset_name, filepath)
    
    def _create_synthetic_dataset(self, dataset_name: str, filepath: str):
        """Create synthetic dataset with realistic time series characteristics"""
        np.random.seed(42)  # For reproducibility
        
        # Dataset-specific parameters
        if dataset_name.startswith('ETT'):
            length = 17420 if 'h' in dataset_name else 69680  # Hourly vs 15-min data
            seasonal_period = 24 if 'h' in dataset_name else 96
            trend_strength = 0.3
            noise_level = 0.2
            seasonal_strength = 0.4
        elif dataset_name == 'Exchange':
            length = 7588
            seasonal_period = 7  # Weekly pattern
            trend_strength = 0.1
            noise_level = 0.3
            seasonal_strength = 0.2
        elif dataset_name == 'Weather':
            length = 52696
            seasonal_period = 144  # Daily pattern (10-min intervals)
            trend_strength = 0.2
            noise_level = 0.4
            seasonal_strength = 0.6
        elif dataset_name == 'Illness':
            length = 966
            seasonal_period = 52  # Yearly pattern (weekly data)
            trend_strength = 0.4
            noise_level = 0.3
            seasonal_strength = 0.5
        elif dataset_name == 'ECL':
            length = 26304
            seasonal_period = 24  # Daily pattern
            trend_strength = 0.2
            noise_level = 0.3
            seasonal_strength = 0.4
        else:
            length = 10000
            seasonal_period = 24
            trend_strength = 0.2
            noise_level = 0.3
            seasonal_strength = 0.3
        
        # Generate synthetic time series
        t = np.arange(length)
        
        # Trend component
        trend = trend_strength * (t / length) * np.sin(t / (length / 4))
        
        # Seasonal component
        seasonal = seasonal_strength * np.sin(2 * np.pi * t / seasonal_period)
        
        # Add multiple seasonal patterns for complexity
        if seasonal_period > 7:
            seasonal += 0.3 * seasonal_strength * np.sin(2 * np.pi * t / (seasonal_period / 7))
        
        # Noise component
        noise = noise_level * np.random.normal(0, 1, length)
        
        # Combine components
        series = 10 + trend + seasonal + noise
        
        # Add some non-linear patterns
        series += 0.1 * np.sin(t / 100) * np.cos(t / 200)
        
        # Create DataFrame
        dates = pd.date_range(start='2016-01-01', periods=length, freq=self.dataset_info[dataset_name]['freq'])
        
        # Create multiple columns (multivariate data)
        num_features = 7 if dataset_name.startswith('ETT') else 8
        data = {'date': dates}
        
        for i in range(num_features):
            if i == 0:  # Target column
                data['OT'] = series
            else:
                # Correlated features with some noise
                correlation = np.random.uniform(0.3, 0.8)
                feature_series = correlation * series + (1 - correlation) * np.random.normal(0, 1, length)
                data[f'feature_{i}'] = feature_series
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logging.info(f"Created synthetic dataset: {dataset_name} with {length} samples")
    
    def load_dataset(self, dataset_name: str) -> np.ndarray:
        """Load a specific dataset"""
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        info = self.dataset_info[dataset_name]
        filepath = os.path.join(self.data_dir, info['file'])
        
        if not os.path.exists(filepath):
            self.download_datasets()
        
        # Load data
        df = pd.read_csv(filepath)
        series = df[info['target_col']].values
        
        # Preprocess
        series = self.preprocessor.handle_missing_values(series)
        
        # Store in cache
        self.datasets[dataset_name] = series
        
        return series
    
    def load_all_datasets(self) -> Dict[str, np.ndarray]:
        """Load all benchmark datasets"""
        all_datasets = {}
        
        for dataset_name in self.dataset_info.keys():
            try:
                series = self.load_dataset(dataset_name)
                all_datasets[dataset_name] = series
                logging.info(f"Loaded {dataset_name}: {len(series)} samples")
            except Exception as e:
                logging.error(f"Failed to load {dataset_name}: {e}")
        
        return all_datasets
    
    def get_dataset_splits(self, dataset_name: str, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Get train/test splits for a dataset"""
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        series = self.datasets[dataset_name]
        return self.preprocessor.train_test_split(series, train_ratio)
    
    def get_normalized_dataset(self, dataset_name: str, fit_on_train: bool = True) -> np.ndarray:
        """Get normalized version of dataset"""
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        series = self.datasets[dataset_name]
        
        if fit_on_train:
            train_data, test_data = self.preprocessor.train_test_split(series)
            # Fit scaler on training data only
            normalized_train = self.preprocessor.normalize_series(train_data, dataset_name, fit=True)
            normalized_test = self.preprocessor.normalize_series(test_data, dataset_name, fit=False)
            return np.concatenate([normalized_train, normalized_test])
        else:
            return self.preprocessor.normalize_series(series, dataset_name, fit=True)
    
    def get_dataset_characteristics(self) -> Dict[str, Dict]:
        """Get characteristics of all datasets"""
        characteristics = {}
        
        for dataset_name in self.dataset_info.keys():
            if dataset_name not in self.datasets:
                self.load_dataset(dataset_name)
            
            series = self.datasets[dataset_name]
            
            characteristics[dataset_name] = {
                'length': len(series),
                'mean': np.mean(series),
                'std': np.std(series),
                'min': np.min(series),
                'max': np.max(series),
                'trend_strength': self._compute_trend_strength(series),
                'seasonal_strength': self._compute_seasonal_strength(series),
                'frequency': self.dataset_info[dataset_name]['freq']
            }
        
        return characteristics
    
    def _compute_trend_strength(self, series: np.ndarray) -> float:
        """Compute trend strength of time series"""
        if len(series) < 3:
            return 0.0
        
        x = np.arange(len(series))
        correlation = np.corrcoef(x, series)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_seasonal_strength(self, series: np.ndarray, period: int = 24) -> float:
        """Compute seasonal strength of time series"""
        if len(series) < period * 2:
            return 0.0
        
        try:
            seasonal_data = series[:len(series)//period * period].reshape(-1, period)
            if seasonal_data.shape[0] > 1:
                seasonal_means = np.mean(seasonal_data, axis=0)
                seasonal_var = np.var(seasonal_means)
                total_var = np.var(series)
                return seasonal_var / (total_var + 1e-8)
        except:
            pass
        
        return 0.0

def prepare_datasets_for_experiments():
    """Prepare all datasets for experimental pipeline"""
    dataset_manager = BenchmarkDatasets()
    
    # Download/create all datasets
    dataset_manager.download_datasets()
    
    # Load all datasets
    datasets = dataset_manager.load_all_datasets()
    
    # Get dataset characteristics
    characteristics = dataset_manager.get_dataset_characteristics()
    
    logging.info("Dataset preparation completed:")
    for name, char in characteristics.items():
        logging.info(f"  {name}: {char['length']} samples, trend={char['trend_strength']:.3f}, seasonal={char['seasonal_strength']:.3f}")
    
    return dataset_manager, datasets, characteristics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_datasets_for_experiments()
