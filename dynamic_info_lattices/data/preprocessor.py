"""
Data preprocessing utilities for time series forecasting
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for time series forecasting
    
    Handles normalization, missing value imputation, outlier detection,
    and data augmentation as specified in the paper.
    """
    
    def __init__(
        self,
        scaler_type: str = "standard",
        handle_missing: str = "interpolate",
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        augmentation: bool = False
    ):
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.augmentation = augmentation
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        self.is_fitted = False
        self.feature_stats = {}
    
    def fit(self, data: np.ndarray) -> 'DataPreprocessor':
        """Fit preprocessor on training data"""
        logger.info("Fitting data preprocessor...")
        
        # Handle missing values
        data_clean = self._handle_missing_values(data)
        
        # Detect and handle outliers
        data_clean = self._handle_outliers(data_clean)
        
        # Fit scaler
        if self.scaler is not None:
            original_shape = data_clean.shape
            data_reshaped = data_clean.reshape(-1, data_clean.shape[-1])
            self.scaler.fit(data_reshaped)
        
        # Compute feature statistics
        self._compute_feature_stats(data_clean)
        
        self.is_fitted = True
        logger.info("Data preprocessor fitted successfully")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        data_clean = self._handle_missing_values(data)
        
        # Handle outliers
        data_clean = self._handle_outliers(data_clean)
        
        # Apply scaling
        if self.scaler is not None:
            original_shape = data_clean.shape
            data_reshaped = data_clean.reshape(-1, data_clean.shape[-1])
            data_scaled = self.scaler.transform(data_reshaped)
            data_clean = data_scaled.reshape(original_shape)
        
        return data_clean
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data"""
        if self.scaler is not None:
            original_shape = data.shape
            data_reshaped = data.reshape(-1, data.shape[-1])
            data_inverse = self.scaler.inverse_transform(data_reshaped)
            return data_inverse.reshape(original_shape)
        return data
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in the data"""
        if not np.isnan(data).any():
            return data
        
        logger.info(f"Handling missing values using method: {self.handle_missing}")
        
        if self.handle_missing == "interpolate":
            # Linear interpolation
            data_filled = data.copy()
            for i in range(data.shape[-1]):  # For each feature
                series = data_filled[:, i] if data.ndim == 2 else data_filled[:, :, i]
                if series.ndim == 1:
                    mask = ~np.isnan(series)
                    if mask.any():
                        indices = np.arange(len(series))
                        series[~mask] = np.interp(indices[~mask], indices[mask], series[mask])
                else:
                    for j in range(series.shape[1]):
                        col = series[:, j]
                        mask = ~np.isnan(col)
                        if mask.any():
                            indices = np.arange(len(col))
                            col[~mask] = np.interp(indices[~mask], indices[mask], col[mask])
            return data_filled
            
        elif self.handle_missing == "forward_fill":
            # Forward fill
            return pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
            
        elif self.handle_missing == "mean":
            # Mean imputation
            return np.where(np.isnan(data), np.nanmean(data, axis=0), data)
            
        elif self.handle_missing == "zero":
            # Zero imputation
            return np.where(np.isnan(data), 0, data)
        
        else:
            raise ValueError(f"Unknown missing value method: {self.handle_missing}")
    
    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """Detect and handle outliers"""
        if self.outlier_method == "none":
            return data
        
        logger.info(f"Handling outliers using method: {self.outlier_method}")
        
        data_clean = data.copy()
        
        if self.outlier_method == "iqr":
            # Interquartile range method
            for i in range(data.shape[-1]):
                feature_data = data[:, i] if data.ndim == 2 else data[:, :, i].flatten()
                
                q1 = np.percentile(feature_data, 25)
                q3 = np.percentile(feature_data, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - self.outlier_threshold * iqr
                upper_bound = q3 + self.outlier_threshold * iqr
                
                # Clip outliers
                if data.ndim == 2:
                    data_clean[:, i] = np.clip(data_clean[:, i], lower_bound, upper_bound)
                else:
                    data_clean[:, :, i] = np.clip(data_clean[:, :, i], lower_bound, upper_bound)
        
        elif self.outlier_method == "zscore":
            # Z-score method
            for i in range(data.shape[-1]):
                feature_data = data[:, i] if data.ndim == 2 else data[:, :, i]
                
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                
                z_scores = np.abs((feature_data - mean_val) / (std_val + 1e-8))
                outlier_mask = z_scores > self.outlier_threshold
                
                # Replace outliers with clipped values
                if data.ndim == 2:
                    data_clean[outlier_mask, i] = np.clip(
                        data_clean[outlier_mask, i],
                        mean_val - self.outlier_threshold * std_val,
                        mean_val + self.outlier_threshold * std_val
                    )
                else:
                    data_clean[outlier_mask, i] = np.clip(
                        data_clean[outlier_mask, i],
                        mean_val - self.outlier_threshold * std_val,
                        mean_val + self.outlier_threshold * std_val
                    )
        
        return data_clean
    
    def _compute_feature_stats(self, data: np.ndarray):
        """Compute and store feature statistics"""
        self.feature_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0)
        }
    
    def get_feature_stats(self) -> Dict:
        """Get computed feature statistics"""
        return self.feature_stats.copy()


class DataAugmenter:
    """Data augmentation for time series"""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        jitter_std: float = 0.03,
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        time_warp_sigma: float = 0.2,
        magnitude_warp_sigma: float = 0.2
    ):
        self.noise_std = noise_std
        self.jitter_std = jitter_std
        self.scaling_range = scaling_range
        self.time_warp_sigma = time_warp_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.noise_std, data.shape)
        return data + noise
    
    def jitter(self, data: np.ndarray) -> np.ndarray:
        """Add jittering (small random perturbations)"""
        jitter = np.random.normal(0, self.jitter_std, data.shape)
        return data + jitter
    
    def scaling(self, data: np.ndarray) -> np.ndarray:
        """Random scaling"""
        scale_factor = np.random.uniform(*self.scaling_range)
        return data * scale_factor
    
    def time_warping(self, data: np.ndarray) -> np.ndarray:
        """Time warping augmentation"""
        # Simple time warping by interpolation
        original_indices = np.arange(data.shape[0])
        
        # Generate warped time indices
        warp_steps = np.random.normal(0, self.time_warp_sigma, data.shape[0])
        warp_steps = np.cumsum(warp_steps)
        warp_steps = (warp_steps - warp_steps[0]) / (warp_steps[-1] - warp_steps[0])
        warp_steps = warp_steps * (data.shape[0] - 1)
        
        # Interpolate
        warped_data = np.zeros_like(data)
        for i in range(data.shape[-1]):
            if data.ndim == 2:
                warped_data[:, i] = np.interp(original_indices, warp_steps, data[:, i])
            else:
                for j in range(data.shape[1]):
                    warped_data[:, j, i] = np.interp(original_indices, warp_steps, data[:, j, i])
        
        return warped_data
    
    def magnitude_warping(self, data: np.ndarray) -> np.ndarray:
        """Magnitude warping augmentation"""
        # Generate smooth random curve for magnitude warping
        warp_curve = np.random.normal(1.0, self.magnitude_warp_sigma, data.shape[0])
        
        # Smooth the curve
        from scipy.ndimage import gaussian_filter1d
        warp_curve = gaussian_filter1d(warp_curve, sigma=data.shape[0] * 0.05)
        
        # Apply warping
        if data.ndim == 2:
            return data * warp_curve[:, np.newaxis]
        else:
            return data * warp_curve[:, np.newaxis, np.newaxis]
    
    def augment(self, data: np.ndarray, methods: List[str] = None) -> np.ndarray:
        """Apply random augmentation"""
        if methods is None:
            methods = ['noise', 'jitter', 'scaling']
        
        augmented_data = data.copy()
        
        for method in methods:
            if method == 'noise':
                augmented_data = self.add_noise(augmented_data)
            elif method == 'jitter':
                augmented_data = self.jitter(augmented_data)
            elif method == 'scaling':
                augmented_data = self.scaling(augmented_data)
            elif method == 'time_warp':
                augmented_data = self.time_warping(augmented_data)
            elif method == 'magnitude_warp':
                augmented_data = self.magnitude_warping(augmented_data)
        
        return augmented_data


class MissingDataSimulator:
    """Simulate missing data patterns for robustness testing"""
    
    def __init__(self):
        pass
    
    def random_missing(self, data: np.ndarray, missing_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate random missing data"""
        mask = np.random.random(data.shape) > missing_rate
        data_missing = data.copy()
        data_missing[~mask] = np.nan
        return data_missing, mask
    
    def block_missing(self, data: np.ndarray, block_size: int = 10, num_blocks: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate block missing data"""
        mask = np.ones(data.shape, dtype=bool)
        data_missing = data.copy()
        
        for _ in range(num_blocks):
            # Random start position
            start_t = np.random.randint(0, max(1, data.shape[0] - block_size))
            end_t = min(start_t + block_size, data.shape[0])
            
            if data.ndim == 2:
                start_f = np.random.randint(0, data.shape[1])
                mask[start_t:end_t, start_f] = False
                data_missing[start_t:end_t, start_f] = np.nan
            else:
                start_f = np.random.randint(0, data.shape[1])
                start_c = np.random.randint(0, data.shape[2])
                mask[start_t:end_t, start_f, start_c] = False
                data_missing[start_t:end_t, start_f, start_c] = np.nan
        
        return data_missing, mask
    
    def sensor_failure(self, data: np.ndarray, failure_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate sensor failure (entire feature missing)"""
        mask = np.ones(data.shape, dtype=bool)
        data_missing = data.copy()
        
        num_features = data.shape[-1]
        num_failed = int(failure_rate * num_features)
        
        failed_features = np.random.choice(num_features, num_failed, replace=False)
        
        for feature in failed_features:
            if data.ndim == 2:
                mask[:, feature] = False
                data_missing[:, feature] = np.nan
            else:
                mask[:, :, feature] = False
                data_missing[:, :, feature] = np.nan
        
        return data_missing, mask
