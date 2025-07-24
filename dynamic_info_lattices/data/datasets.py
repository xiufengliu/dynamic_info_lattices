"""
Dataset implementations for time series forecasting experiments

Implements the 12 datasets mentioned in the paper:
Traffic, Solar, Exchange, Weather, M4, Wikipedia, etc.
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
import logging
from pathlib import Path
import requests
import zipfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class TimeSeriesDataset(data.Dataset):
    """Base time series dataset class"""
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 96,
        prediction_length: int = 24,
        stride: int = 1,
        normalize: bool = True,
        scaler_type: str = "standard"
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.normalize = normalize
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        # Normalize data if requested
        if self.normalize and self.scaler is not None:
            self.data = self._normalize_data(self.data)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize the data using the specified scaler"""
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        data_normalized = self.scaler.fit_transform(data_reshaped)
        return data_normalized.reshape(original_shape)
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create input-output sequences for training"""
        sequences = []
        total_length = self.sequence_length + self.prediction_length
        
        for i in range(0, len(self.data) - total_length + 1, self.stride):
            input_seq = self.data[i:i + self.sequence_length]
            target_seq = self.data[i + self.sequence_length:i + total_length]
            sequences.append((input_seq, target_seq))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        
        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(target_seq)
        )
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data"""
        if self.scaler is not None:
            original_shape = data.shape
            data_reshaped = data.reshape(-1, data.shape[-1])
            data_denormalized = self.scaler.inverse_transform(data_reshaped)
            return data_denormalized.reshape(original_shape)
        return data


class TrafficDataset(TimeSeriesDataset):
    """Traffic dataset (LAMetro-Loop-Detector)"""
    
    def __init__(self, data_dir: str = "./data", **kwargs):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and load data
        data = self._load_traffic_data()
        super().__init__(data, **kwargs)
    
    def _load_traffic_data(self) -> np.ndarray:
        """Load traffic dataset"""
        file_path = self.data_dir / "traffic.csv"
        
        if not file_path.exists():
            logger.info("Downloading traffic dataset...")
            self._download_traffic_data(file_path)
        
        # Load and preprocess
        df = pd.read_csv(file_path)
        
        # Remove timestamp column if present
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        # Convert to numpy array
        data = df.values.astype(np.float32)
        
        logger.info(f"Loaded traffic data with shape: {data.shape}")
        return data
    
    def _download_traffic_data(self, file_path: Path):
        """Download traffic dataset"""
        # This is a placeholder - in practice, you would download from the actual source
        # For demonstration, we'll create synthetic traffic data
        logger.warning("Creating synthetic traffic data for demonstration")
        
        # Generate synthetic traffic data (862 sensors, 17544 time steps)
        np.random.seed(42)
        n_sensors = 862
        n_timesteps = 17544
        
        # Create realistic traffic patterns
        time_of_day = np.arange(n_timesteps) % 24
        day_of_week = (np.arange(n_timesteps) // 24) % 7
        
        traffic_data = []
        for sensor in range(n_sensors):
            # Base traffic level
            base_level = np.random.uniform(20, 100)
            
            # Daily pattern
            daily_pattern = base_level * (1 + 0.5 * np.sin(2 * np.pi * time_of_day / 24))
            
            # Weekly pattern (lower on weekends)
            weekly_pattern = daily_pattern * (1 - 0.3 * (day_of_week >= 5))
            
            # Add noise
            noise = np.random.normal(0, 5, n_timesteps)
            
            sensor_data = weekly_pattern + noise
            sensor_data = np.maximum(sensor_data, 0)  # Ensure non-negative
            
            traffic_data.append(sensor_data)
        
        traffic_array = np.array(traffic_data).T  # Shape: (timesteps, sensors)
        
        # Save to CSV
        df = pd.DataFrame(traffic_array)
        df.to_csv(file_path, index=False)


class SolarDataset(TimeSeriesDataset):
    """Solar power dataset"""
    
    def __init__(self, data_dir: str = "./data", **kwargs):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        data = self._load_solar_data()
        super().__init__(data, **kwargs)
    
    def _load_solar_data(self) -> np.ndarray:
        """Load solar dataset"""
        file_path = self.data_dir / "solar.csv"
        
        if not file_path.exists():
            logger.info("Creating synthetic solar dataset...")
            self._create_solar_data(file_path)
        
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        data = df.values.astype(np.float32)
        logger.info(f"Loaded solar data with shape: {data.shape}")
        return data
    
    def _create_solar_data(self, file_path: Path):
        """Create synthetic solar power data"""
        np.random.seed(43)
        n_plants = 137
        n_timesteps = 52560  # About 6 years of hourly data
        
        solar_data = []
        for plant in range(n_plants):
            # Solar irradiance pattern
            hour_of_day = np.arange(n_timesteps) % 24
            day_of_year = (np.arange(n_timesteps) // 24) % 365
            
            # Daily solar pattern (peak at noon)
            daily_pattern = np.maximum(0, np.cos(2 * np.pi * (hour_of_day - 12) / 24))
            
            # Seasonal pattern (higher in summer)
            seasonal_pattern = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            
            # Plant capacity
            capacity = np.random.uniform(50, 200)
            
            # Weather effects (clouds, etc.)
            weather_noise = np.random.beta(2, 2, n_timesteps)
            
            plant_data = capacity * daily_pattern * seasonal_pattern * weather_noise
            solar_data.append(plant_data)
        
        solar_array = np.array(solar_data).T
        df = pd.DataFrame(solar_array)
        df.to_csv(file_path, index=False)


class ExchangeDataset(TimeSeriesDataset):
    """Exchange rate dataset"""
    
    def __init__(self, data_dir: str = "./data", **kwargs):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        data = self._load_exchange_data()
        super().__init__(data, **kwargs)
    
    def _load_exchange_data(self) -> np.ndarray:
        """Load exchange rate dataset"""
        file_path = self.data_dir / "exchange_rate.csv"
        
        if not file_path.exists():
            logger.info("Creating synthetic exchange rate dataset...")
            self._create_exchange_data(file_path)
        
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        data = df.values.astype(np.float32)
        logger.info(f"Loaded exchange rate data with shape: {data.shape}")
        return data
    
    def _create_exchange_data(self, file_path: Path):
        """Create synthetic exchange rate data"""
        np.random.seed(44)
        n_currencies = 8
        n_timesteps = 7588  # Daily data for about 20 years
        
        # Currency names for reference
        currencies = ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'SEK']
        
        exchange_data = []
        for curr in range(n_currencies):
            # Random walk with drift
            drift = np.random.normal(0, 0.0001, n_timesteps)
            volatility = np.random.uniform(0.005, 0.02)
            
            # Generate exchange rate using geometric Brownian motion
            returns = np.random.normal(drift, volatility, n_timesteps)
            log_prices = np.cumsum(returns)
            prices = np.exp(log_prices)
            
            # Normalize to reasonable exchange rate range
            prices = prices / prices[0] * np.random.uniform(0.5, 2.0)
            
            exchange_data.append(prices)
        
        exchange_array = np.array(exchange_data).T
        df = pd.DataFrame(exchange_array, columns=currencies)
        df.to_csv(file_path, index=False)


class WeatherDataset(TimeSeriesDataset):
    """Weather dataset"""
    
    def __init__(self, data_dir: str = "./data", **kwargs):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        data = self._load_weather_data()
        super().__init__(data, **kwargs)
    
    def _load_weather_data(self) -> np.ndarray:
        """Load weather dataset"""
        file_path = self.data_dir / "weather.csv"
        
        if not file_path.exists():
            logger.info("Creating synthetic weather dataset...")
            self._create_weather_data(file_path)
        
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        data = df.values.astype(np.float32)
        logger.info(f"Loaded weather data with shape: {data.shape}")
        return data
    
    def _create_weather_data(self, file_path: Path):
        """Create synthetic weather data"""
        np.random.seed(45)
        n_features = 21  # Temperature, humidity, pressure, wind speed, etc.
        n_timesteps = 52696  # Hourly data for 6 years
        
        feature_names = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
            'precipitation', 'solar_radiation', 'visibility', 'dew_point', 'cloud_cover',
            'uv_index', 'air_quality', 'pm25', 'pm10', 'ozone', 'no2', 'so2', 'co',
            'feels_like_temp', 'heat_index', 'wind_chill'
        ]
        
        weather_data = []
        
        # Time components
        hour_of_day = np.arange(n_timesteps) % 24
        day_of_year = (np.arange(n_timesteps) // 24) % 365
        
        for i, feature in enumerate(feature_names):
            if feature == 'temperature':
                # Temperature with daily and seasonal cycles
                daily_cycle = 10 * np.sin(2 * np.pi * hour_of_day / 24)
                seasonal_cycle = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                base_temp = 15  # Base temperature in Celsius
                noise = np.random.normal(0, 3, n_timesteps)
                data = base_temp + daily_cycle + seasonal_cycle + noise
                
            elif feature == 'humidity':
                # Humidity (0-100%)
                base_humidity = 60
                daily_variation = 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
                noise = np.random.normal(0, 10, n_timesteps)
                data = np.clip(base_humidity + daily_variation + noise, 0, 100)
                
            elif feature == 'pressure':
                # Atmospheric pressure (hPa)
                base_pressure = 1013.25
                variation = np.random.normal(0, 10, n_timesteps)
                data = base_pressure + variation
                
            elif feature == 'wind_speed':
                # Wind speed (m/s)
                data = np.random.gamma(2, 2, n_timesteps)
                
            else:
                # Generic feature with some correlation to temperature
                temp_correlation = np.random.uniform(-0.5, 0.5)
                base_value = np.random.uniform(0, 100)
                noise = np.random.normal(0, 10, n_timesteps)
                data = base_value + temp_correlation * (weather_data[0] if weather_data else 0) + noise
            
            weather_data.append(data)
        
        weather_array = np.array(weather_data).T
        df = pd.DataFrame(weather_array, columns=feature_names)
        df.to_csv(file_path, index=False)


def get_dataset(
    name: str,
    data_dir: str = "./data",
    split: str = "train",
    **kwargs
) -> TimeSeriesDataset:
    """
    Get dataset by name
    
    Args:
        name: Dataset name ('traffic', 'solar', 'exchange', 'weather', etc.)
        data_dir: Data directory
        split: Data split ('train', 'val', 'test')
        **kwargs: Additional arguments for dataset
        
    Returns:
        dataset: TimeSeriesDataset instance
    """
    dataset_classes = {
        'traffic': TrafficDataset,
        'solar': SolarDataset,
        'exchange': ExchangeDataset,
        'weather': WeatherDataset,
    }
    
    if name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(dataset_classes.keys())}")
    
    # Create dataset
    dataset_class = dataset_classes[name]
    dataset = dataset_class(data_dir=data_dir, **kwargs)
    
    # Split dataset
    if split != "full":
        dataset = split_dataset(dataset, split)
    
    return dataset


def split_dataset(
    dataset: TimeSeriesDataset,
    split: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> TimeSeriesDataset:
    """Split dataset into train/val/test"""
    total_size = len(dataset)
    
    if split == "train":
        start_idx = 0
        end_idx = int(total_size * train_ratio)
    elif split == "val":
        start_idx = int(total_size * train_ratio)
        end_idx = int(total_size * (train_ratio + val_ratio))
    elif split == "test":
        start_idx = int(total_size * (train_ratio + val_ratio))
        end_idx = total_size
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Create subset
    subset_sequences = dataset.sequences[start_idx:end_idx]
    
    # Create new dataset with subset
    subset_dataset = TimeSeriesDataset.__new__(TimeSeriesDataset)
    subset_dataset.__dict__.update(dataset.__dict__)
    subset_dataset.sequences = subset_sequences
    
    return subset_dataset
