"""
Real dataset loader for CSV files in the data directory

This module provides a simple interface to load the real time series datasets
that are now available in the data/ folder.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RealTimeSeriesDataset(Dataset):
    """
    Dataset class for real CSV time series data
    
    Loads data from CSV files and creates sequences for forecasting.
    """
    
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        split: str = "train",
        sequence_length: int = 96,
        prediction_length: int = 24,
        target_columns: Optional[List[str]] = None,
        scale_data: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2
    ):
        """
        Initialize real time series dataset
        
        Args:
            data_path: Path to the CSV file
            dataset_name: Name of the dataset
            split: 'train', 'val', or 'test'
            sequence_length: Length of input sequences
            prediction_length: Length of prediction sequences
            target_columns: Columns to predict (if None, auto-detect)
            scale_data: Whether to apply standard scaling
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
        """
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.split = split
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scale_data = scale_data
        
        # Load and preprocess data
        self.data, self.scaler = self._load_and_preprocess_data(
            target_columns, train_ratio, val_ratio
        )
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Loaded {dataset_name} dataset ({split}): {len(self.sequences)} sequences")
    
    def _load_and_preprocess_data(
        self, 
        target_columns: Optional[List[str]], 
        train_ratio: float,
        val_ratio: float
    ) -> Tuple[np.ndarray, Optional[StandardScaler]]:
        """Load and preprocess the dataset"""
        
        # Load CSV file
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {self.dataset_name}: {df.shape}")
        
        # Handle datetime column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        
        # Select numeric columns for modeling
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove time-based feature columns that shouldn't be predicted
        exclude_cols = ['hour', 'day_of_week', 'day_of_year', 'month', 'quarter', 'year', 
                       'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                       'is_weekend', 'season']
        
        if target_columns is None:
            # Use first numeric column that's not a time feature
            target_columns = [col for col in numeric_columns if col not in exclude_cols][:1]
            logger.info(f"Auto-detected target columns: {target_columns}")
        
        # Select available target columns
        available_targets = [col for col in target_columns if col in df.columns]
        if not available_targets:
            # Fallback to first numeric column
            available_targets = [col for col in numeric_columns if col not in exclude_cols][:1]
            logger.warning(f"Target columns not found, using: {available_targets}")
        
        # Extract data
        data = df[available_targets].values.astype(np.float32)
        
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0)
        
        # Split data
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        if self.split == "train":
            data = data[:train_end]
        elif self.split == "val":
            data = data[train_end:val_end]
        else:  # test
            data = data[val_end:]
        
        # Scale data
        scaler = None
        if self.scale_data:
            scaler = StandardScaler()
            if self.split == "train":
                data = scaler.fit_transform(data)
            else:
                # For val/test, fit on train data first
                train_data = df[available_targets].values[:train_end].astype(np.float32)
                train_data = np.nan_to_num(train_data, nan=0.0)
                scaler.fit(train_data)
                data = scaler.transform(data)
        
        logger.info(f"Data shape after preprocessing: {data.shape}")
        return data, scaler
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create input-output sequences for forecasting"""
        sequences = []
        total_length = self.sequence_length + self.prediction_length
        
        for i in range(len(self.data) - total_length + 1):
            # Input sequence
            x = self.data[i:i + self.sequence_length]
            
            # Target sequence (prediction)
            y = self.data[i + self.sequence_length:i + total_length]
            
            # Mask (all observed for real data)
            mask = np.ones_like(x)
            
            sequences.append((x, y, mask))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, mask = self.sequences[idx]
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y), 
            torch.FloatTensor(mask)
        )
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data


def get_real_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    split: str = "train",
    sequence_length: int = 96,
    prediction_length: int = 24,
    **kwargs
) -> RealTimeSeriesDataset:
    """
    Get a real time series dataset by name
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing data files
        split: 'train', 'val', or 'test'
        sequence_length: Input sequence length
        prediction_length: Prediction sequence length
        **kwargs: Additional arguments for dataset
    
    Returns:
        RealTimeSeriesDataset instance
    """
    
    # Dataset configurations
    dataset_configs = {
        "etth1": {
            "file": "ETTh1.csv",
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (hourly)"
        },
        "etth2": {
            "file": "ETTh2.csv", 
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (hourly)"
        },
        "ettm1": {
            "file": "ETTm1.csv",
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (15min)"
        },
        "ettm2": {
            "file": "ETTm2.csv",
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (15min)"
        },
        "ecl": {
            "file": "ECL.csv",
            "target_columns": ["MT_362"],
            "description": "Electricity Consuming Load"
        },
        "gefcom2014": {
            "file": "gefcom2014.csv",
            "target_columns": None,  # Will auto-detect
            "description": "GEFCom2014 Load Forecasting"
        },
        "southern_china": {
            "file": "southern_china.csv", 
            "target_columns": None,  # Will auto-detect
            "description": "Southern China Load Data"
        },
        # Legacy names for compatibility with examples
        "traffic": {"file": "ECL.csv", "target_columns": ["MT_362"], "description": "Traffic (ECL)"},
        "solar": {"file": "gefcom2014.csv", "target_columns": None, "description": "Solar (GEFCom)"},
        "exchange": {"file": "southern_china.csv", "target_columns": None, "description": "Exchange (Southern China)"},
        "weather": {"file": "ETTh1.csv", "target_columns": ["OT"], "description": "Weather (ETT)"},
    }
    
    if dataset_name.lower() not in dataset_configs:
        available = list(dataset_configs.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    config = dataset_configs[dataset_name.lower()]
    data_path = Path(data_dir) / config["file"]
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    return RealTimeSeriesDataset(
        data_path=str(data_path),
        dataset_name=dataset_name,
        split=split,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        target_columns=config["target_columns"],
        **kwargs
    )
