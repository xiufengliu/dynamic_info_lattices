"""
Data loader for processed datasets in standardized NPY format
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List, Dict, Any


class ProcessedDataset(Dataset):
    """
    Dataset class for processed time series data in NPY format
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        sequence_length: int = 96,
        prediction_length: int = 24,
        data_dir: str = "data",
        normalize: bool = True
    ):
        """
        Initialize the processed dataset
        
        Args:
            dataset_name: Name of the dataset (e.g., 'electricity', 'traffic', etc.)
            split: Data split ('train', 'test')
            sequence_length: Length of input sequences
            prediction_length: Length of prediction sequences
            data_dir: Directory containing the data files
            normalize: Whether to normalize the data
        """
        self.dataset_name = dataset_name
        self.split = split
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.data_dir = data_dir
        self.normalize = normalize
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load data
        self.data = self._load_data()
        
        # Normalize if requested
        if self.normalize:
            self.scaler = StandardScaler()
            self.data = self._normalize_data()
        else:
            self.scaler = None
        
        # Calculate number of samples
        self.num_samples = max(0, len(self.data) - sequence_length - prediction_length + 1)
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        metadata_path = os.path.join(self.data_dir, "processed", "datasets_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        # Find metadata for this dataset
        for metadata in all_metadata:
            if metadata['name'] == self.dataset_name:
                return metadata
        
        raise ValueError(f"Dataset '{self.dataset_name}' not found in metadata")
    
    def _load_data(self) -> np.ndarray:
        """Load the data from NPY files"""
        if self.split in ['train', 'test']:
            # Load from splits directory
            data_path = os.path.join(self.data_dir, "splits", f"{self.dataset_name}_{self.split}.npy")
        else:
            # Load full dataset
            data_path = os.path.join(self.data_dir, "processed", f"{self.dataset_name}.npy")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = np.load(data_path)
        
        # Handle NaN values
        if np.isnan(data).any():
            print(f"Warning: Found NaN values in {self.dataset_name}, filling with forward/backward fill")
            import pandas as pd
            data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
        
        return data.astype(np.float32)
    
    def _normalize_data(self) -> np.ndarray:
        """Normalize the data using StandardScaler"""
        # Fit scaler on training data shape
        original_shape = self.data.shape
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        if self.split == 'train':
            # Fit and transform for training data
            normalized = self.scaler.fit_transform(data_reshaped)
        else:
            # For test data, we need to load training data to fit scaler
            train_data_path = os.path.join(self.data_dir, "splits", f"{self.dataset_name}_train.npy")
            if os.path.exists(train_data_path):
                train_data = np.load(train_data_path)
                train_reshaped = train_data.reshape(-1, train_data.shape[-1])
                self.scaler.fit(train_reshaped)
                normalized = self.scaler.transform(data_reshaped)
            else:
                # Fallback: fit on current data
                normalized = self.scaler.fit_transform(data_reshaped)
        
        return normalized.reshape(original_shape)
    
    def __len__(self) -> int:
        """Return the number of samples"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples")
        
        # Extract input and target sequences
        start_idx = idx
        end_input_idx = start_idx + self.sequence_length
        end_target_idx = end_input_idx + self.prediction_length
        
        input_seq = self.data[start_idx:end_input_idx]
        target_seq = self.data[end_input_idx:end_target_idx]
        
        # Convert to tensors (keep on CPU to avoid multiprocessing issues)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.float32)
        
        return input_tensor, target_tensor
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            'dataset_name': self.dataset_name,
            'split': self.split,
            'data_shape': self.data.shape,
            'num_series': self.data.shape[1],
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'num_samples': self.num_samples,
            'normalized': self.normalize,
            'metadata': self.metadata
        }
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale"""
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        inverse_data = self.scaler.inverse_transform(data_reshaped)
        return inverse_data.reshape(original_shape)


def get_available_datasets(data_dir: str = "data") -> List[str]:
    """Get list of available processed datasets"""
    metadata_path = os.path.join(data_dir, "processed", "datasets_metadata.json")
    
    if not os.path.exists(metadata_path):
        return []
    
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)
    
    return [metadata['name'] for metadata in all_metadata]


def get_dataset_info(dataset_name: str, data_dir: str = "data") -> Dict[str, Any]:
    """Get information about a specific dataset"""
    metadata_path = os.path.join(data_dir, "processed", "datasets_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)
    
    for metadata in all_metadata:
        if metadata['name'] == dataset_name:
            return metadata
    
    raise ValueError(f"Dataset '{dataset_name}' not found")


def create_dataset(
    dataset_name: str,
    split: str = "train",
    sequence_length: int = 96,
    prediction_length: int = 24,
    data_dir: str = "data",
    normalize: bool = True
) -> ProcessedDataset:
    """
    Convenience function to create a ProcessedDataset
    
    Args:
        dataset_name: Name of the dataset
        split: Data split ('train', 'test')
        sequence_length: Length of input sequences
        prediction_length: Length of prediction sequences
        data_dir: Directory containing the data files
        normalize: Whether to normalize the data
        
    Returns:
        ProcessedDataset instance
    """
    return ProcessedDataset(
        dataset_name=dataset_name,
        split=split,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        data_dir=data_dir,
        normalize=normalize
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Available datasets:")
    datasets = get_available_datasets()
    for dataset in datasets:
        info = get_dataset_info(dataset)
        print(f"  {dataset}: {info['shape']} - {info['description']}")
    
    # Test loading a dataset
    if datasets:
        test_dataset = datasets[0]
        print(f"\nTesting dataset: {test_dataset}")
        
        dataset = create_dataset(test_dataset, split="train", sequence_length=96, prediction_length=24)
        print(f"Dataset info: {dataset.get_data_info()}")
        
        if len(dataset) > 0:
            sample_input, sample_target = dataset[0]
            print(f"Sample input shape: {sample_input.shape}")
            print(f"Sample target shape: {sample_target.shape}")
