"""
Utility functions for Dynamic Information Lattices
"""

import torch
import numpy as np
import random
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pickle


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(prefer_cuda: bool = True) -> str:
    """Get the best available device"""
    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("Using CPU device")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']}")
    
    return checkpoint


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
):
    """Setup logging configuration"""
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
        'total_mb': size_mb
    }


def save_config(config: Any, filepath: str):
    """Save configuration to file"""
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {filepath}")
    return config


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory with timestamp"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'free_gb': cached - allocated
        }
    else:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / 1024**3,
            'available_gb': memory.available / 1024**3,
            'percent': memory.percent
        }


def print_memory_usage():
    """Print current memory usage"""
    usage = memory_usage()
    if torch.cuda.is_available():
        print(f"GPU Memory - Allocated: {usage['allocated_gb']:.2f} GB, "
              f"Cached: {usage['cached_gb']:.2f} GB, "
              f"Free: {usage['free_gb']:.2f} GB")
    else:
        print(f"CPU Memory - Used: {usage['used_gb']:.2f} GB, "
              f"Available: {usage['available_gb']:.2f} GB, "
              f"Usage: {usage['percent']:.1f}%")


class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available() and hasattr(self.start_time, 'record'):
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.end_time.record()
            torch.cuda.synchronize()
            elapsed = self.start_time.elapsed_time(self.end_time) / 1000  # Convert to seconds
        else:
            import time
            elapsed = time.time() - self.start_time
        
        print(f"{self.name} took {elapsed:.4f} seconds")


def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """Save predictions and targets"""
    data = {
        'predictions': predictions,
        'targets': targets,
        'metadata': metadata or {}
    }
    
    if filepath.endswith('.npz'):
        np.savez_compressed(filepath, **data)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError("Unsupported file format. Use .npz or .pkl")
    
    print(f"Predictions saved to {filepath}")


def load_predictions(filepath: str) -> Dict[str, Any]:
    """Load predictions and targets"""
    if filepath.endswith('.npz'):
        data = np.load(filepath, allow_pickle=True)
        return {
            'predictions': data['predictions'],
            'targets': data['targets'],
            'metadata': data.get('metadata', {})
        }
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .npz or .pkl")


def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def get_git_commit():
    """Get current git commit hash"""
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit
    except:
        return "unknown"


def log_system_info():
    """Log system information"""
    import platform
    import torch
    
    print("=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA: Not available")
    
    print(f"Git commit: {get_git_commit()}")
    print("=" * 30)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
