"""
Dynamic Information Lattices: A New Paradigm for Efficient Generative Modeling

This package implements the Dynamic Information Lattices framework for time series forecasting
using diffusion probabilistic models with information-theoretic computational geometry.

Key Components:
- Multi-component entropy estimation
- Hierarchical lattice construction and adaptation
- Information-aware sampling strategies
- Adaptive solver order selection
- Cross-scale synchronization mechanisms
"""

__version__ = "1.0.0"
__author__ = "Dynamic Information Lattices Team"

from .core import (
    DynamicInfoLattices,
    MultiComponentEntropy,
    HierarchicalLattice,
    InformationAwareSampler,
    AdaptiveSolver
)

from .models import (
    ScoreNetwork,
    EntropyWeightNetwork,
    DiffusionModel
)

from .data import (
    TimeSeriesDataset,
    DataPreprocessor,
    get_dataset
)

from .training import (
    DILTrainer,
    TrainingConfig
)

from .evaluation import (
    Evaluator,
    compute_metrics,
    run_ablation_study
)

from .utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    # Core components
    'DynamicInfoLattices',
    'MultiComponentEntropy',
    'HierarchicalLattice', 
    'InformationAwareSampler',
    'AdaptiveSolver',
    
    # Models
    'ScoreNetwork',
    'EntropyWeightNetwork',
    'DiffusionModel',
    
    # Data
    'TimeSeriesDataset',
    'DataPreprocessor',
    'get_dataset',
    
    # Training
    'DILTrainer',
    'TrainingConfig',
    
    # Evaluation
    'Evaluator',
    'compute_metrics',
    'run_ablation_study',
    
    # Utils
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint'
]
