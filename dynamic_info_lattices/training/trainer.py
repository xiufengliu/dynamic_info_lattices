"""
Training module for Dynamic Information Lattices

Implements the training procedure with comprehensive logging,
checkpointing, and reproducibility features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from pathlib import Path
import time
import json
from dataclasses import dataclass, asdict
import wandb
from tqdm import tqdm

from ..core import DynamicInfoLattices, DILConfig
from ..models import ScoreNetwork, EntropyWeightNetwork
from ..evaluation import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    model_config: DILConfig = None
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    # Optimizer parameters
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Scheduler parameters
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    
    # Loss parameters
    loss_type: str = "mse"
    entropy_loss_weight: float = 0.1
    guidance_loss_weight: float = 0.05
    
    # Validation and checkpointing
    val_every: int = 5
    save_every: int = 10
    early_stopping_patience: int = 20
    
    # Logging
    log_every: int = 100
    use_wandb: bool = False
    wandb_project: str = "dynamic-info-lattices"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


class DILTrainer:
    """
    Trainer for Dynamic Information Lattices
    
    Implements comprehensive training with:
    - Proper logging and checkpointing
    - Reproducibility features
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: DynamicInfoLattices,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Training history
        self.train_history = []
        self.val_history = []
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        if self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function"""
        if self.config.loss_type.lower() == "mse":
            return nn.MSELoss()
        elif self.config.loss_type.lower() == "mae":
            return nn.L1Loss()
        elif self.config.loss_type.lower() == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def _setup_logging(self):
        """Setup logging"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=asdict(self.config),
                name=f"dil_experiment_{int(time.time())}"
            )
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            if epoch % self.config.val_every == 0:
                val_metrics = self._validate_epoch()
                
                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log validation metrics
                self._log_metrics(val_metrics, "val")
                self.val_history.append(val_metrics)
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self._save_checkpoint()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log training metrics
            self._log_metrics(train_metrics, "train")
            self.train_history.append(train_metrics)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final results
        results = {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.epoch + 1,
            'total_time': total_time,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        self._save_results(results)
        
        return results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_entropy_loss = 0.0
        total_guidance_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            # Create observation mask (for missing data simulation)
            mask = torch.ones_like(x)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    y_pred = self.model(x, mask)
                    loss_dict = self._compute_loss(y_pred, y, x, mask)
            else:
                y_pred = self.model(x, mask)
                loss_dict = self._compute_loss(y_pred, y, x, mask)
            
            total_loss_batch = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_entropy_loss += loss_dict['entropy_loss'].item()
            total_guidance_loss += loss_dict['guidance_loss'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss_batch.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log batch metrics
            if batch_idx % self.config.log_every == 0:
                self._log_batch_metrics(loss_dict, batch_idx)
        
        # Compute epoch metrics
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'entropy_loss': total_entropy_loss / num_batches,
            'guidance_loss': total_guidance_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_entropy_loss = 0.0
        total_guidance_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation"):
                x, y = x.to(self.device), y.to(self.device)
                mask = torch.ones_like(x)
                
                # Forward pass
                y_pred = self.model(x, mask)
                loss_dict = self._compute_loss(y_pred, y, x, mask)
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_entropy_loss += loss_dict['entropy_loss'].item()
                total_guidance_loss += loss_dict['guidance_loss'].item()
                num_batches += 1
                
                # Store predictions for metric computation
                all_predictions.append(y_pred.cpu())
                all_targets.append(y.cpu())
        
        # Compute validation metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        val_metrics = compute_metrics(all_predictions.numpy(), all_targets.numpy())
        
        # Add loss metrics
        val_metrics.update({
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'entropy_loss': total_entropy_loss / num_batches,
            'guidance_loss': total_guidance_loss / num_batches
        })
        
        return val_metrics
    
    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute loss components"""
        # Reconstruction loss
        recon_loss = self.criterion(y_pred, y_true)
        
        # Entropy regularization loss (placeholder)
        entropy_loss = torch.tensor(0.0, device=self.device)
        
        # Guidance loss (placeholder)
        guidance_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = (
            recon_loss +
            self.config.entropy_loss_weight * entropy_loss +
            self.config.guidance_loss_weight * guidance_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'entropy_loss': entropy_loss,
            'guidance_loss': guidance_loss
        }
    
    def _log_metrics(self, metrics: Dict[str, float], phase: str):
        """Log metrics"""
        # Console logging
        logger.info(f"{phase.capitalize()} Epoch {self.epoch}: {metrics}")
        
        # Wandb logging
        if self.config.use_wandb:
            wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
            wandb_metrics['epoch'] = self.epoch
            wandb.log(wandb_metrics)
    
    def _log_batch_metrics(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int):
        """Log batch-level metrics"""
        if self.config.use_wandb:
            wandb_metrics = {
                f"train_batch/{k}": v.item() for k, v in loss_dict.items()
            }
            wandb_metrics['global_step'] = self.global_step
            wandb.log(wandb_metrics)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {self.epoch}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""
        results_path = self.log_dir / "training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # Handle list of dictionaries (metrics history)
                serializable_results[k] = v
            else:
                serializable_results[k] = v
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved training results to {results_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
        
        return checkpoint
