"""
src/training/trainer.py

Complete training framework for Liquid Neural Networks.
Includes training loop, validation, checkpointing, and logging.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"  # For Mac M1 compatibility

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List
from tqdm import tqdm
import time
import json
from collections import defaultdict


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")


class LNNTrainer:
    """
    Comprehensive trainer for Liquid Neural Networks.
    
    Features:
        - Training and validation loops
        - Automatic checkpointing
        - Learning rate scheduling
        - Gradient clipping
        - Mixed precision training (optional)
        - TensorBoard logging
        - Early stopping
        - Metric tracking
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scheduler: Optional learning rate scheduler
        gradient_clip: Gradient clipping value
        mixed_precision: Whether to use mixed precision training
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip: float = 1.0,
        mixed_precision: bool = False,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        use_tensorboard: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)
        
        # Early stopping
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            log_interval: Logging interval in batches
        
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs, loss, batch_metrics = self._forward_step(batch)
            else:
                outputs, loss, batch_metrics = self._forward_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            metrics['loss'] += loss.item()
            for key, value in batch_metrics.items():
                metrics[key] += value
            
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = metrics['loss'] / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                if self.use_tensorboard:
                    self.writer.add_scalar(
                        'Train/BatchLoss',
                        loss.item(),
                        self.global_step
                    )
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return dict(metrics)
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            
            for batch in pbar:
                batch = self._to_device(batch)
                
                # Forward pass
                outputs, loss, batch_metrics = self._forward_step(batch)
                
                # Update metrics
                metrics['loss'] += loss.item()
                for key, value in batch_metrics.items():
                    metrics[key] += value
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return dict(metrics)
    
    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Single forward step.
        
        Args:
            batch: Dictionary containing batch data
        
        Returns:
            outputs: Model outputs
            loss: Loss value
            metrics: Dictionary of additional metrics
        """
        # Extract inputs and targets
        if 'events' in batch:
            inputs = batch['events']
        elif 'input' in batch:
            inputs = batch['input']
        else:
            raise ValueError("Batch must contain 'events' or 'input' key")
        
        # Forward pass
        if 'timestamps' in batch:
            outputs, _ = self.model(inputs, timestamps=batch['timestamps'])
        else:
            outputs, _ = self.model(inputs)
        
        # Compute loss
        if 'labels' in batch:
            targets = batch['labels']
        elif 'target' in batch:
            targets = batch['target']
        else:
            # For autoencoder-type models
            targets = inputs
        
        loss = self.criterion(outputs, targets)
        
        # Compute additional metrics
        metrics = {}
        if hasattr(self, 'compute_metrics'):
            metrics = self.compute_metrics(outputs, targets)
        
        return outputs, loss, metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_interval: int = 5,
        log_interval: int = 10
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            save_interval: Checkpoint saving interval
            log_interval: Logging interval
        """
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(
                train_loader,
                epoch,
                log_interval
            )
            
            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            
            # Checkpointing
            if epoch % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Save best model
            current_val_loss = val_metrics.get('loss', train_metrics['loss'])
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.save_checkpoint('best_model.pt')
                self.patience_counter = 0
                print(f"✓ New best model saved (val_loss: {current_val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        
        # Save final model and history
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        if self.use_tensorboard:
            self.writer.close()
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics to console and tensorboard."""
        # Console logging
        print(f"\nEpoch {epoch}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        if val_metrics:
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        
        # Log additional metrics
        for key, value in train_metrics.items():
            if key != 'loss':
                print(f"  Train {key}: {value:.4f}")
        for key, value in val_metrics.items():
            if key != 'loss':
                print(f"  Val {key}: {value:.4f}")
        
        # TensorBoard logging
        if self.use_tensorboard:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Log learning rate
            if self.scheduler is not None:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', lr, epoch)
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        path = self.checkpoint_dir / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = defaultdict(list, checkpoint['history'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'history.json'
        
        # Convert to regular dict for JSON serialization
        history_dict = {k: list(v) for k, v in self.history.items()}
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved: {history_path}")


class MetricsTracker:
    """
    Utility class for tracking and computing metrics.
    
    Supports:
        - MSE, MAE, RMSE
        - Classification accuracy, F1
        - Custom metrics
    """
    
    @staticmethod
    def compute_regression_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        mse = nn.MSELoss()(predictions, targets).item()
        mae = nn.L1Loss()(predictions, targets).item()
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    @staticmethod
    def compute_classification_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        pred_labels = predictions.argmax(dim=-1)
        accuracy = (pred_labels == targets).float().mean().item()
        
        # Per-class accuracy
        class_acc = []
        for c in range(num_classes):
            mask = targets == c
            if mask.sum() > 0:
                class_acc.append((pred_labels[mask] == c).float().mean().item())
        
        return {
            'accuracy': accuracy,
            'mean_class_accuracy': np.mean(class_acc) if class_acc else 0.0
        }
    
    @staticmethod
    def compute_tracking_error(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute trajectory tracking error."""
        # Euclidean distance per time step
        errors = torch.sqrt(((predictions - targets) ** 2).sum(dim=-1))
        
        return {
            'mean_tracking_error': errors.mean().item(),
            'max_tracking_error': errors.max().item(),
            'std_tracking_error': errors.std().item()
        }


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: 'adam', 'adamw', 'sgd', 'rmsprop'
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizers[optimizer_name](
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )


def create_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = 'cosine',
    epochs: int = 100,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: 'cosine', 'step', 'plateau', 'exponential'
        epochs: Total number of epochs
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler instance
    """
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            **kwargs
        )
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=epochs // 3,
            **kwargs
        )
    elif scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            **kwargs
        )
    elif scheduler_name == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95,
            **kwargs
        )
    else:
        return None


if __name__ == "__main__":
    print("Testing Training Framework")
    print("=" * 60)
    import sys
    sys.path.insert(0, '../src')
    # Create dummy model and data
    from models.cfc_network import CfCSequenceModel
    from data.event_dataset import create_event_dataloader
    
    model = CfCSequenceModel(
        input_size=32,
        hidden_size=64,
        output_size=10,
        num_layers=2
    )
    
    train_loader = create_event_dataloader(
        'synthetic',
        {
            'num_samples': 100,
            'sequence_length': 10,
            'spatial_size': (64, 64)
        },
        batch_size=8,
        num_workers=0
    )
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = create_optimizer(model, 'adam', lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = LNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_tensorboard=False  # Disable for testing
    )
    
    print("\n✓ Trainer initialized successfully")
    print(f"  Device: {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")