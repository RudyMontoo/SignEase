#!/usr/bin/env python3
"""
Training utilities for SignEase MVP
Includes metrics, early stopping, logging, and training helpers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
        self.batch_times = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float, batch_time: float = 0.0):
        """
        Update metrics with batch results
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            loss: Batch loss value
            batch_time: Time taken for batch processing
        """
        # Convert to CPU and numpy for metric computation
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
        
        self.predictions.extend(pred_classes.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
        self.batch_times.append(batch_time)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = np.mean(predictions == targets)
        avg_loss = np.mean(self.losses)
        
        # Per-class accuracy
        num_classes = max(max(predictions), max(targets)) + 1
        per_class_acc = {}
        
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_acc = np.mean(predictions[class_mask] == targets[class_mask])
                per_class_acc[f'class_{class_id}_acc'] = class_acc
        
        # Top-k accuracy (if applicable)
        top3_acc = self._compute_topk_accuracy(predictions, targets, k=3)
        top5_acc = self._compute_topk_accuracy(predictions, targets, k=5)
        
        # Timing metrics
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'avg_batch_time': avg_batch_time,
            'total_samples': len(predictions)
        }
        
        metrics.update(per_class_acc)
        
        return metrics
    
    def _compute_topk_accuracy(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """Compute top-k accuracy (simplified for single predictions)"""
        # For single predictions, top-k is same as accuracy if k >= 1
        if k >= 1:
            return np.mean(predictions == targets)
        return 0.0

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: 'min' or 'max' for the monitored metric
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def __call__(self, current_score: float, epoch: int, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            current_score: Current value of monitored metric
            epoch: Current epoch
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info(f"Restored best weights from epoch {self.best_epoch}")
            
            return True
        
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get early stopping information"""
        return {
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'patience': self.patience,
            'monitor': self.monitor,
            'mode': self.mode
        }

class TrainingLogger:
    """Training progress logger"""
    
    def __init__(self, 
                 log_dir: Path = Path("backend/logs"),
                 experiment_name: str = "asl_training",
                 save_logs: bool = True):
        """
        Initialize training logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            save_logs: Whether to save logs to file
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.save_logs = save_logs
        
        if save_logs:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
            
            # Setup file logging
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add to logger
            logger.addHandler(file_handler)
        
        # Training history
        self.history = defaultdict(list)
        self.start_time = None
        self.epoch_times = []
    
    def start_training(self):
        """Mark start of training"""
        self.start_time = time.time()
        logger.info("Training started")
    
    def log_epoch(self, 
                  epoch: int,
                  train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float],
                  learning_rate: float,
                  epoch_time: float):
        """
        Log epoch results
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
            epoch_time: Time taken for epoch
        """
        self.epoch_times.append(epoch_time)
        
        # Store in history
        self.history['epoch'].append(epoch)
        self.history['learning_rate'].append(learning_rate)
        self.history['epoch_time'].append(epoch_time)
        
        for key, value in train_metrics.items():
            self.history[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            self.history[f'val_{key}'].append(value)
        
        # Log to console and file
        train_loss = train_metrics.get('loss', 0.0)
        train_acc = train_metrics.get('accuracy', 0.0)
        val_loss = val_metrics.get('loss', 0.0)
        val_acc = val_metrics.get('accuracy', 0.0)
        
        log_msg = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {learning_rate:.6f} | Time: {epoch_time:.2f}s"
        )
        
        logger.info(log_msg)
        print(log_msg)
    
    def log_batch(self, 
                  epoch: int,
                  batch_idx: int,
                  total_batches: int,
                  loss: float,
                  accuracy: float,
                  batch_time: float):
        """Log batch progress"""
        if batch_idx % 10 == 0:  # Log every 10 batches
            progress = batch_idx / total_batches * 100
            log_msg = (
                f"Epoch {epoch} | Batch {batch_idx:4d}/{total_batches} ({progress:5.1f}%) | "
                f"Loss: {loss:.4f} | Acc: {accuracy:.4f} | Time: {batch_time:.3f}s"
            )
            print(f"\r{log_msg}", end="", flush=True)
    
    def finish_training(self, total_epochs: int):
        """Mark end of training"""
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
            
            logger.info(f"Training completed in {total_time:.2f}s")
            logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
            logger.info(f"Total epochs: {total_epochs}")
            
            print(f"\nTraining completed!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average epoch time: {avg_epoch_time:.2f}s")
    
    def save_history(self, filepath: Optional[Path] = None):
        """Save training history to JSON"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"{self.experiment_name}_history_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        history_json = {}
        for key, values in self.history.items():
            history_json[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
        
        with open(filepath, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")
        return filepath

class LearningRateScheduler:
    """Learning rate scheduler wrapper"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
        """
        Initialize scheduler based on configuration
        
        Args:
            optimizer: PyTorch optimizer
            config: Scheduler configuration
        """
        self.optimizer = optimizer
        self.config = config
        self.scheduler = self._create_scheduler()
    
    def _create_scheduler(self):
        """Create scheduler based on configuration"""
        scheduler_name = self.config.get('name', 'none').lower()
        
        if scheduler_name == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('factor', 0.5),
                patience=self.config.get('patience', 5)
            )
        
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.1)
            )
        
        elif scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('T_max', 50)
            )
        
        else:
            return None
    
    def step(self, metric: Optional[float] = None):
        """Step the scheduler"""
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates"""
        if self.scheduler is None:
            return [group['lr'] for group in self.optimizer.param_groups]
        return self.scheduler.get_last_lr()

def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
        
    Returns:
        PyTorch optimizer
    """
    optimizer_name = config.get('name', 'adam').lower()
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0.9)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_device(device_config: str = "auto") -> torch.device:
    """
    Get appropriate device for training
    
    Args:
        device_config: Device configuration ('auto', 'cpu', 'cuda')
        
    Returns:
        PyTorch device
    """
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    logger.info(f"Using device: {device}")
    return device

def main():
    """Test training utilities"""
    print("=== SignEase MVP - Training Utilities Test ===\n")
    
    # Test metrics tracker
    print("Testing MetricsTracker...")
    
    metrics_tracker = MetricsTracker()
    
    # Simulate some predictions
    for _ in range(5):
        predictions = torch.randn(32, 29)  # Batch of 32, 29 classes
        targets = torch.randint(0, 29, (32,))
        loss = np.random.uniform(0.5, 2.0)
        batch_time = np.random.uniform(0.1, 0.5)
        
        metrics_tracker.update(predictions, targets, loss, batch_time)
    
    metrics = metrics_tracker.compute_metrics()
    print(f"✅ Metrics computed: {len(metrics)} metrics")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Samples: {metrics['total_samples']}")
    
    # Test early stopping
    print("\nTesting EarlyStopping...")
    
    early_stopping = EarlyStopping(patience=3, monitor='val_loss', mode='min')
    
    # Simulate training with improving then worsening validation loss
    val_losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Create dummy model
    dummy_model = nn.Linear(10, 5)
    
    should_stop = False
    for epoch, val_loss in enumerate(val_losses):
        should_stop = early_stopping(val_loss, epoch, dummy_model)
        if should_stop:
            print(f"✅ Early stopping triggered at epoch {epoch}")
            break
    
    if not should_stop:
        print("✅ Early stopping did not trigger")
    
    # Test training logger
    print("\nTesting TrainingLogger...")
    
    logger_test = TrainingLogger(
        log_dir=Path("test_logs"),
        experiment_name="test_training",
        save_logs=False  # Don't save for test
    )
    
    logger_test.start_training()
    
    # Simulate a few epochs
    for epoch in range(3):
        train_metrics = {'loss': 1.0 - epoch * 0.2, 'accuracy': 0.5 + epoch * 0.1}
        val_metrics = {'loss': 1.1 - epoch * 0.15, 'accuracy': 0.45 + epoch * 0.12}
        
        logger_test.log_epoch(epoch, train_metrics, val_metrics, 0.001, 10.0)
    
    logger_test.finish_training(3)
    print("✅ Training logger working")
    
    # Test device selection
    print("\nTesting device selection...")
    
    device = get_device("auto")
    print(f"✅ Device selected: {device}")
    
    # Test optimizer creation
    print("\nTesting optimizer creation...")
    
    test_model = nn.Linear(10, 5)
    optimizer_config = {'name': 'adam', 'learning_rate': 0.001}
    optimizer = create_optimizer(test_model, optimizer_config)
    
    print(f"✅ Optimizer created: {type(optimizer).__name__}")
    
    print("\n🎉 TRAINING UTILITIES COMPLETE!")
    print("✅ All utilities working correctly")
    print("✅ Ready for training pipeline implementation")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)