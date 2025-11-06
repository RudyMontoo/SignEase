#!/usr/bin/env python3
"""
Main training script for SignEase MVP
Comprehensive training pipeline with GPU acceleration, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import sys
import time
import argparse
import logging
from typing import Dict, Any, Optional, Tuple

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from models.asl_classifier import ASLClassifier
from datasets.asl_dataset import ASLDatasetLoader
from utils.model_utils import ModelManager, ModelCheckpoint
from utils.training_utils import (
    MetricsTracker, EarlyStopping, TrainingLogger, 
    LearningRateScheduler, create_optimizer, get_device
)
from config.training_config import FullConfig, get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLTrainer:
    """
    Main trainer class for ASL gesture recognition
    """
    
    def __init__(self, config: FullConfig):
        """
        Initialize trainer with configuration
        
        Args:
            config: Full training configuration
        """
        self.config = config
        self.device = get_device(config.training.device)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        
        # Training utilities
        self.model_manager = ModelManager(Path(config.training.checkpoint_dir))
        self.logger = TrainingLogger(
            log_dir=Path(config.training.log_dir),
            experiment_name=config.training.experiment_name,
            save_logs=config.training.save_logs
        )
        self.early_stopping = None
        self.checkpoint = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def setup_model(self):
        """Setup model, optimizer, scheduler, and loss function"""
        logger.info("Setting up model...")
        
        # Create model
        self.model = ASLClassifier(
            input_size=self.config.model.input_size,
            hidden_sizes=self.config.model.hidden_sizes,
            num_classes=self.config.model.num_classes,
            dropout_rate=self.config.model.dropout_rate,
            use_batch_norm=self.config.model.use_batch_norm
        )
        
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = create_optimizer(self.model, self.config.optimizer.__dict__)
        
        # Create scheduler
        self.scheduler = LearningRateScheduler(self.optimizer, self.config.scheduler.__dict__)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        if self.config.training.mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
        
        # Early stopping
        if self.config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                min_delta=self.config.training.early_stopping_min_delta,
                monitor=self.config.training.early_stopping_monitor,
                mode=self.config.training.early_stopping_mode
            )
        
        # Model checkpointing
        if self.config.training.save_checkpoints:
            self.checkpoint = ModelCheckpoint(
                model_manager=self.model_manager,
                model_name=self.config.training.experiment_name,
                monitor=self.config.training.monitor_metric,
                mode=self.config.training.monitor_mode,
                save_best_only=self.config.training.save_best_only,
                save_freq=self.config.training.checkpoint_frequency
            )
        
        logger.info(f"Model setup complete: {self.model.count_parameters():,} parameters")
    
    def setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        self.train_loader, self.val_loader, self.test_loader = ASLDatasetLoader.create_data_loaders(
            data_dir=Path(self.config.data.data_dir),
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            device='cpu',  # Keep data on CPU, move to device in training loop
            augment_train=self.config.data.augment_train,
            augmentation_factor=self.config.data.augmentation_factor,
            balanced_sampling=self.config.data.balanced_sampling
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(self.train_loader)} batches")
        logger.info(f"  Val: {len(self.val_loader)} batches")
        logger.info(f"  Test: {len(self.test_loader)} batches")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        epoch_start_time = time.time()
        
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            batch_time = time.time() - batch_start_time
            metrics_tracker.update(outputs, targets, loss.item(), batch_time)
            
            # Log batch progress
            if self.config.training.verbose and batch_idx % self.config.training.log_frequency == 0:
                accuracy = torch.mean((torch.argmax(outputs, dim=1) == targets).float()).item()
                self.logger.log_batch(
                    epoch, batch_idx, len(self.train_loader), 
                    loss.item(), accuracy, batch_time
                )
        
        # Clear line after batch logging
        if self.config.training.verbose:
            print()
        
        return metrics_tracker.compute_metrics()
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                # Move data to device
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                metrics_tracker.update(outputs, targets, loss.item())
        
        return metrics_tracker.compute_metrics()
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        self.logger.start_training()
        
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self.train_epoch(epoch)
                
                # Validation phase
                val_metrics = {}
                if epoch % self.config.training.validate_frequency == 0:
                    val_metrics = self.validate_epoch(epoch)
                
                # Learning rate scheduling
                if self.scheduler.scheduler is not None:
                    if isinstance(self.scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_metrics:
                            self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                    else:
                        self.scheduler.step()
                
                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch results
                self.logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time)
                
                # Model checkpointing
                if self.checkpoint and val_metrics:
                    self.checkpoint(self.model, epoch, val_metrics, self.optimizer, self.scheduler.scheduler)
                
                # Early stopping
                if self.early_stopping and val_metrics:
                    monitor_value = val_metrics.get(self.config.training.early_stopping_monitor)
                    if monitor_value is not None:
                        if self.early_stopping(monitor_value, epoch, self.model):
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            break
                
                # Update current epoch
                self.current_epoch = epoch + 1
                
                # Store history
                self.training_history.append({
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                })
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Finish training
            self.logger.finish_training(self.current_epoch)
            
            # Save training history
            if self.config.training.save_logs:
                self.logger.save_history()
            
            # Save final model
            if self.config.training.save_checkpoints:
                final_metrics = self.training_history[-1]['val_metrics'] if self.training_history else {}
                self.model_manager.save_model(
                    model=self.model,
                    model_name=f"{self.config.training.experiment_name}_final",
                    optimizer=self.optimizer,
                    scheduler=self.scheduler.scheduler,
                    epoch=self.current_epoch,
                    metrics=final_metrics,
                    metadata={'config': self.config.to_dict()}
                )
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on given data loader
        
        Args:
            data_loader: DataLoader to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                metrics_tracker.update(outputs, targets, loss.item())
        
        return metrics_tracker.compute_metrics()
    
    def test(self) -> Dict[str, float]:
        """Test model on test set"""
        logger.info("Testing model...")
        test_metrics = self.evaluate(self.test_loader)
        
        logger.info("Test Results:")
        logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        return test_metrics

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ASL Classifier')
    parser.add_argument('--config', type=str, default='full', 
                       help='Configuration name (quick, full, large, cpu, gpu)')
    parser.add_argument('--config-file', type=str, 
                       help='Path to configuration JSON file')
    parser.add_argument('--resume', type=str, 
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing, no training')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        config = FullConfig.load(Path(args.config_file))
    else:
        config = get_config(args.config)
    
    # Override resume checkpoint if provided
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    
    print("=== SignEase MVP - ASL Classifier Training ===\n")
    print(f"Configuration: {args.config}")
    print(f"Device: {config.training.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    print()
    
    # Create trainer
    trainer = ASLTrainer(config)
    
    # Setup model and data
    trainer.setup_model()
    trainer.setup_data()
    
    if args.test_only:
        # Only run testing
        test_metrics = trainer.test()
        print(f"\nFinal Test Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
    else:
        # Run training
        trainer.train()
        
        # Run final test
        test_metrics = trainer.test()
        print(f"\nFinal Test Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()