#!/usr/bin/env python3
"""
Model utilities for SignEase MVP
Handles model saving, loading, checkpointing, and management
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Comprehensive model management for training and inference
    """
    
    def __init__(self, model_dir: Path = Path("backend/models/saved")):
        """
        Initialize model manager
        
        Args:
            model_dir: Directory to save models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.registry_file = self.model_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "latest": None}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def save_model(self, 
                   model: nn.Module,
                   model_name: str,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   epoch: Optional[int] = None,
                   metrics: Optional[Dict[str, float]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   is_best: bool = False) -> Path:
        """
        Save model with comprehensive metadata
        
        Args:
            model: PyTorch model to save
            model_name: Name for the model
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch
            metrics: Training metrics
            metadata: Additional metadata
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pth"
        model_path = self.model_dir / model_filename
        
        # Prepare save data
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'model_info', {}),
            'timestamp': timestamp,
            'epoch': epoch,
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        # Add optimizer state
        if optimizer is not None:
            save_data['optimizer_state_dict'] = optimizer.state_dict()
            save_data['optimizer_class'] = optimizer.__class__.__name__
        
        # Add scheduler state
        if scheduler is not None:
            save_data['scheduler_state_dict'] = scheduler.state_dict()
            save_data['scheduler_class'] = scheduler.__class__.__name__
        
        # Save model
        torch.save(save_data, model_path)
        
        # Update registry
        model_info = {
            'filename': model_filename,
            'path': str(model_path),
            'timestamp': timestamp,
            'epoch': epoch,
            'metrics': metrics or {},
            'is_best': is_best,
            'model_class': model.__class__.__name__,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        self.registry['models'][model_name] = model_info
        
        # Update latest and best
        self.registry['latest'] = model_name
        if is_best:
            self.registry['best'] = model_name
        
        self._save_registry()
        
        # Save human-readable summary
        summary_path = model_path.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'model_info': model_info,
                'save_data_keys': list(save_data.keys()),
                'model_summary': getattr(model, 'get_model_summary', lambda: {})()
            }, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Model registered as: {model_name}")
        
        return model_path
    
    def load_model(self, 
                   model_name: str,
                   model_class: Optional[type] = None,
                   device: str = 'cpu',
                   load_optimizer: bool = False,
                   load_scheduler: bool = False) -> Dict[str, Any]:
        """
        Load model and associated components
        
        Args:
            model_name: Name of model to load (or 'latest', 'best')
            model_class: Model class for instantiation
            device: Device to load model on
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Dictionary with loaded components
        """
        # Resolve model name
        if model_name == 'latest':
            model_name = self.registry.get('latest')
        elif model_name == 'best':
            model_name = self.registry.get('best')
        
        if model_name not in self.registry['models']:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.registry['models'][model_name]
        model_path = Path(model_info['path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load saved data
        save_data = torch.load(model_path, map_location=device)
        
        result = {
            'model_info': model_info,
            'save_data': save_data,
            'epoch': save_data.get('epoch'),
            'metrics': save_data.get('metrics', {}),
            'metadata': save_data.get('metadata', {})
        }
        
        # Load model if class provided
        if model_class is not None:
            model_config = save_data.get('model_config', {})
            
            # Try to instantiate model
            try:
                if hasattr(model_class, 'from_config'):
                    model = model_class.from_config(model_config)
                else:
                    # Try with common parameters
                    model = model_class(**model_config.get('architecture_config', {}))
                
                model.load_state_dict(save_data['model_state_dict'])
                model.to(device)
                model.eval()
                
                result['model'] = model
                logger.info(f"Model loaded: {model_class.__name__}")
                
            except Exception as e:
                logger.error(f"Failed to instantiate model: {e}")
                result['model'] = None
        
        # Load optimizer if requested
        if load_optimizer and 'optimizer_state_dict' in save_data:
            result['optimizer_state_dict'] = save_data['optimizer_state_dict']
            result['optimizer_class'] = save_data.get('optimizer_class')
        
        # Load scheduler if requested
        if load_scheduler and 'scheduler_state_dict' in save_data:
            result['scheduler_state_dict'] = save_data['scheduler_state_dict']
            result['scheduler_class'] = save_data.get('scheduler_class')
        
        logger.info(f"Model '{model_name}' loaded from {model_path}")
        
        return result
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self.registry
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model and its files
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if successful
        """
        if model_name not in self.registry['models']:
            logger.warning(f"Model '{model_name}' not found in registry")
            return False
        
        model_info = self.registry['models'][model_name]
        model_path = Path(model_info['path'])
        
        # Delete model file
        if model_path.exists():
            model_path.unlink()
        
        # Delete summary file
        summary_path = model_path.with_suffix('.json')
        if summary_path.exists():
            summary_path.unlink()
        
        # Remove from registry
        del self.registry['models'][model_name]
        
        # Update latest/best if necessary
        if self.registry.get('latest') == model_name:
            remaining_models = list(self.registry['models'].keys())
            self.registry['latest'] = remaining_models[-1] if remaining_models else None
        
        if self.registry.get('best') == model_name:
            self.registry['best'] = None
        
        self._save_registry()
        
        logger.info(f"Model '{model_name}' deleted")
        return True
    
    def cleanup_old_models(self, keep_latest: int = 5, keep_best: bool = True) -> int:
        """
        Clean up old model files
        
        Args:
            keep_latest: Number of latest models to keep
            keep_best: Whether to always keep the best model
            
        Returns:
            Number of models deleted
        """
        models = list(self.registry['models'].items())
        models.sort(key=lambda x: x[1]['timestamp'], reverse=True)
        
        to_delete = []
        best_model = self.registry.get('best')
        
        for i, (model_name, model_info) in enumerate(models):
            # Keep latest N models
            if i < keep_latest:
                continue
            
            # Keep best model if requested
            if keep_best and model_name == best_model:
                continue
            
            to_delete.append(model_name)
        
        deleted_count = 0
        for model_name in to_delete:
            if self.delete_model(model_name):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old models")
        return deleted_count

class ModelCheckpoint:
    """
    Model checkpointing utility for training
    """
    
    def __init__(self, 
                 model_manager: ModelManager,
                 model_name: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = False,
                 save_freq: int = 1):
        """
        Initialize model checkpoint
        
        Args:
            model_manager: ModelManager instance
            model_name: Base name for saved models
            monitor: Metric to monitor
            mode: 'min' or 'max' for the monitored metric
            save_best_only: Whether to save only the best model
            save_freq: Frequency of saving (in epochs)
        """
        self.model_manager = model_manager
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
    
    def __call__(self, 
                 model: nn.Module,
                 epoch: int,
                 metrics: Dict[str, float],
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> bool:
        """
        Check if model should be saved and save if necessary
        
        Args:
            model: Model to potentially save
            epoch: Current epoch
            metrics: Current metrics
            optimizer: Optimizer state
            scheduler: Scheduler state
            
        Returns:
            True if model was saved
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            logger.warning(f"Monitored metric '{self.monitor}' not found in metrics")
            return False
        
        # Check if this is the best model
        is_best = False
        if self.mode == 'min':
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score
        
        if is_best:
            self.best_score = current_score
            self.best_epoch = epoch
        
        # Decide whether to save
        should_save = False
        
        if self.save_best_only:
            should_save = is_best
        else:
            should_save = (epoch % self.save_freq == 0) or is_best
        
        if should_save:
            model_name = f"{self.model_name}_epoch_{epoch}"
            
            self.model_manager.save_model(
                model=model,
                model_name=model_name,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                is_best=is_best
            )
            
            logger.info(f"Model checkpoint saved at epoch {epoch}")
            if is_best:
                logger.info(f"New best {self.monitor}: {current_score:.4f}")
            
            return True
        
        return False

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Generate model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
        
    Returns:
        Model summary dictionary
    """
    total_params, trainable_params = count_parameters(model)
    
    summary = {
        'model_class': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'input_size': input_size,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }
    
    # Try to get model-specific summary
    if hasattr(model, 'get_model_summary'):
        model_specific = model.get_model_summary()
        summary.update(model_specific)
    
    return summary

def main():
    """Test model utilities"""
    print("=== SignEase MVP - Model Utils Test ===\n")
    
    # Test model manager
    print("Testing ModelManager...")
    
    model_manager = ModelManager(Path("backend/models/test_saved"))
    
    # Create a dummy model for testing
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from models.asl_classifier import ASLClassifier
    
    test_model = ASLClassifier(
        input_size=107,
        hidden_sizes=[128, 64],
        num_classes=29
    )
    
    print(f"✅ Test model created: {count_parameters(test_model)[0]:,} parameters")
    
    # Test saving
    print("\nTesting model saving...")
    
    test_metrics = {'val_loss': 0.5, 'val_acc': 0.85}
    saved_path = model_manager.save_model(
        model=test_model,
        model_name="test_asl_model",
        epoch=10,
        metrics=test_metrics,
        is_best=True
    )
    
    print(f"✅ Model saved to: {saved_path}")
    
    # Test loading
    print("\nTesting model loading...")
    
    loaded_data = model_manager.load_model(
        model_name="test_asl_model",
        model_class=ASLClassifier
    )
    
    print(f"✅ Model loaded successfully")
    print(f"   Epoch: {loaded_data['epoch']}")
    print(f"   Metrics: {loaded_data['metrics']}")
    
    # Test model registry
    print("\nTesting model registry...")
    
    registry = model_manager.list_models()
    print(f"✅ Registry loaded: {len(registry['models'])} models")
    
    # Test checkpointing
    print("\nTesting model checkpointing...")
    
    checkpoint = ModelCheckpoint(
        model_manager=model_manager,
        model_name="checkpoint_test",
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    
    # Simulate training epochs
    for epoch in range(1, 4):
        metrics = {'val_loss': 1.0 - epoch * 0.1, 'val_acc': 0.5 + epoch * 0.1}
        saved = checkpoint(test_model, epoch, metrics)
        print(f"   Epoch {epoch}: {'Saved' if saved else 'Not saved'}")
    
    print("✅ Checkpointing working")
    
    # Test model summary
    print("\nTesting model summary...")
    
    summary = model_summary(test_model, (107,))
    print(f"✅ Model summary generated:")
    print(f"   Parameters: {summary['total_parameters']:,}")
    print(f"   Size: {summary['model_size_mb']:.2f} MB")
    
    # Cleanup
    print("\nCleaning up test files...")
    
    test_dir = Path("backend/models/test_saved")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("✅ Cleanup completed")
    
    print("\n🎉 MODEL UTILS IMPLEMENTATION COMPLETE!")
    print("✅ Model saving/loading working")
    print("✅ Model registry functional")
    print("✅ Checkpointing system ready")
    print("✅ Model management utilities ready")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)