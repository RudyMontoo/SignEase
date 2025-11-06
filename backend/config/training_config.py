#!/usr/bin/env python3
"""
Training configuration for SignEase MVP
Centralized configuration management for model training
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_size: int = 107
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    num_classes: int = 29
    dropout_rate: float = 0.3
    use_batch_norm: bool = True

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    data_dir: str = "backend/processed_data"
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    augment_train: bool = True
    augmentation_factor: int = 2
    balanced_sampling: bool = False

@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "adam"  # adam, sgd, adamw
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD
    betas: tuple = (0.9, 0.999)  # for Adam/AdamW
    eps: float = 1e-8

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    name: str = "reduce_on_plateau"  # reduce_on_plateau, step, cosine, none
    factor: float = 0.5  # for ReduceLROnPlateau
    patience: int = 5  # for ReduceLROnPlateau
    step_size: int = 10  # for StepLR
    gamma: float = 0.1  # for StepLR
    T_max: int = 50  # for CosineAnnealingLR

@dataclass
class TrainingConfig:
    """Main training configuration"""
    # Training parameters
    num_epochs: int = 100
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = False
    gradient_clip_norm: Optional[float] = 1.0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"  # min, max
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "backend/models/checkpoints"
    checkpoint_frequency: int = 5  # save every N epochs
    save_best_only: bool = False
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    
    # Logging
    log_frequency: int = 10  # log every N batches
    log_dir: str = "backend/logs"
    save_logs: bool = True
    verbose: bool = True
    
    # Validation
    validate_frequency: int = 1  # validate every N epochs
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Experiment tracking
    experiment_name: str = "asl_classifier"
    run_name: Optional[str] = None

@dataclass
class FullConfig:
    """Complete configuration combining all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def save(self, filepath: Path):
        """Save configuration to JSON file"""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'optimizer': self.optimizer.__dict__,
            'scheduler': self.scheduler.__dict__,
            'training': self.training.__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'FullConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            data=DataConfig(**config_dict['data']),
            optimizer=OptimizerConfig(**config_dict['optimizer']),
            scheduler=SchedulerConfig(**config_dict['scheduler']),
            training=TrainingConfig(**config_dict['training'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'optimizer': self.optimizer.__dict__,
            'scheduler': self.scheduler.__dict__,
            'training': self.training.__dict__
        }

# Predefined configurations for different scenarios
def get_quick_config() -> FullConfig:
    """Quick training configuration for testing"""
    config = FullConfig()
    config.training.num_epochs = 10
    config.training.early_stopping_patience = 3
    config.model.hidden_sizes = [128, 64]
    config.data.batch_size = 64
    return config

def get_full_config() -> FullConfig:
    """Full training configuration for production"""
    config = FullConfig()
    config.training.num_epochs = 100
    config.training.early_stopping_patience = 10
    config.model.hidden_sizes = [256, 128, 64]
    config.data.batch_size = 32
    config.optimizer.learning_rate = 0.001
    return config

def get_large_model_config() -> FullConfig:
    """Configuration for large model training"""
    config = FullConfig()
    config.training.num_epochs = 150
    config.training.early_stopping_patience = 15
    config.model.hidden_sizes = [512, 256, 128, 64]
    config.data.batch_size = 16  # Smaller batch for larger model
    config.optimizer.learning_rate = 0.0005
    config.training.gradient_clip_norm = 0.5
    return config

def get_cpu_config() -> FullConfig:
    """Configuration optimized for CPU training"""
    config = FullConfig()
    config.training.device = "cpu"
    config.training.mixed_precision = False
    config.data.pin_memory = False
    config.data.num_workers = 2
    config.data.batch_size = 16
    return config

def get_gpu_config() -> FullConfig:
    """Configuration optimized for GPU training"""
    config = FullConfig()
    config.training.device = "cuda"
    config.training.mixed_precision = True
    config.data.pin_memory = True
    config.data.num_workers = 4
    config.data.batch_size = 64
    return config

# Configuration registry
CONFIG_REGISTRY = {
    'quick': get_quick_config,
    'full': get_full_config,
    'large': get_large_model_config,
    'cpu': get_cpu_config,
    'gpu': get_gpu_config
}

def get_config(config_name: str = 'full') -> FullConfig:
    """
    Get predefined configuration by name
    
    Args:
        config_name: Name of configuration ('quick', 'full', 'large', 'cpu', 'gpu')
        
    Returns:
        FullConfig instance
    """
    if config_name not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return CONFIG_REGISTRY[config_name]()

def main():
    """Test configuration system"""
    print("=== SignEase MVP - Training Configuration Test ===\n")
    
    # Test configuration creation
    print("Testing configuration creation...")
    
    config = get_full_config()
    print(f"✅ Full config created")
    print(f"   Model: {config.model.hidden_sizes}")
    print(f"   Epochs: {config.training.num_epochs}")
    print(f"   Batch size: {config.data.batch_size}")
    print(f"   Learning rate: {config.optimizer.learning_rate}")
    
    # Test different configurations
    configs_to_test = ['quick', 'cpu', 'gpu']
    
    for config_name in configs_to_test:
        test_config = get_config(config_name)
        print(f"✅ {config_name.capitalize()} config: {test_config.training.num_epochs} epochs")
    
    # Test save/load
    print("\nTesting save/load functionality...")
    
    save_path = Path("test_config.json")
    config.save(save_path)
    print(f"✅ Config saved to {save_path}")
    
    loaded_config = FullConfig.load(save_path)
    print(f"✅ Config loaded successfully")
    
    # Verify loaded config
    assert loaded_config.training.num_epochs == config.training.num_epochs
    assert loaded_config.model.hidden_sizes == config.model.hidden_sizes
    print("✅ Loaded config matches original")
    
    # Test to_dict
    config_dict = config.to_dict()
    print(f"✅ Config converted to dict: {len(config_dict)} sections")
    
    # Cleanup
    if save_path.exists():
        save_path.unlink()
    
    print("\n🎉 TRAINING CONFIGURATION SYSTEM COMPLETE!")
    print("✅ Configuration classes working")
    print("✅ Predefined configs available")
    print("✅ Save/load functionality working")
    print("✅ Ready for training pipeline")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)