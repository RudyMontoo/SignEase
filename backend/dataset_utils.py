#!/usr/bin/env python3
"""
Dataset utilities for SignEase MVP
Includes data augmentation, train/val/test splits, and dataset management
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import pickle
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LandmarkAugmenter:
    """Data augmentation for hand landmarks"""
    
    def __init__(self, 
                 rotation_range: float = 0.3,
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.02,
                 translation_range: float = 0.1):
        """
        Initialize augmenter with parameters
        
        Args:
            rotation_range: Maximum rotation angle in radians
            scale_range: Min and max scale factors
            noise_std: Standard deviation for Gaussian noise
            translation_range: Maximum translation distance
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.translation_range = translation_range
    
    def rotate_landmarks(self, landmarks: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Rotate landmarks around the wrist (z-axis rotation)
        
        Args:
            landmarks: Array of shape (21, 3)
            angle: Rotation angle in radians (random if None)
            
        Returns:
            Rotated landmarks
        """
        if landmarks.shape != (21, 3):
            return landmarks
        
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix for z-axis rotation
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation
        rotated = landmarks @ rotation_matrix.T
        
        return rotated.astype(np.float32)
    
    def scale_landmarks(self, landmarks: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """
        Scale landmarks uniformly
        
        Args:
            landmarks: Array of shape (21, 3)
            scale: Scale factor (random if None)
            
        Returns:
            Scaled landmarks
        """
        if landmarks.shape != (21, 3):
            return landmarks
        
        if scale is None:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        scaled = landmarks * scale
        
        return scaled.astype(np.float32)
    
    def add_noise(self, landmarks: np.ndarray, noise_std: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to landmarks
        
        Args:
            landmarks: Array of shape (21, 3)
            noise_std: Noise standard deviation (default if None)
            
        Returns:
            Noisy landmarks
        """
        if landmarks.shape != (21, 3):
            return landmarks
        
        if noise_std is None:
            noise_std = self.noise_std
        
        noise = np.random.normal(0, noise_std, landmarks.shape)
        noisy = landmarks + noise
        
        return noisy.astype(np.float32)
    
    def translate_landmarks(self, landmarks: np.ndarray, 
                          translation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Translate landmarks
        
        Args:
            landmarks: Array of shape (21, 3)
            translation: Translation vector (random if None)
            
        Returns:
            Translated landmarks
        """
        if landmarks.shape != (21, 3):
            return landmarks
        
        if translation is None:
            translation = np.random.uniform(
                -self.translation_range, 
                self.translation_range, 
                size=3
            )
        
        translated = landmarks + translation
        
        return translated.astype(np.float32)
    
    def augment_landmarks(self, landmarks: np.ndarray, 
                         apply_rotation: bool = True,
                         apply_scaling: bool = True,
                         apply_noise: bool = True,
                         apply_translation: bool = True) -> np.ndarray:
        """
        Apply random augmentations to landmarks
        
        Args:
            landmarks: Array of shape (21, 3)
            apply_rotation: Whether to apply rotation
            apply_scaling: Whether to apply scaling
            apply_noise: Whether to apply noise
            apply_translation: Whether to apply translation
            
        Returns:
            Augmented landmarks
        """
        augmented = landmarks.copy()
        
        if apply_rotation:
            augmented = self.rotate_landmarks(augmented)
        
        if apply_scaling:
            augmented = self.scale_landmarks(augmented)
        
        if apply_translation:
            augmented = self.translate_landmarks(augmented)
        
        if apply_noise:
            augmented = self.add_noise(augmented)
        
        return augmented

class ASLLandmarkDataset(Dataset):
    """PyTorch Dataset for ASL landmarks with augmentation"""
    
    def __init__(self, 
                 features: np.ndarray,
                 labels: np.ndarray,
                 augment: bool = False,
                 augmentation_factor: int = 2):
        """
        Initialize dataset
        
        Args:
            features: Feature array of shape (N, feature_dim)
            labels: Label array of shape (N,)
            augment: Whether to apply data augmentation
            augmentation_factor: How many augmented samples per original
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        
        if augment:
            self.augmenter = LandmarkAugmenter()
            # Expand dataset size with augmentation
            self.effective_size = len(self.features) * (1 + augmentation_factor)
        else:
            self.effective_size = len(self.features)
    
    def __len__(self):
        return self.effective_size
    
    def __getitem__(self, idx):
        if not self.augment:
            return self.features[idx], self.labels[idx]
        
        # Determine if this is an original or augmented sample
        original_size = len(self.features)
        
        if idx < original_size:
            # Original sample
            return self.features[idx], self.labels[idx]
        else:
            # Augmented sample
            original_idx = (idx - original_size) % original_size
            
            # Get original feature and label
            original_feature = self.features[original_idx].numpy()
            label = self.labels[original_idx]
            
            # Extract landmarks from features (first 63 elements are flattened landmarks)
            landmarks = original_feature[:63].reshape(21, 3)
            
            # Apply augmentation
            augmented_landmarks = self.augmenter.augment_landmarks(landmarks)
            
            # Reconstruct feature vector (replace first 63 elements)
            augmented_feature = original_feature.copy()
            augmented_feature[:63] = augmented_landmarks.flatten()
            
            return torch.tensor(augmented_feature, dtype=torch.float32), label

class DataSplitter:
    """Handle train/validation/test splits"""
    
    @staticmethod
    def create_splits(features: np.ndarray, 
                     labels: np.ndarray,
                     train_size: float = 0.7,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     random_state: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        Create train/validation/test splits
        
        Args:
            features: Feature array
            labels: Label array
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, val_data, test_data) where each is (features, labels)
        """
        # Validate split sizes
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(features)*100:.1f}%)")
        logger.info(f"  Val: {len(X_val)} samples ({len(X_val)/len(features)*100:.1f}%)")
        logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/len(features)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    @staticmethod
    def save_splits(train_data: Tuple[np.ndarray, np.ndarray],
                   val_data: Tuple[np.ndarray, np.ndarray],
                   test_data: Tuple[np.ndarray, np.ndarray],
                   output_dir: Path):
        """Save data splits to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save train data
        np.save(output_dir / 'train_features.npy', train_data[0])
        np.save(output_dir / 'train_labels.npy', train_data[1])
        
        # Save validation data
        np.save(output_dir / 'val_features.npy', val_data[0])
        np.save(output_dir / 'val_labels.npy', val_data[1])
        
        # Save test data
        np.save(output_dir / 'test_features.npy', test_data[0])
        np.save(output_dir / 'test_labels.npy', test_data[1])
        
        logger.info(f"Data splits saved to {output_dir}")
    
    @staticmethod
    def load_splits(data_dir: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """Load data splits from disk"""
        data_dir = Path(data_dir)
        
        # Load train data
        train_features = np.load(data_dir / 'train_features.npy')
        train_labels = np.load(data_dir / 'train_labels.npy')
        
        # Load validation data
        val_features = np.load(data_dir / 'val_features.npy')
        val_labels = np.load(data_dir / 'val_labels.npy')
        
        # Load test data
        test_features = np.load(data_dir / 'test_features.npy')
        test_labels = np.load(data_dir / 'test_labels.npy')
        
        return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)

def create_data_loaders(train_data: Tuple[np.ndarray, np.ndarray],
                       val_data: Tuple[np.ndarray, np.ndarray],
                       test_data: Tuple[np.ndarray, np.ndarray],
                       batch_size: int = 32,
                       augment_train: bool = True,
                       augmentation_factor: int = 2,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders
    
    Args:
        train_data: Training data tuple (features, labels)
        val_data: Validation data tuple (features, labels)
        test_data: Test data tuple (features, labels)
        batch_size: Batch size for data loaders
        augment_train: Whether to augment training data
        augmentation_factor: Augmentation factor for training data
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ASLLandmarkDataset(
        train_data[0], train_data[1], 
        augment=augment_train, 
        augmentation_factor=augmentation_factor
    )
    
    val_dataset = ASLLandmarkDataset(
        val_data[0], val_data[1], 
        augment=False
    )
    
    test_dataset = ASLLandmarkDataset(
        test_data[0], test_data[1], 
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def main():
    """Test data augmentation and splitting"""
    print("=== SignEase MVP - Data Augmentation & Splitting ===\n")
    
    # Load processed data
    data_dir = Path("backend/processed_data")
    
    if not (data_dir / 'features.npy').exists():
        print("❌ Processed data not found. Run feature_extraction.py first.")
        return False
    
    # Load data
    features = np.load(data_dir / 'features.npy')
    labels = np.load(data_dir / 'labels.npy')
    
    print(f"✅ Loaded data:")
    print(f"   Features: {features.shape}")
    print(f"   Labels: {labels.shape}")
    
    # Create splits
    train_data, val_data, test_data = DataSplitter.create_splits(features, labels)
    
    # Save splits
    DataSplitter.save_splits(train_data, val_data, test_data, data_dir)
    
    # Create data loaders with augmentation
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        batch_size=32,
        augment_train=True,
        augmentation_factor=2
    )
    
    # Test data loading
    print(f"\n=== Testing Data Loading ===")
    
    # Test training loader (with augmentation)
    train_batch = next(iter(train_loader))
    print(f"✅ Train batch: {train_batch[0].shape}, {train_batch[1].shape}")
    
    # Test validation loader
    val_batch = next(iter(val_loader))
    print(f"✅ Val batch: {val_batch[0].shape}, {val_batch[1].shape}")
    
    # Test augmentation
    print(f"\n=== Testing Augmentation ===")
    augmenter = LandmarkAugmenter()
    
    # Create test landmarks
    test_landmarks = np.random.randn(21, 3).astype(np.float32)
    
    # Apply different augmentations
    rotated = augmenter.rotate_landmarks(test_landmarks, angle=0.2)
    scaled = augmenter.scale_landmarks(test_landmarks, scale=1.1)
    noisy = augmenter.add_noise(test_landmarks, noise_std=0.01)
    
    print(f"✅ Original landmarks shape: {test_landmarks.shape}")
    print(f"✅ Rotated landmarks shape: {rotated.shape}")
    print(f"✅ Scaled landmarks shape: {scaled.shape}")
    print(f"✅ Noisy landmarks shape: {noisy.shape}")
    
    print(f"\n🎉 DATA AUGMENTATION & SPLITTING COMPLETE!")
    print("✅ Data splits created and saved")
    print("✅ Data augmentation pipeline working")
    print("✅ DataLoaders with augmentation ready")
    print("✅ Ready for model training")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)