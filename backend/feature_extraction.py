#!/usr/bin/env python3
"""
Feature extraction module for SignEase MVP
Handles landmark normalization, feature engineering, and synthetic data generation
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)

class SyntheticLandmarkGenerator:
    """Generate synthetic hand landmarks for development when MediaPipe fails"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        np.random.seed(seed)
        
        # Base hand template (normalized coordinates)
        self.base_landmarks = np.array([
            [0.0, 0.0, 0.0],      # 0: WRIST
            [-0.1, -0.1, 0.02],   # 1: THUMB_CMC
            [-0.15, -0.2, 0.04],  # 2: THUMB_MCP
            [-0.18, -0.25, 0.06], # 3: THUMB_IP
            [-0.2, -0.3, 0.08],   # 4: THUMB_TIP
            [0.05, -0.15, 0.02],  # 5: INDEX_FINGER_MCP
            [0.08, -0.25, 0.04],  # 6: INDEX_FINGER_PIP
            [0.1, -0.35, 0.06],   # 7: INDEX_FINGER_DIP
            [0.12, -0.45, 0.08],  # 8: INDEX_FINGER_TIP
            [0.0, -0.15, 0.02],   # 9: MIDDLE_FINGER_MCP
            [0.0, -0.25, 0.04],   # 10: MIDDLE_FINGER_PIP
            [0.0, -0.35, 0.06],   # 11: MIDDLE_FINGER_DIP
            [0.0, -0.45, 0.08],   # 12: MIDDLE_FINGER_TIP
            [-0.05, -0.15, 0.02], # 13: RING_FINGER_MCP
            [-0.08, -0.25, 0.04], # 14: RING_FINGER_PIP
            [-0.1, -0.35, 0.06],  # 15: RING_FINGER_DIP
            [-0.12, -0.45, 0.08], # 16: RING_FINGER_TIP
            [-0.1, -0.12, 0.02],  # 17: PINKY_MCP
            [-0.15, -0.2, 0.04],  # 18: PINKY_PIP
            [-0.18, -0.28, 0.06], # 19: PINKY_DIP
            [-0.2, -0.35, 0.08],  # 20: PINKY_TIP
        ], dtype=np.float32)
    
    def generate_gesture_landmarks(self, class_name: str, variation_id: int = 0) -> np.ndarray:
        """
        Generate synthetic landmarks for a specific gesture class
        
        Args:
            class_name: Name of the gesture class (A-Z, space, del, nothing)
            variation_id: ID for creating variations within the same class
            
        Returns:
            Array of shape (21, 3) with synthetic landmarks
        """
        # Start with base landmarks
        landmarks = self.base_landmarks.copy()
        
        # Set random seed based on class and variation for reproducibility
        np.random.seed(hash(class_name + str(variation_id)) % 2**32)
        
        # Apply class-specific modifications
        landmarks = self._apply_gesture_modifications(landmarks, class_name)
        
        # Add random variation
        landmarks = self._add_random_variation(landmarks, variation_id)
        
        return landmarks
    
    def _apply_gesture_modifications(self, landmarks: np.ndarray, class_name: str) -> np.ndarray:
        """Apply gesture-specific modifications to landmarks"""
        
        if class_name in ['A', 'E', 'I', 'O', 'U']:  # Vowels - closed fist variations
            # Bend fingers more
            finger_tips = [4, 8, 12, 16, 20]
            for tip_idx in finger_tips:
                landmarks[tip_idx][1] *= 0.6  # Bring fingertips closer to palm
                landmarks[tip_idx][2] += 0.02  # Slightly forward
        
        elif class_name in ['B', 'C', 'D']:  # Specific hand shapes
            # B - flat hand
            if class_name == 'B':
                finger_tips = [8, 12, 16, 20]  # Not thumb
                for tip_idx in finger_tips:
                    landmarks[tip_idx][1] = -0.4  # Extend fingers
                    landmarks[tip_idx][2] = 0.05
            
            # C - curved hand
            elif class_name == 'C':
                for i in range(5, 21):  # All fingers except thumb
                    landmarks[i][0] *= 1.2  # Spread fingers
                    landmarks[i][1] *= 0.8  # Curve inward
        
        elif class_name in ['L', 'Y']:  # Extended finger gestures
            if class_name == 'L':
                # Extend index finger and thumb
                landmarks[8][1] = -0.5  # Index finger extended
                landmarks[4][0] = -0.3  # Thumb extended
            elif class_name == 'Y':
                # Extend thumb and pinky
                landmarks[4][0] = -0.35  # Thumb extended
                landmarks[20][0] = -0.25  # Pinky extended
        
        elif class_name == 'space':
            # Open hand gesture
            for i in range(1, 21):
                landmarks[i] *= 1.3  # Spread all landmarks
        
        elif class_name == 'del':
            # Pointing gesture
            landmarks[8][1] = -0.5  # Index finger extended
            # Other fingers bent
            for tip_idx in [12, 16, 20]:
                landmarks[tip_idx][1] *= 0.5
        
        elif class_name == 'nothing':
            # Neutral/rest position
            landmarks *= 0.8  # Smaller, relaxed hand
        
        return landmarks
    
    def _add_random_variation(self, landmarks: np.ndarray, variation_id: int) -> np.ndarray:
        """Add random variation to landmarks"""
        # Small random noise for natural variation
        noise_scale = 0.02
        noise = np.random.normal(0, noise_scale, landmarks.shape)
        
        # Scale variation
        scale_variation = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
        
        # Rotation variation (small)
        angle = np.random.uniform(-0.2, 0.2)  # Small rotation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Apply transformations
        landmarks_varied = landmarks * scale_variation + noise
        
        # Apply rotation in x-y plane
        for i in range(len(landmarks_varied)):
            x, y = landmarks_varied[i][0], landmarks_varied[i][1]
            landmarks_varied[i][0] = x * cos_a - y * sin_a
            landmarks_varied[i][1] = x * sin_a + y * cos_a
        
        return landmarks_varied

class AdvancedFeatureExtractor:
    """Advanced feature extraction from normalized landmarks"""
    
    @staticmethod
    def extract_geometric_features(landmarks: np.ndarray) -> np.ndarray:
        """Extract geometric features from landmarks"""
        if landmarks is None or landmarks.shape != (21, 3):
            return np.array([])
        
        features = []
        
        # 1. Fingertip distances from wrist
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip_idx in fingertips:
            dist = np.linalg.norm(landmarks[tip_idx] - wrist)
            features.append(dist)
        
        # 2. Inter-fingertip distances
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                features.append(dist)
        
        # 3. Finger bend angles
        finger_chains = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for chain in finger_chains:
            for i in range(len(chain) - 2):
                p1, p2, p3 = landmarks[chain[i]], landmarks[chain[i+1]], landmarks[chain[i+2]]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                features.append(angle)
        
        # 4. Hand orientation features
        # Vector from wrist to middle finger MCP
        orientation_vector = landmarks[9] - landmarks[0]
        features.extend(orientation_vector)
        
        # 5. Hand span (distance between thumb and pinky tips)
        hand_span = np.linalg.norm(landmarks[4] - landmarks[20])
        features.append(hand_span)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_statistical_features(landmarks: np.ndarray) -> np.ndarray:
        """Extract statistical features from landmarks"""
        if landmarks is None or landmarks.shape != (21, 3):
            return np.array([])
        
        features = []
        
        # Statistical measures for each coordinate
        for coord in range(3):  # x, y, z
            coord_values = landmarks[:, coord]
            
            features.extend([
                np.mean(coord_values),
                np.std(coord_values),
                np.min(coord_values),
                np.max(coord_values),
                np.median(coord_values)
            ])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_all_features(landmarks: np.ndarray) -> np.ndarray:
        """Extract all features from landmarks"""
        if landmarks is None:
            return np.array([])
        
        # Basic landmark features (flattened)
        basic_features = landmarks.flatten()
        
        # Geometric features
        geometric_features = AdvancedFeatureExtractor.extract_geometric_features(landmarks)
        
        # Statistical features
        statistical_features = AdvancedFeatureExtractor.extract_statistical_features(landmarks)
        
        # Combine all features
        all_features = np.concatenate([
            basic_features,
            geometric_features,
            statistical_features
        ])
        
        return all_features

class DataNormalizer:
    """Advanced data normalization and scaling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit scaler and transform features"""
        if len(features) == 0:
            return features
        
        normalized = self.scaler.fit_transform(features)
        self.is_fitted = True
        return normalized
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        if len(features) == 0:
            return features
        
        return self.scaler.transform(features)
    
    def save(self, filepath: Path):
        """Save fitted scaler"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, filepath: Path):
        """Load fitted scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True

def create_synthetic_dataset(output_dir: Path, samples_per_class: int = 200):
    """Create a synthetic dataset with realistic landmarks"""
    print("=== Creating Synthetic Landmark Dataset ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = SyntheticLandmarkGenerator()
    feature_extractor = AdvancedFeatureExtractor()
    
    # Define classes
    classes = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['space', 'del', 'nothing']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Generate synthetic data
    all_features = []
    all_labels = []
    
    print(f"Generating {samples_per_class} samples per class...")
    
    for class_name, class_idx in class_to_idx.items():
        print(f"  Generating {class_name}...")
        
        for i in range(samples_per_class):
            # Generate synthetic landmarks
            landmarks = generator.generate_gesture_landmarks(class_name, i)
            
            # Extract features
            features = feature_extractor.extract_all_features(landmarks)
            
            all_features.append(features)
            all_labels.append(class_idx)
    
    # Convert to numpy arrays
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)
    
    print(f"✅ Generated dataset:")
    print(f"   Features shape: {features_array.shape}")
    print(f"   Labels shape: {labels_array.shape}")
    print(f"   Classes: {len(classes)}")
    
    # Normalize features
    normalizer = DataNormalizer()
    normalized_features = normalizer.fit_transform(features_array)
    
    # Save data
    np.save(output_dir / 'features.npy', normalized_features)
    np.save(output_dir / 'labels.npy', labels_array)
    
    # Save normalizer
    normalizer.save(output_dir / 'normalizer.pkl')
    
    # Save metadata
    metadata = {
        'feature_shape': normalized_features.shape,
        'num_classes': len(classes),
        'class_to_idx': class_to_idx,
        'classes': classes,
        'feature_size': normalized_features.shape[1],
        'samples_per_class': samples_per_class,
        'total_samples': len(normalized_features),
        'data_type': 'synthetic_landmarks'
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Synthetic dataset saved to {output_dir}")
    
    return normalized_features, labels_array, class_to_idx

def main():
    """Test feature extraction and create synthetic dataset"""
    print("=== SignEase MVP - Feature Extraction & Normalization ===\n")
    
    # Create synthetic dataset
    output_dir = Path("backend/processed_data")
    features, labels, class_to_idx = create_synthetic_dataset(output_dir, samples_per_class=100)
    
    print(f"\n🎉 FEATURE EXTRACTION COMPLETE!")
    print("✅ Synthetic landmarks generated")
    print("✅ Advanced features extracted")
    print("✅ Data normalized and scaled")
    print("✅ Ready for model training")
    
    # Test feature extraction on a single sample
    print(f"\n📊 Feature Analysis:")
    print(f"   Feature vector size: {features.shape[1]}")
    print(f"   Total samples: {features.shape[0]}")
    print(f"   Classes: {len(class_to_idx)}")
    print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)