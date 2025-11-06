#!/usr/bin/env python3
"""
Data preprocessing pipeline for SignEase MVP
Handles landmark extraction, normalization, and feature engineering
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from PIL import Image
import json
import pickle
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipeLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Hands
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark indices for reference
        self.LANDMARK_NAMES = [
            'WRIST',
            'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
    
    def extract_landmarks(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of shape (21, 3) with x, y, z coordinates or None if no hand detected
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Get first hand landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                return np.array(landmarks, dtype=np.float32)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def extract_landmarks_batch(self, image_paths: List[Path], 
                              show_progress: bool = True) -> List[Optional[np.ndarray]]:
        """
        Extract landmarks from multiple images
        
        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of landmark arrays (or None for failed extractions)
        """
        landmarks_list = []
        
        iterator = tqdm(image_paths, desc="Extracting landmarks") if show_progress else image_paths
        
        for image_path in iterator:
            landmarks = self.extract_landmarks(image_path)
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()

class LandmarkNormalizer:
    """Normalize hand landmarks for consistent feature representation"""
    
    @staticmethod
    def normalize_to_wrist(landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position
        
        Args:
            landmarks: Array of shape (21, 3) with x, y, z coordinates
            
        Returns:
            Normalized landmarks with wrist at origin
        """
        if landmarks is None or landmarks.shape != (21, 3):
            return landmarks
        
        # Wrist is at index 0
        wrist = landmarks[0].copy()
        
        # Translate all landmarks so wrist is at origin
        normalized = landmarks - wrist
        
        return normalized
    
    @staticmethod
    def normalize_by_hand_size(landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks by hand size (distance from wrist to middle finger MCP)
        
        Args:
            landmarks: Array of shape (21, 3) with normalized coordinates
            
        Returns:
            Size-normalized landmarks
        """
        if landmarks is None or landmarks.shape != (21, 3):
            return landmarks
        
        # Calculate hand size (wrist to middle finger MCP)
        wrist = landmarks[0]  # Should be [0, 0, 0] after wrist normalization
        middle_mcp = landmarks[9]  # Middle finger MCP
        
        hand_size = np.linalg.norm(middle_mcp - wrist)
        
        # Avoid division by zero
        if hand_size < 1e-6:
            return landmarks
        
        # Normalize by hand size
        normalized = landmarks / hand_size
        
        return normalized
    
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """
        Apply full normalization pipeline
        
        Args:
            landmarks: Raw landmarks array of shape (21, 3)
            
        Returns:
            Fully normalized landmarks
        """
        if landmarks is None:
            return None
        
        # Step 1: Normalize to wrist
        normalized = LandmarkNormalizer.normalize_to_wrist(landmarks)
        
        # Step 2: Normalize by hand size
        normalized = LandmarkNormalizer.normalize_by_hand_size(normalized)
        
        return normalized

class FeatureExtractor:
    """Extract additional features from normalized landmarks"""
    
    @staticmethod
    def calculate_distances(landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate distances between key landmarks
        
        Args:
            landmarks: Normalized landmarks of shape (21, 3)
            
        Returns:
            Array of distance features
        """
        if landmarks is None or landmarks.shape != (21, 3):
            return np.array([])
        
        distances = []
        
        # Distances from wrist to fingertips
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        
        for tip_idx in fingertips:
            dist = np.linalg.norm(landmarks[tip_idx] - wrist)
            distances.append(dist)
        
        # Distances between adjacent fingertips
        for i in range(len(fingertips) - 1):
            dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[i+1]])
            distances.append(dist)
        
        return np.array(distances, dtype=np.float32)
    
    @staticmethod
    def calculate_angles(landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate angles between finger segments
        
        Args:
            landmarks: Normalized landmarks of shape (21, 3)
            
        Returns:
            Array of angle features
        """
        if landmarks is None or landmarks.shape != (21, 3):
            return np.array([])
        
        angles = []
        
        # Define finger segments (base, middle, tip indices)
        fingers = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for finger in fingers:
            # Calculate angles between consecutive segments
            for i in range(len(finger) - 2):
                p1, p2, p3 = landmarks[finger[i]], landmarks[finger[i+1]], landmarks[finger[i+2]]
                
                # Vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        return np.array(angles, dtype=np.float32)
    
    @staticmethod
    def extract_all_features(landmarks: np.ndarray) -> np.ndarray:
        """
        Extract all features from landmarks
        
        Args:
            landmarks: Normalized landmarks of shape (21, 3)
            
        Returns:
            Combined feature vector
        """
        if landmarks is None:
            return np.array([])
        
        # Flatten landmarks (63 features)
        landmark_features = landmarks.flatten()
        
        # Distance features
        distance_features = FeatureExtractor.calculate_distances(landmarks)
        
        # Angle features
        angle_features = FeatureExtractor.calculate_angles(landmarks)
        
        # Combine all features
        all_features = np.concatenate([
            landmark_features,
            distance_features,
            angle_features
        ])
        
        return all_features

class DataPreprocessor:
    """Main data preprocessing pipeline"""
    
    def __init__(self, dataset_dir: Path, output_dir: Path):
        """
        Initialize data preprocessor
        
        Args:
            dataset_dir: Path to dataset directory
            output_dir: Path to save processed data
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.extractor = MediaPipeLandmarkExtractor()
        self.normalizer = LandmarkNormalizer()
        self.feature_extractor = FeatureExtractor()
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'classes': []
        }
    
    def discover_classes(self) -> Dict[str, int]:
        """
        Discover classes in the dataset
        
        Returns:
            Dictionary mapping class names to indices
        """
        classes = []
        for class_dir in sorted(self.dataset_dir.iterdir()):
            if class_dir.is_dir():
                classes.append(class_dir.name)
        
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.stats['classes'] = classes
        
        logger.info(f"Found {len(classes)} classes: {classes}")
        return class_to_idx
    
    def collect_image_paths(self, class_to_idx: Dict[str, int]) -> Tuple[List[Path], List[int]]:
        """
        Collect all image paths and labels
        
        Args:
            class_to_idx: Class name to index mapping
            
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        for class_name, class_idx in class_to_idx.items():
            class_dir = self.dataset_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Find image files
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            class_images = []
            
            for ext in extensions:
                class_images.extend(list(class_dir.glob(ext)))
            
            # Add to lists
            for img_path in class_images:
                image_paths.append(img_path)
                labels.append(class_idx)
            
            logger.info(f"Class {class_name}: {len(class_images)} images")
        
        self.stats['total_images'] = len(image_paths)
        logger.info(f"Total images collected: {len(image_paths)}")
        
        return image_paths, labels
    
    def process_images(self, image_paths: List[Path], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all images to extract and normalize landmarks
        
        Args:
            image_paths: List of image paths
            labels: List of corresponding labels
            
        Returns:
            Tuple of (features, labels) arrays
        """
        logger.info("Starting landmark extraction...")
        
        # Extract landmarks
        landmarks_list = self.extractor.extract_landmarks_batch(image_paths)
        
        # Process landmarks and extract features
        processed_features = []
        processed_labels = []
        
        logger.info("Processing landmarks and extracting features...")
        
        for i, (landmarks, label) in enumerate(tqdm(zip(landmarks_list, labels), 
                                                   desc="Processing landmarks",
                                                   total=len(landmarks_list))):
            if landmarks is not None:
                # Normalize landmarks
                normalized_landmarks = self.normalizer.normalize_landmarks(landmarks)
                
                if normalized_landmarks is not None:
                    # Extract features
                    features = self.feature_extractor.extract_all_features(normalized_landmarks)
                    
                    if len(features) > 0:
                        processed_features.append(features)
                        processed_labels.append(label)
                        self.stats['successful_extractions'] += 1
                    else:
                        self.stats['failed_extractions'] += 1
                else:
                    self.stats['failed_extractions'] += 1
            else:
                self.stats['failed_extractions'] += 1
        
        logger.info(f"Successfully processed {len(processed_features)} images")
        logger.info(f"Failed to process {self.stats['failed_extractions']} images")
        
        # Convert to numpy arrays
        if processed_features:
            features_array = np.array(processed_features, dtype=np.float32)
            labels_array = np.array(processed_labels, dtype=np.int64)
        else:
            features_array = np.array([], dtype=np.float32)
            labels_array = np.array([], dtype=np.int64)
        
        return features_array, labels_array
    
    def save_processed_data(self, features: np.ndarray, labels: np.ndarray, 
                          class_to_idx: Dict[str, int]):
        """
        Save processed data to disk
        
        Args:
            features: Processed feature array
            labels: Label array
            class_to_idx: Class mapping
        """
        # Save features and labels
        np.save(self.output_dir / 'features.npy', features)
        np.save(self.output_dir / 'labels.npy', labels)
        
        # Save metadata
        metadata = {
            'feature_shape': features.shape,
            'num_classes': len(class_to_idx),
            'class_to_idx': class_to_idx,
            'feature_size': features.shape[1] if len(features.shape) > 1 else 0,
            'stats': self.stats
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save class mapping separately
        with open(self.output_dir / 'class_mapping.pkl', 'wb') as f:
            pickle.dump(class_to_idx, f)
        
        logger.info(f"Processed data saved to {self.output_dir}")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
    
    def run_preprocessing(self):
        """
        Run the complete preprocessing pipeline
        """
        logger.info("Starting data preprocessing pipeline...")
        
        try:
            # Step 1: Discover classes
            class_to_idx = self.discover_classes()
            
            # Step 2: Collect image paths
            image_paths, labels = self.collect_image_paths(class_to_idx)
            
            # Step 3: Process images
            features, labels = self.process_images(image_paths, labels)
            
            # Step 4: Save processed data
            self.save_processed_data(features, labels, class_to_idx)
            
            logger.info("Data preprocessing completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return False
        
        finally:
            # Cleanup
            self.extractor.cleanup()

def main():
    """Main function for testing data preprocessing"""
    print("=== SignEase MVP - Data Preprocessing Pipeline ===\n")
    
    # Configuration
    dataset_dir = Path("data/ASL_Alphabet_Dataset/asl_alphabet_train")  # Use REAL ASL dataset
    output_dir = Path("backend/processed_data")
    
    # Check if dataset exists
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("Please run create_mock_dataset.py first")
        return False
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(dataset_dir, output_dir)
    
    # Run preprocessing
    success = preprocessor.run_preprocessing()
    
    if success:
        print("\n🎉 DATA PREPROCESSING COMPLETE!")
        print("✅ Landmarks extracted and normalized")
        print("✅ Features engineered and saved")
        print("✅ Ready for model training")
    else:
        print("\n❌ Data preprocessing failed")
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)