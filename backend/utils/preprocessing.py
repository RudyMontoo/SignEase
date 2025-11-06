#!/usr/bin/env python3
"""
Preprocessing Utilities
======================

Utility functions for preprocessing ASL gesture data including
landmark normalization, validation, and augmentation.
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class HandLandmarkProcessor:
    """Process and normalize hand landmarks for ASL recognition"""
    
    # MediaPipe hand landmark indices
    LANDMARK_INDICES = {
        'WRIST': 0,
        'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
        'INDEX_FINGER_MCP': 5, 'INDEX_FINGER_PIP': 6, 'INDEX_FINGER_DIP': 7, 'INDEX_FINGER_TIP': 8,
        'MIDDLE_FINGER_MCP': 9, 'MIDDLE_FINGER_PIP': 10, 'MIDDLE_FINGER_DIP': 11, 'MIDDLE_FINGER_TIP': 12,
        'RING_FINGER_MCP': 13, 'RING_FINGER_PIP': 14, 'RING_FINGER_DIP': 15, 'RING_FINGER_TIP': 16,
        'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
    }
    
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray, method: str = 'wrist_relative') -> np.ndarray:
        """
        Normalize hand landmarks using different methods
        
        Args:
            landmarks: Array of shape (63,) representing 21 landmarks with x,y,z coordinates
            method: Normalization method ('wrist_relative', 'hand_size', 'bbox')
        
        Returns:
            Normalized landmarks array
        """
        if landmarks.shape[0] != 63:
            raise ValueError("Expected 63 landmark coordinates (21 points × 3)")
        
        # Reshape to (21, 3) for easier processing
        points = landmarks.reshape(21, 3)
        
        if method == 'wrist_relative':
            return HandLandmarkProcessor._normalize_wrist_relative(points)
        elif method == 'hand_size':
            return HandLandmarkProcessor._normalize_hand_size(points)
        elif method == 'bbox':
            return HandLandmarkProcessor._normalize_bbox(points)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def _normalize_wrist_relative(points: np.ndarray) -> np.ndarray:
        """Normalize relative to wrist position"""
        wrist = points[0].copy()
        normalized = points - wrist
        return normalized.flatten().astype(np.float32)
    
    @staticmethod
    def _normalize_hand_size(points: np.ndarray) -> np.ndarray:
        """Normalize by hand size (wrist to middle finger tip distance)"""
        wrist = points[0]
        middle_tip = points[12]  # Middle finger tip
        
        # Calculate hand size
        hand_size = np.linalg.norm(middle_tip - wrist)
        
        # Normalize relative to wrist and scale by hand size
        if hand_size > 1e-6:
            normalized = (points - wrist) / hand_size
        else:
            normalized = points - wrist
        
        return normalized.flatten().astype(np.float32)
    
    @staticmethod
    def _normalize_bbox(points: np.ndarray) -> np.ndarray:
        """Normalize using bounding box"""
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Calculate center and size
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        
        # Avoid division by zero
        size = np.where(size < 1e-6, 1.0, size)
        
        # Normalize to [-1, 1] range
        normalized = (points - center) / (size / 2)
        
        return normalized.flatten().astype(np.float32)
    
    @staticmethod
    def extract_geometric_features(landmarks: np.ndarray) -> np.ndarray:
        """Extract geometric features from landmarks"""
        points = landmarks.reshape(21, 3)
        features = []
        
        # Finger tip positions relative to wrist
        wrist = points[0]
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip_idx in finger_tips:
            tip_pos = points[tip_idx] - wrist
            features.extend(tip_pos)
        
        # Distances between finger tips
        for i, tip1 in enumerate(finger_tips):
            for tip2 in finger_tips[i+1:]:
                distance = np.linalg.norm(points[tip1] - points[tip2])
                features.append(distance)
        
        # Angles between fingers
        for i in range(len(finger_tips)-1):
            v1 = points[finger_tips[i]] - wrist
            v2 = points[finger_tips[i+1]] - wrist
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            features.append(angle)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def validate_landmarks(landmarks: np.ndarray, strict: bool = True) -> Tuple[bool, str]:
        """
        Validate landmark data quality
        
        Args:
            landmarks: Landmark array to validate
            strict: If True, apply strict validation rules
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check shape
        if landmarks.shape[0] != 63:
            return False, f"Expected 63 coordinates, got {landmarks.shape[0]}"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(landmarks)):
            return False, "Contains NaN values"
        
        if np.any(np.isinf(landmarks)):
            return False, "Contains infinite values"
        
        if strict:
            # Check reasonable coordinate ranges (after normalization)
            if np.any(np.abs(landmarks) > 5):
                return False, "Coordinates outside reasonable range"
            
            # Check if hand landmarks form a reasonable hand shape
            points = landmarks.reshape(21, 3)
            
            # Check if wrist is at origin (for normalized landmarks)
            wrist_distance = np.linalg.norm(points[0])
            if wrist_distance > 0.1:  # Small tolerance
                return False, "Wrist not properly normalized"
        
        return True, "Valid"

class ImagePreprocessor:
    """Preprocess images for ASL recognition"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return padded
    
    @staticmethod
    def normalize_image(image: np.ndarray, method: str = 'imagenet') -> np.ndarray:
        """Normalize image pixel values"""
        image = image.astype(np.float32) / 255.0
        
        if method == 'imagenet':
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        elif method == 'zero_one':
            # Already normalized to [0, 1]
            pass
        elif method == 'minus_one_one':
            # Normalize to [-1, 1]
            image = image * 2.0 - 1.0
        
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
        """Enhance image contrast"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply Gaussian blur for noise reduction"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

class DataAugmentor:
    """Data augmentation for training and inference robustness"""
    
    @staticmethod
    def augment_landmarks(landmarks: np.ndarray, 
                         noise_std: float = 0.01,
                         rotation_angle: float = 0.1,
                         scale_factor: float = 0.1) -> np.ndarray:
        """Apply augmentation to landmarks"""
        points = landmarks.reshape(21, 3)
        
        # Add noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, points.shape)
            points = points + noise
        
        # Apply rotation (around z-axis)
        if rotation_angle > 0:
            angle = np.random.uniform(-rotation_angle, rotation_angle)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            points = points @ rotation_matrix.T
        
        # Apply scaling
        if scale_factor > 0:
            scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
            points = points * scale
        
        return points.flatten().astype(np.float32)
    
    @staticmethod
    def augment_image(image: np.ndarray,
                     brightness_range: Tuple[float, float] = (0.8, 1.2),
                     contrast_range: Tuple[float, float] = (0.8, 1.2),
                     saturation_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply image augmentation"""
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust brightness
        brightness_factor = np.random.uniform(*brightness_range)
        hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
        
        # Adjust saturation
        saturation_factor = np.random.uniform(*saturation_range)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        
        # Convert back to BGR
        hsv = np.clip(hsv, 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Adjust contrast
        contrast_factor = np.random.uniform(*contrast_range)
        augmented = cv2.convertScaleAbs(augmented, alpha=contrast_factor, beta=0)
        
        return augmented

def create_preprocessing_pipeline(config: Dict) -> callable:
    """Create a preprocessing pipeline based on configuration"""
    
    def pipeline(landmarks: Optional[np.ndarray] = None, 
                image: Optional[np.ndarray] = None) -> Dict:
        """Preprocessing pipeline"""
        result = {}
        
        if landmarks is not None:
            # Validate landmarks
            is_valid, error_msg = HandLandmarkProcessor.validate_landmarks(landmarks)
            if not is_valid:
                result['landmarks_error'] = error_msg
                return result
            
            # Normalize landmarks
            normalization_method = config.get('landmark_normalization', 'hand_size')
            normalized_landmarks = HandLandmarkProcessor.normalize_landmarks(
                landmarks, method=normalization_method
            )
            
            # Extract geometric features if requested
            if config.get('extract_geometric_features', False):
                geometric_features = HandLandmarkProcessor.extract_geometric_features(landmarks)
                result['geometric_features'] = geometric_features
            
            result['landmarks'] = normalized_landmarks
        
        if image is not None:
            # Resize image
            target_size = config.get('image_size', (224, 224))
            resized_image = ImagePreprocessor.resize_image(image, target_size)
            
            # Normalize image
            normalization_method = config.get('image_normalization', 'imagenet')
            normalized_image = ImagePreprocessor.normalize_image(
                resized_image, method=normalization_method
            )
            
            result['image'] = normalized_image
        
        return result
    
    return pipeline