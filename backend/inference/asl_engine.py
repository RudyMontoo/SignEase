#!/usr/bin/env python3
"""
ASL Inference Engine
===================

Advanced inference engine for ASL gesture recognition with GPU acceleration,
preprocessing, and confidence scoring.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

logger = logging.getLogger(__name__)

class MediaPipeProcessor:
    """MediaPipe hand landmark processor for inference"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand landmarks from image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                # Extract x, y, z coordinates for all 21 landmarks
                coords = []
                for landmark in landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                return np.array(coords, dtype=np.float32)
            
            return None
            
        except Exception as e:
            logger.warning(f"Landmark extraction failed: {e}")
            return None
    
    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()

class LandmarkPreprocessor:
    """Advanced landmark preprocessing for inference"""
    
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks relative to wrist position and hand size"""
        if landmarks.shape[0] != 63:
            raise ValueError("Expected 63 landmark coordinates (21 points × 3)")
        
        # Reshape to (21, 3) for easier processing
        points = landmarks.reshape(21, 3)
        
        # Use wrist (point 0) as reference
        wrist = points[0].copy()
        
        # Translate to wrist origin
        normalized_points = points - wrist
        
        # Calculate hand size (distance from wrist to middle finger tip)
        middle_tip = normalized_points[12]  # Middle finger tip
        hand_size = np.linalg.norm(middle_tip)
        
        # Scale by hand size (avoid division by zero)
        if hand_size > 1e-6:
            normalized_points = normalized_points / hand_size
        
        # Flatten back to 63 coordinates
        return normalized_points.flatten().astype(np.float32)
    
    @staticmethod
    def augment_landmarks(landmarks: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add slight noise for robustness (optional for inference)"""
        noise = np.random.normal(0, noise_level, landmarks.shape)
        return landmarks + noise.astype(np.float32)
    
    @staticmethod
    def validate_landmarks(landmarks: np.ndarray) -> bool:
        """Validate landmark data quality"""
        if landmarks.shape[0] != 63:
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
            return False
        
        # Check if landmarks are within reasonable bounds
        if np.any(np.abs(landmarks) > 10):  # After normalization, should be reasonable
            return False
        
        return True

class ASLInferenceModel(nn.Module):
    """Enhanced ASL model for inference"""
    
    def __init__(self, num_classes: int = 29, dropout_rate: float = 0.3):
        super().__init__()
        
        # Enhanced architecture with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(63, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.4),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class ASLInferenceEngine:
    """Advanced ASL inference engine with comprehensive features"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Class names mapping
        self.class_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        
        # Initialize components
        self.model = None
        self.preprocessor = LandmarkPreprocessor()
        self.mediapipe_processor = MediaPipeProcessor()
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Load model
        self.load_model(model_path)
        
        logger.info(f"ASL Inference Engine initialized on {self.device}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load ASL model with fallback options"""
        try:
            # Create model
            self.model = ASLInferenceModel(num_classes=len(self.class_names)).to(self.device)
            
            if model_path and Path(model_path).exists():
                # Try to load specific model
                self._load_checkpoint(model_path)
            else:
                # Try to find best available model
                models_dir = Path('models')
                if models_dir.exists():
                    model_files = list(models_dir.glob('*.pth'))
                    if model_files:
                        best_model = max(model_files, key=lambda x: x.stat().st_mtime)
                        self._load_checkpoint(str(best_model))
                    else:
                        logger.warning("No model files found, using randomly initialized model")
                else:
                    logger.warning("Models directory not found, using randomly initialized model")
            
            self.model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_checkpoint(self, model_path: str):
        """Load model checkpoint with error handling"""
        try:
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                # Try to load compatible weights
                state_dict = checkpoint['model_state_dict']
                
                # Filter compatible weights for our simplified model
                compatible_weights = {}
                for key, value in state_dict.items():
                    if 'classifier' in key and 'attention' not in key:
                        # Map complex model weights to simple model
                        simple_key = key.replace('classifier.', '')
                        if simple_key in [name for name, _ in self.model.named_parameters()]:
                            compatible_weights[simple_key] = value
                
                # Load compatible weights
                if compatible_weights:
                    self.model.load_state_dict(compatible_weights, strict=False)
                    logger.info(f"✅ Loaded {len(compatible_weights)} compatible weight tensors")
                    
                    if 'accuracy' in checkpoint:
                        logger.info(f"📊 Original model accuracy: {checkpoint['accuracy']:.2f}%")
                else:
                    logger.warning("No compatible weights found, using random initialization")
            else:
                logger.warning("No model_state_dict found in checkpoint")
                
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    
    def preprocess_landmarks(self, landmarks: np.ndarray) -> torch.Tensor:
        """Preprocess landmarks for inference"""
        # Validate input
        if not self.preprocessor.validate_landmarks(landmarks):
            raise ValueError("Invalid landmark data")
        
        # Normalize landmarks
        normalized = self.preprocessor.normalize_landmarks(landmarks)
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).float().to(self.device)
        
        # Add batch dimension if needed
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_from_landmarks(self, landmarks: np.ndarray, 
                             return_alternatives: int = 3) -> Dict:
        """Make prediction from preprocessed landmarks"""
        start_time = time.time()
        
        try:
            # Preprocess landmarks
            input_tensor = self.preprocess_landmarks(landmarks)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top prediction
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Get alternatives
                alternatives = []
                if return_alternatives > 0:
                    top_probs, top_indices = torch.topk(probabilities, 
                                                       min(return_alternatives, len(self.class_names)), 
                                                       dim=1)
                    for i in range(top_probs.shape[1]):
                        alternatives.append({
                            'gesture': self.class_names[top_indices[0][i].item()],
                            'confidence': top_probs[0][i].item()
                        })
                
                # Calculate inference time
                inference_time = time.time() - start_time
                
                # Update metrics
                self.inference_count += 1
                self.total_inference_time += inference_time
                
                return {
                    'gesture': predicted_class,
                    'confidence': confidence_score,
                    'alternatives': alternatives,
                    'inference_time_ms': inference_time * 1000,
                    'preprocessing_success': True
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'gesture': 'error',
                'confidence': 0.0,
                'alternatives': [],
                'inference_time_ms': (time.time() - start_time) * 1000,
                'preprocessing_success': False,
                'error': str(e)
            }
    
    def predict_from_image(self, image: np.ndarray, 
                          return_alternatives: int = 3) -> Dict:
        """Make prediction from image (extract landmarks first)"""
        start_time = time.time()
        
        try:
            # Extract landmarks from image
            landmarks = self.mediapipe_processor.extract_landmarks(image)
            
            if landmarks is None:
                return {
                    'gesture': 'no_hand_detected',
                    'confidence': 0.0,
                    'alternatives': [],
                    'inference_time_ms': (time.time() - start_time) * 1000,
                    'preprocessing_success': False,
                    'error': 'No hand landmarks detected'
                }
            
            # Make prediction from landmarks
            result = self.predict_from_landmarks(landmarks, return_alternatives)
            result['landmark_extraction_success'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Image prediction failed: {e}")
            return {
                'gesture': 'error',
                'confidence': 0.0,
                'alternatives': [],
                'inference_time_ms': (time.time() - start_time) * 1000,
                'preprocessing_success': False,
                'landmark_extraction_success': False,
                'error': str(e)
            }
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        
        return {
            'total_inferences': self.inference_count,
            'average_inference_time_ms': avg_time * 1000,
            'total_inference_time_s': self.total_inference_time,
            'device': str(self.device),
            'model_loaded': self.model is not None
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ASLInferenceModel',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'device': str(self.device)
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'mediapipe_processor'):
            del self.mediapipe_processor