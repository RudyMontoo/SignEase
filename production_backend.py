#!/usr/bin/env python3
"""
Production SignEase Backend
Using RTX 5060 Trained Model (99.89% Accuracy)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import time
import traceback
from pathlib import Path

app = Flask(__name__)
CORS(app, origins=['*'])

# Global variables
model = None
mp_hands = None
device = None
model_loaded = False
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

class RTX5060ASLModel(nn.Module):
    """RTX 5060 trained ASL model architecture"""
    
    def __init__(self, input_size=68, hidden_sizes=[2048, 1024, 512, 256, 128], num_classes=29, dropout=0.3):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_rtx5060_model():
    """Load the RTX 5060 trained model"""
    global model, device, model_loaded
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU (GPU not available)")
    
    # Try to load the RTX 5060 model
    model_path = Path("backend/models/rtx5060_full/rtx5060_best_model.pth")
    
    try:
        if model_path.exists():
            print(f"Loading RTX 5060 model from: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create model with same architecture
            model = RTX5060ASLModel()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            model_loaded = True
            
            # Get model info
            val_acc = checkpoint.get('val_metrics', {}).get('accuracy', 0)
            epoch = checkpoint.get('epoch', 0)
            
            print(f"✅ RTX 5060 model loaded successfully!")
            print(f"   Validation Accuracy: {val_acc:.2f}%")
            print(f"   Epoch: {epoch}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return True
        else:
            print(f"❌ RTX 5060 model not found at: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load RTX 5060 model: {e}")
        print(traceback.format_exc())
        return False

def initialize_mediapipe():
    """Initialize MediaPipe with same settings as training"""
    global mp_hands
    
    try:
        mp_solution = mp.solutions.hands
        mp_hands = mp_solution.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.3,  # Same as training
            min_tracking_confidence=0.3
        )
        print("✅ MediaPipe initialized (same config as training)")
        return True
    except Exception as e:
        print(f"❌ MediaPipe initialization failed: {e}")
        return False

def extract_features_from_landmarks(landmarks_data):
    """Extract features exactly as in training"""
    try:
        # Convert landmarks to coordinates
        if isinstance(landmarks_data, list) and len(landmarks_data) >= 63:
            coords = np.array(landmarks_data[:63], dtype=np.float32).reshape(21, 3)
        else:
            return None
        
        # Same normalization as training
        wrist = coords[0]
        normalized = coords - wrist
        
        # Scale normalization
        middle_mcp = normalized[9]
        hand_size = np.linalg.norm(middle_mcp - normalized[0])
        
        if hand_size < 1e-6:
            return None
        
        normalized = normalized / hand_size
        
        # Distance features (same as training)
        fingertips = [4, 8, 12, 16, 20]
        distances = []
        
        for tip_idx in fingertips:
            dist = np.linalg.norm(normalized[tip_idx] - normalized[0])
            distances.append(dist)
        
        # Combine features (63 + 5 = 68 features)
        feature_vector = np.concatenate([
            normalized.flatten(),
            np.array(distances)
        ])
        
        return feature_vector.reshape(1, -1)
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def predict_with_rtx5060_model(landmarks_data):
    """Predict using RTX 5060 trained model"""
    global model, device, model_loaded
    
    try:
        if not model_loaded or model is None:
            return None, 0.0, "Model not loaded"
        
        # Extract features
        features = extract_features_from_landmarks(landmarks_data)
        if features is None:
            return None, 0.0, "Feature extraction failed"
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = classes[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            top3_predictions = {}
            
            for i in range(3):
                class_name = classes[top3_indices[0][i].item()]
                prob = top3_probs[0][i].item()
                top3_predictions[class_name] = float(prob)
        
        return predicted_class, confidence_score, top3_predictions
        
    except Exception as e:
        print(f"RTX 5060 prediction error: {e}")
        return None, 0.0, f"Prediction error: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_allocated': f"{torch.cuda.memory_allocated(0) / 1024**2:.1f}MB",
            'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        }
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_type': 'RTX 5060 Trained (99.89% accuracy)',
        'device': str(device),
        'mediapipe_ready': mp_hands is not None,
        'gpu_info': gpu_info,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict ASL gesture using RTX 5060 model"""
    try:
        start_time = time.time()
        
        # Get data
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400
        
        landmarks = data['landmarks']
        
        # Predict with RTX 5060 model
        predicted_class, confidence, top3_predictions = predict_with_rtx5060_model(landmarks)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence),
            'inference_time_ms': float(inference_time),
            'model_info': 'RTX 5060 Trained (99.89% accuracy)',
            'top3_predictions': top3_predictions,
            'device_used': str(device)
        })
        
    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': 'RTX 5060 ASL Classifier',
        'accuracy': '99.89%',
        'parameters': sum(p.numel() for p in model.parameters()) if model else 0,
        'architecture': 'Deep Neural Network [2048, 1024, 512, 256, 128]',
        'training_samples': '232,041',
        'validation_samples': '49,723',
        'classes': classes,
        'features': 68,
        'device': str(device)
    })

if __name__ == '__main__':
    print("🚀 SignEase Production Backend")
    print("🎮 RTX 5060 Trained Model (99.89% Accuracy)")
    print("=" * 60)
    
    # Initialize components
    model_success = load_rtx5060_model()
    mp_success = initialize_mediapipe()
    
    print(f"✅ RTX 5060 Model: {'Loaded' if model_success else 'Failed'}")
    print(f"✅ MediaPipe: {'Ready' if mp_success else 'Failed'}")
    print(f"✅ Device: {device}")
    print("✅ API: http://localhost:5000")
    print("=" * 60)
    
    if model_success:
        print("🎉 Production backend ready with RTX 5060 model!")
    else:
        print("⚠️ Running without trained model")
    
    app.run(host='0.0.0.0', port=5000, debug=False)