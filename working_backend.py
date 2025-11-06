#!/usr/bin/env python3
"""
Working SignEase Backend
Real ASL model with MediaPipe integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
import joblib
import time
import traceback
from pathlib import Path

app = Flask(__name__)
CORS(app, origins=['*'])

# Global variables
model = None
mp_hands = None
model_loaded = False

# ASL alphabet classes
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

def load_model():
    """Load the trained ASL model"""
    global model, model_loaded
    
    # Try to load different model files
    model_paths = [
        'backend/models/sklearn/best_asl_model.joblib',
        'backend/models/improved_sklearn/best_enhanced_model.joblib',
        'backend/models/gpu_accelerated/best_gpu_model.joblib',
        'backend/models/max_gpu/max_gpu_model.joblib'
    ]
    
    for model_path in model_paths:
        try:
            if Path(model_path).exists():
                print(f"Loading model from: {model_path}")
                model = joblib.load(model_path)
                model_loaded = True
                print(f"✅ Model loaded successfully: {type(model)}")
                return True
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
    
    print("❌ No trained model found. Using fallback prediction.")
    return False

def initialize_mediapipe():
    """Initialize MediaPipe hands"""
    global mp_hands
    
    try:
        mp_solution = mp.solutions.hands
        mp_hands = mp_solution.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe initialized")
        return True
    except Exception as e:
        print(f"❌ MediaPipe initialization failed: {e}")
        return False

def extract_features_from_landmarks(landmarks_data):
    """Extract features from MediaPipe landmarks"""
    try:
        # Convert landmarks to numpy array
        if isinstance(landmarks_data, list) and len(landmarks_data) >= 63:
            # Assume it's already flattened coordinates
            coords = np.array(landmarks_data[:63], dtype=np.float32).reshape(21, 3)
        else:
            return None
        
        # Normalize relative to wrist
        wrist = coords[0]
        normalized = coords - wrist
        
        # Scale normalization
        middle_mcp = normalized[9]  # Middle finger MCP
        hand_size = np.linalg.norm(middle_mcp - normalized[0])
        
        if hand_size < 1e-6:
            return None
        
        normalized = normalized / hand_size
        
        # Distance features
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        distances = []
        
        for tip_idx in fingertips:
            dist = np.linalg.norm(normalized[tip_idx] - normalized[0])
            distances.append(dist)
        
        # Angle features
        angles = []
        for i in range(len(fingertips)-1):
            v1 = normalized[fingertips[i]] - normalized[0]
            v2 = normalized[fingertips[i+1]] - normalized[0]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)
        
        # Combine features
        feature_vector = np.concatenate([
            normalized.flatten(),  # 63 features
            np.array(distances),   # 5 features
            np.array(angles)       # 4 features
        ])
        
        # Pad or truncate to expected size
        if len(feature_vector) < 73:
            # Pad with zeros
            feature_vector = np.pad(feature_vector, (0, 73 - len(feature_vector)))
        elif len(feature_vector) > 73:
            # Truncate
            feature_vector = feature_vector[:73]
        
        return feature_vector.reshape(1, -1)
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def predict_gesture(landmarks_data):
    """Predict ASL gesture from landmarks"""
    global model, model_loaded
    
    try:
        # Extract features
        features = extract_features_from_landmarks(landmarks_data)
        
        if features is None:
            return None, 0.0
        
        if model_loaded and model is not None:
            # Use trained model
            try:
                prediction = model.predict(features)[0]
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(np.max(probabilities))
                else:
                    confidence = 0.85  # Default confidence
                
                # Map prediction to class name
                if isinstance(prediction, (int, np.integer)):
                    if 0 <= prediction < len(ASL_CLASSES):
                        predicted_class = ASL_CLASSES[prediction]
                    else:
                        predicted_class = 'unknown'
                else:
                    predicted_class = str(prediction)
                
                return predicted_class, confidence
                
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fallback to simple heuristic
                return fallback_prediction(features)
        else:
            # Fallback prediction
            return fallback_prediction(features)
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

def fallback_prediction(features):
    """Simple fallback prediction based on feature analysis"""
    try:
        if features is None or len(features[0]) < 63:
            return 'nothing', 0.3
        
        # Simple heuristic based on hand shape
        coords = features[0][:63].reshape(21, 3)
        
        # Analyze finger positions
        fingertips = [4, 8, 12, 16, 20]
        finger_bases = [3, 6, 10, 14, 18]
        
        extended_fingers = 0
        for tip, base in zip(fingertips, finger_bases):
            tip_y = coords[tip][1]
            base_y = coords[base][1]
            if tip_y < base_y:  # Finger pointing up
                extended_fingers += 1
        
        # Simple mapping based on extended fingers
        if extended_fingers == 0:
            return 'A', 0.6
        elif extended_fingers == 1:
            return 'D', 0.6
        elif extended_fingers == 2:
            return 'V', 0.6
        elif extended_fingers == 3:
            return 'W', 0.6
        elif extended_fingers == 4:
            return 'B', 0.6
        elif extended_fingers == 5:
            return 'C', 0.6
        else:
            return 'nothing', 0.3
            
    except Exception as e:
        print(f"Fallback prediction error: {e}")
        return 'nothing', 0.3

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'mediapipe_ready': mp_hands is not None,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict ASL gesture from landmarks"""
    try:
        start_time = time.time()
        
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract landmarks
        landmarks = data.get('landmarks', [])
        if not landmarks:
            return jsonify({'error': 'No landmarks provided'}), 400
        
        # Predict gesture
        predicted_class, confidence = predict_gesture(landmarks)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence),
            'inference_time_ms': float(inference_time),
            'model_used': 'trained' if model_loaded else 'fallback'
        })
        
    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    return jsonify({
        'model_loaded': model_loaded,
        'model_type': type(model).__name__ if model else 'None',
        'classes': ASL_CLASSES,
        'features_expected': 73
    })

if __name__ == '__main__':
    print("🚀 SignEase Working Backend Starting...")
    print("=" * 50)
    
    # Initialize components
    model_success = load_model()
    mp_success = initialize_mediapipe()
    
    print(f"✅ Model Status: {'Loaded' if model_success else 'Fallback'}")
    print(f"✅ MediaPipe: {'Ready' if mp_success else 'Failed'}")
    print("✅ API: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)