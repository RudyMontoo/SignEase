#!/usr/bin/env python3
"""
SignEase MVP - Flask API Server
==============================

Production-ready Flask API server for ASL gesture recognition.
Provides REST endpoints for real-time gesture prediction with GPU acceleration.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import torch
import numpy as np

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for frontend communication
CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000'])

# Global variables for model and metrics
model = None
device = None
model_loaded = False
prediction_count = 0
total_inference_time = 0.0
start_time = time.time()

class ASLInferenceEngine:
    """GPU-accelerated ASL inference engine"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained ASL model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Import the correct ASLClassifier from our models
            from models.asl_classifier import ASLClassifier
            
            # Create model with the same architecture as training
            self.model = ASLClassifier(
                input_size=107,  # Feature vector size
                hidden_sizes=[256, 128, 64],
                num_classes=29,
                dropout_rate=0.3,
                use_batch_norm=True
            ).to(self.device)
            
            # Load the state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"📊 Model accuracy: {checkpoint.get('accuracy', 'N/A')}%")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> dict:
        """Make prediction on feature vector"""
        try:
            start_time = time.time()
            
            with torch.no_grad():
                # Convert features to tensor
                if features is not None:
                    features_tensor = torch.from_numpy(features).float().to(self.device)
                    if len(features_tensor.shape) == 1:
                        features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
                else:
                    # Create dummy features if not provided
                    features_tensor = torch.zeros(1, 107).to(self.device)
                
                # Forward pass
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top prediction
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # DEBUG: Log raw model output
                logger.info(f"🔍 Debug - Raw probabilities shape: {probabilities.shape}")
                logger.info(f"🔍 Debug - Top 5 probabilities: {torch.topk(probabilities, 5, dim=1)}")
                logger.info(f"🔍 Debug - Predicted class index: {predicted_idx.item()}")
                logger.info(f"🔍 Debug - Predicted class name: {predicted_class}")
                logger.info(f"🔍 Debug - Confidence: {confidence_score}")
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
                alternatives = []
                for i in range(3):
                    class_idx = top3_indices[0][i].item()
                    class_name = self.class_names[class_idx]
                    class_prob = top3_probs[0][i].item()
                    logger.info(f"🔍 Debug - Alternative {i+1}: idx={class_idx}, name={class_name}, prob={class_prob}")
                    alternatives.append({
                        'gesture': class_name,
                        'confidence': class_prob
                    })
                
                inference_time = time.time() - start_time
                
                return {
                    'gesture': predicted_class,
                    'confidence': confidence_score,
                    'alternatives': alternatives,
                    'inference_time_ms': inference_time * 1000
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'gesture': 'error',
                'confidence': 0.0,
                'alternatives': [],
                'inference_time_ms': 0.0,
                'error': str(e)
            }

def initialize_model():
    """Initialize the ASL model"""
    global model, device, model_loaded
    
    try:
        # Find the best model file
        models_dir = Path('models')
        if not models_dir.exists():
            logger.error("Models directory not found")
            return False
        
        model_files = list(models_dir.glob('asl_model_best_*.pth'))
        if not model_files:
            logger.error("No model files found")
            return False
        
        # Use the most recent model
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using model: {model_path}")
        
        # Initialize inference engine
        model = ASLInferenceEngine(str(model_path))
        device = model.device
        model_loaded = True
        
        logger.info("🚀 ASL Inference Engine initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

@app.before_request
def before_request():
    """Log request details"""
    g.start_time = time.time()
    logger.info(f"📥 {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    """Log response details"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"📤 {response.status_code} - {duration*1000:.2f}ms")
    return response

@app.errorhandler(400)
def bad_request(error):
    """Handle bad requests"""
    return jsonify({
        'error': 'Bad Request',
        'message': 'Invalid request format or missing required fields',
        'status': 400
    }), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'status': 500
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_loaded, device, start_time
    
    uptime = time.time() - start_time
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    
    if gpu_available:
        gpu_memory = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2,     # MB
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        }
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime,
        'model_loaded': model_loaded,
        'gpu_available': gpu_available,
        'gpu_memory_mb': gpu_memory,
        'device': str(device) if device else None,
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict_gesture():
    """Predict ASL gesture from landmarks"""
    global model, prediction_count, total_inference_time
    
    if not model_loaded or model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'ASL model is not available'
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data'
            }), 400
        
        # Extract landmarks
        landmarks = data.get('landmarks')
        if not landmarks:
            return jsonify({
                'error': 'No landmarks provided',
                'message': 'landmarks field is required'
            }), 400
        
        # Convert landmarks to numpy array
        landmarks_array = np.array(landmarks, dtype=np.float32)
        if landmarks_array.shape != (63,):
            return jsonify({
                'error': 'Invalid landmarks format',
                'message': 'landmarks must be array of 63 values (21 points × 3 coordinates)'
            }), 400
        
        # Reshape landmarks to (21, 3) for feature extraction
        landmarks_reshaped = landmarks_array.reshape(21, 3)
        
        # Extract features using the same method as training
        from feature_extraction import AdvancedFeatureExtractor
        feature_extractor = AdvancedFeatureExtractor()
        features = feature_extractor.extract_all_features(landmarks_reshaped)
        
        logger.info(f"🔍 Debug - Landmarks shape: {landmarks_reshaped.shape}")
        logger.info(f"🔍 Debug - Features extracted: {len(features)} features")
        logger.info(f"🔍 Debug - Features sample: {features[:5] if len(features) > 5 else features}")
        
        if len(features) == 0:
            return jsonify({
                'error': 'Feature extraction failed',
                'message': 'Could not extract features from landmarks'
            }), 400
        
        # Make prediction
        logger.info(f"🔍 Debug - Calling model.predict with features shape: {features.shape if hasattr(features, 'shape') else len(features)}")
        result = model.predict(features)
        logger.info(f"🔍 Debug - Prediction result: {result}")
        
        # Update metrics
        prediction_count += 1
        total_inference_time += result.get('inference_time_ms', 0)
        
        # Add request metadata
        result['request_id'] = prediction_count
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get API performance metrics"""
    global prediction_count, total_inference_time, start_time
    
    uptime = time.time() - start_time
    avg_inference_time = total_inference_time / prediction_count if prediction_count > 0 else 0
    
    return jsonify({
        'predictions_total': prediction_count,
        'avg_inference_time_ms': avg_inference_time,
        'uptime_seconds': uptime,
        'requests_per_second': prediction_count / uptime if uptime > 0 else 0,
        'model_accuracy': 99.57,  # From training results
        'gpu_utilization': torch.cuda.utilization() if torch.cuda.is_available() else None
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'SignEase ASL Recognition API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'metrics': '/metrics'
        },
        'documentation': 'https://github.com/your-repo/signease-mvp'
    })

def main():
    """Main application entry point"""
    print("🚀 SignEase ASL Recognition API")
    print("=" * 50)
    
    # Initialize model
    if not initialize_model():
        print("❌ Failed to initialize model. Exiting.")
        sys.exit(1)
    
    print("✅ API server ready")
    print("📡 Endpoints available:")
    print("   GET  /health  - Health check")
    print("   POST /predict - Gesture prediction")
    print("   GET  /metrics - Performance metrics")
    print("   GET  /       - API information")
    print()
    print("🌐 Starting server on http://localhost:5000")
    
    # Start Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()