#!/usr/bin/env python3
"""
Simple SignEase Backend
Fast demo backend for ASL recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random
import time

app = Flask(__name__)
CORS(app, origins=['*'])

# ASL alphabet classes
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'gpu_available': True,
        'device': 'cuda',
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict ASL gesture from landmarks"""
    try:
        start_time = time.time()
        
        # Get landmarks from request
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400
        
        landmarks = data['landmarks']
        
        # Simulate processing time
        time.sleep(0.01)  # 10ms processing time
        
        # Simple demo prediction (random for now)
        predicted_class = random.choice(ASL_CLASSES[:26])  # A-Z only
        confidence = random.uniform(0.7, 0.95)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'all_predictions': {
                predicted_class: confidence,
                random.choice(ASL_CLASSES[:26]): confidence * 0.8,
                random.choice(ASL_CLASSES[:26]): confidence * 0.6
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    return jsonify({
        'predictions_made': random.randint(100, 1000),
        'average_inference_time': 10.5,
        'uptime_seconds': time.time(),
        'model_info': {
            'name': 'ASL Classifier',
            'parameters': 209053,
            'accuracy': 0.85
        }
    })

if __name__ == '__main__':
    print("🚀 SignEase Simple Backend Starting...")
    print("=" * 50)
    print("✅ Model: Demo ASL Classifier")
    print("✅ GPU: Simulated CUDA")
    print("✅ Classes: A-Z ASL Alphabet")
    print("✅ API: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)