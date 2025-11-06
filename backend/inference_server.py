#!/usr/bin/env python3
"""
SignEase MVP - Enhanced Inference Server
========================================

Production-ready Flask API server for ASL gesture recognition with
advanced inference engine and comprehensive preprocessing.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import torch
import numpy as np

# Import our enhanced inference engine
from inference.asl_engine import ASLInferenceEngine
from api.prediction_routes import prediction_bp
from performance.optimizer import PerformanceOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:5173'])

# Register blueprints
app.register_blueprint(prediction_bp)

# Global variables
inference_engine = None
performance_optimizer = None
start_time = time.time()



def initialize_inference_engine():
    """Initialize the enhanced ASL inference engine with performance optimization"""
    global inference_engine, performance_optimizer
    
    try:
        # Initialize enhanced inference engine
        inference_engine = ASLInferenceEngine()
        
        # Initialize performance optimizer
        performance_optimizer = PerformanceOptimizer(inference_engine)
        
        # Store in app config for access in routes
        app.config['INFERENCE_ENGINE'] = inference_engine
        app.config['PERFORMANCE_OPTIMIZER'] = performance_optimizer
        
        logger.info("🚀 Enhanced ASL Inference Engine with Performance Optimization initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        return False

@app.before_request
def before_request():
    """Log request details"""
    g.start_time = time.time()

@app.after_request
def after_request(response):
    """Log response details"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"📤 {response.status_code} - {duration*1000:.2f}ms")
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    global inference_engine, start_time
    
    uptime = time.time() - start_time
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    
    if gpu_available:
        try:
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'cached': torch.cuda.memory_reserved() / 1024**2,     # MB
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            }
        except:
            gpu_memory = {'error': 'Could not get GPU memory info'}
    
    # Get inference engine status
    engine_status = {}
    if inference_engine:
        engine_status = inference_engine.get_performance_stats()
        model_info = inference_engine.get_model_info()
        engine_status.update({
            'model_type': model_info.get('model_type', 'Unknown'),
            'total_parameters': model_info.get('total_parameters', 0)
        })
    
    return jsonify({
        'status': 'healthy' if inference_engine else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime,
        'model_loaded': inference_engine is not None,
        'gpu_available': gpu_available,
        'gpu_memory_mb': gpu_memory,
        'device': str(inference_engine.device) if inference_engine else None,
        'version': '2.0.0',
        'inference_engine': engine_status
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get comprehensive API performance metrics with optimization stats"""
    global inference_engine, performance_optimizer, start_time
    
    uptime = time.time() - start_time
    
    if inference_engine and performance_optimizer:
        performance_stats = inference_engine.get_performance_stats()
        model_info = inference_engine.get_model_info()
        optimization_stats = performance_optimizer.get_optimization_stats()
        
        return jsonify({
            'uptime_seconds': uptime,
            'performance': performance_stats,
            'optimization': optimization_stats,
            'model': {
                'type': model_info.get('model_type', 'Unknown'),
                'parameters': model_info.get('total_parameters', 0),
                'classes': len(model_info.get('class_names', []))
            },
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'error': 'Inference engine not available',
            'uptime_seconds': uptime,
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/performance/cleanup', methods=['POST'])
def cleanup_performance():
    """Trigger performance cleanup and memory optimization"""
    global performance_optimizer
    
    if performance_optimizer:
        try:
            performance_optimizer.cleanup_resources()
            return jsonify({
                'status': 'success',
                'message': 'Performance cleanup completed',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    else:
        return jsonify({
            'error': 'Performance optimizer not available'
        }), 503

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with comprehensive API information"""
    global inference_engine
    
    endpoints = {
        'health': '/health',
        'predict': '/predict (POST)',
        'predict_batch': '/predict/batch (POST)',
        'model_info': '/model/info',
        'model_classes': '/model/classes',
        'metrics': '/metrics'
    }
    
    model_info = {}
    if inference_engine:
        model_data = inference_engine.get_model_info()
        model_info = {
            'type': model_data.get('model_type', 'Unknown'),
            'classes': len(model_data.get('class_names', [])),
            'parameters': model_data.get('total_parameters', 0)
        }
    
    return jsonify({
        'service': 'SignEase ASL Recognition API',
        'version': '2.0.0',
        'status': 'running',
        'endpoints': endpoints,
        'model': model_info,
        'features': [
            'GPU acceleration',
            'Landmark preprocessing',
            'Confidence scoring',
            'Batch prediction',
            'Image-based prediction',
            'Performance monitoring'
        ]
    })

def main():
    """Main application entry point"""
    print("🚀 SignEase ASL Recognition API")
    print("=" * 50)
    
    # Initialize enhanced inference engine
    if not initialize_inference_engine():
        print("❌ Failed to initialize inference engine. Exiting.")
        sys.exit(1)
    
    print("✅ API server ready")
    print("📡 Enhanced endpoints available:")
    print("   GET  /health         - Comprehensive health check")
    print("   POST /predict        - Single gesture prediction")
    print("   POST /predict/batch  - Batch gesture prediction")
    print("   GET  /model/info     - Model information")
    print("   GET  /model/classes  - Available gesture classes")
    print("   GET  /metrics        - Performance metrics")
    print("   GET  /              - API information")
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