#!/usr/bin/env python3
"""
Prediction API Routes
====================

Flask routes for ASL gesture prediction with comprehensive validation
and error handling.
"""

from flask import Blueprint, request, jsonify, current_app
import numpy as np
import cv2
import base64
import logging
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

# Create blueprint
prediction_bp = Blueprint('prediction', __name__)

def validate_landmarks(landmarks) -> tuple[bool, str]:
    """Validate landmark data"""
    if not landmarks:
        return False, "No landmarks provided"
    
    if not isinstance(landmarks, list):
        return False, "Landmarks must be a list"
    
    if len(landmarks) != 63:
        return False, f"Expected 63 landmark coordinates, got {len(landmarks)}"
    
    # Check if all values are numeric
    try:
        landmarks_array = np.array(landmarks, dtype=np.float32)
    except (ValueError, TypeError):
        return False, "All landmark values must be numeric"
    
    # Check for NaN or infinite values
    if np.any(np.isnan(landmarks_array)) or np.any(np.isinf(landmarks_array)):
        return False, "Landmarks contain invalid values (NaN or infinite)"
    
    return True, "Valid"

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image data"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
        
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

@prediction_bp.route('/predict', methods=['POST'])
def predict_gesture():
    """Predict ASL gesture from landmarks or image"""
    start_time = time.time()
    
    try:
        # Get inference engine from app context
        inference_engine = current_app.config.get('INFERENCE_ENGINE')
        if not inference_engine:
            return jsonify({
                'error': 'Inference engine not available',
                'message': 'ASL model is not loaded'
            }), 503
        
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data'
            }), 400
        
        # Check if landmarks are provided
        landmarks = data.get('landmarks')
        image_data = data.get('image')
        
        if not landmarks and not image_data:
            return jsonify({
                'error': 'No input data',
                'message': 'Either landmarks or image data must be provided'
            }), 400
        
        # Prediction result
        result = None
        
        # Predict from landmarks if provided
        if landmarks:
            # Validate landmarks
            is_valid, error_msg = validate_landmarks(landmarks)
            if not is_valid:
                return jsonify({
                    'error': 'Invalid landmarks',
                    'message': error_msg
                }), 400
            
            # Convert to numpy array
            landmarks_array = np.array(landmarks, dtype=np.float32)
            
            # Get number of alternatives
            num_alternatives = data.get('alternatives', 3)
            num_alternatives = max(1, min(num_alternatives, 10))  # Limit between 1-10
            
            # Make prediction
            result = inference_engine.predict_from_landmarks(
                landmarks_array, 
                return_alternatives=num_alternatives
            )
        
        # Predict from image if landmarks not provided
        elif image_data:
            try:
                # Decode image
                image = decode_base64_image(image_data)
                
                # Get number of alternatives
                num_alternatives = data.get('alternatives', 3)
                num_alternatives = max(1, min(num_alternatives, 10))
                
                # Make prediction
                result = inference_engine.predict_from_image(
                    image,
                    return_alternatives=num_alternatives
                )
                
            except ValueError as e:
                return jsonify({
                    'error': 'Invalid image data',
                    'message': str(e)
                }), 400
        
        # Add request metadata
        total_time = time.time() - start_time
        result.update({
            'request_id': getattr(current_app, 'request_counter', 0) + 1,
            'timestamp': time.time(),
            'total_request_time_ms': total_time * 1000,
            'handedness': data.get('handedness', 'unknown')
        })
        
        # Update request counter
        current_app.request_counter = getattr(current_app, 'request_counter', 0) + 1
        
        # Log successful prediction
        logger.info(f"Prediction: {result['gesture']} (confidence: {result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': 'An unexpected error occurred during prediction',
            'details': str(e) if current_app.debug else None
        }), 500

@prediction_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict multiple gestures in batch"""
    start_time = time.time()
    
    try:
        # Get inference engine
        inference_engine = current_app.config.get('INFERENCE_ENGINE')
        if not inference_engine:
            return jsonify({
                'error': 'Inference engine not available'
            }), 503
        
        # Parse request data
        data = request.get_json()
        if not data or 'batch' not in data:
            return jsonify({
                'error': 'No batch data provided',
                'message': 'Request must contain batch array'
            }), 400
        
        batch_data = data['batch']
        if not isinstance(batch_data, list):
            return jsonify({
                'error': 'Invalid batch format',
                'message': 'Batch must be an array'
            }), 400
        
        if len(batch_data) > 50:  # Limit batch size
            return jsonify({
                'error': 'Batch too large',
                'message': 'Maximum batch size is 50'
            }), 400
        
        # Process batch
        results = []
        for i, item in enumerate(batch_data):
            try:
                landmarks = item.get('landmarks')
                if not landmarks:
                    results.append({
                        'index': i,
                        'error': 'No landmarks provided'
                    })
                    continue
                
                # Validate landmarks
                is_valid, error_msg = validate_landmarks(landmarks)
                if not is_valid:
                    results.append({
                        'index': i,
                        'error': error_msg
                    })
                    continue
                
                # Make prediction
                landmarks_array = np.array(landmarks, dtype=np.float32)
                result = inference_engine.predict_from_landmarks(landmarks_array)
                result['index'] = i
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        # Add batch metadata
        total_time = time.time() - start_time
        response = {
            'results': results,
            'batch_size': len(batch_data),
            'successful_predictions': len([r for r in results if 'gesture' in r]),
            'total_batch_time_ms': total_time * 1000,
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@prediction_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        inference_engine = current_app.config.get('INFERENCE_ENGINE')
        if not inference_engine:
            return jsonify({
                'error': 'Inference engine not available'
            }), 503
        
        model_info = inference_engine.get_model_info()
        performance_stats = inference_engine.get_performance_stats()
        
        return jsonify({
            'model': model_info,
            'performance': performance_stats,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({
            'error': 'Could not retrieve model information',
            'message': str(e)
        }), 500

@prediction_bp.route('/model/classes', methods=['GET'])
def get_model_classes():
    """Get available gesture classes"""
    try:
        inference_engine = current_app.config.get('INFERENCE_ENGINE')
        if not inference_engine:
            return jsonify({
                'error': 'Inference engine not available'
            }), 503
        
        return jsonify({
            'classes': inference_engine.class_names,
            'num_classes': len(inference_engine.class_names),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Classes info error: {e}")
        return jsonify({
            'error': 'Could not retrieve class information',
            'message': str(e)
        }), 500