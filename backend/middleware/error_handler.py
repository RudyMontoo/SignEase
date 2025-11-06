"""
Error Handling Middleware
========================

Custom error handlers for the SignEase Flask API.
"""

import logging
from flask import jsonify, request
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    """Register custom error handlers with Flask app"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request"""
        logger.warning(f"Bad request from {request.remote_addr}: {error}")
        return jsonify({
            'error': 'Bad Request',
            'message': 'The request could not be understood by the server',
            'status': 400,
            'path': request.path
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found"""
        logger.warning(f"Not found: {request.method} {request.path}")
        return jsonify({
            'error': 'Not Found',
            'message': f'The requested endpoint {request.path} was not found',
            'status': 404,
            'available_endpoints': [
                'GET /',
                'GET /health',
                'POST /predict',
                'GET /metrics'
            ]
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed"""
        logger.warning(f"Method not allowed: {request.method} {request.path}")
        return jsonify({
            'error': 'Method Not Allowed',
            'message': f'The {request.method} method is not allowed for {request.path}',
            'status': 405
        }), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle 413 Request Entity Too Large"""
        logger.warning(f"Request too large from {request.remote_addr}")
        return jsonify({
            'error': 'Request Entity Too Large',
            'message': 'The request payload is too large',
            'status': 413,
            'max_size': '16MB'
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle 429 Too Many Requests"""
        logger.warning(f"Rate limit exceeded from {request.remote_addr}")
        return jsonify({
            'error': 'Too Many Requests',
            'message': 'Rate limit exceeded. Please slow down your requests',
            'status': 429
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error"""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred on the server',
            'status': 500
        }), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable"""
        logger.error(f"Service unavailable: {error}")
        return jsonify({
            'error': 'Service Unavailable',
            'message': 'The service is temporarily unavailable',
            'status': 503
        }), 503
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle generic HTTP exceptions"""
        logger.warning(f"HTTP exception: {error.code} - {error.description}")
        return jsonify({
            'error': error.name,
            'message': error.description,
            'status': error.code
        }), error.code
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected error: {type(error).__name__}: {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status': 500
        }), 500