#!/usr/bin/env python3
"""
SignEase Backend Startup Script
==============================

Simple script to start the SignEase backend server locally for development.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'flask-cors', 'torch', 'torchvision', 
        'numpy', 'opencv-python', 'mediapipe', 'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    return True

def check_model():
    """Check if trained model exists"""
    models_dir = Path('models')
    if not models_dir.exists():
        print("⚠️  Models directory not found")
        return False
    
    model_files = list(models_dir.glob('*.pth'))
    if not model_files:
        print("⚠️  No model files found in models/ directory")
        return False
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"✅ Found model: {latest_model}")
    return True

def start_server():
    """Start the backend server"""
    print("🚀 Starting SignEase Backend Server...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model
    check_model()  # Warning only, not blocking
    
    # Start server
    try:
        print("🌐 Server will be available at: http://localhost:5000")
        print("📡 API Documentation: python backend/api_documentation.py")
        print("🛑 Press Ctrl+C to stop the server")
        print()
        
        # Run the inference server
        subprocess.run([sys.executable, 'backend/inference_server.py'])
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()