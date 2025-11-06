#!/usr/bin/env python3
"""
Live Debug Dashboard
Real-time debugging of SignEase system
"""

import requests
import time
import json
import os
import threading
from datetime import datetime

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_backend_status():
    """Get backend status"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            return response.json(), True
        else:
            return {"error": f"HTTP {response.status_code}"}, False
    except Exception as e:
        return {"error": str(e)}, False

def get_model_info():
    """Get model information"""
    try:
        response = requests.get("http://localhost:5000/model-info", timeout=2)
        if response.status_code == 200:
            return response.json(), True
        else:
            return {"error": f"HTTP {response.status_code}"}, False
    except Exception as e:
        return {"error": str(e)}, False

def test_prediction():
    """Test prediction with sample data"""
    test_landmarks = [0.5, 0.5, 0.0] + [0.4 + i*0.01 for i in range(60)]
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:5000/predict",
            json={"landmarks": test_landmarks},
            timeout=5
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            data['response_time'] = (end_time - start_time) * 1000
            return data, True
        else:
            return {"error": f"HTTP {response.status_code}"}, False
    except Exception as e:
        return {"error": str(e)}, False

def print_dashboard():
    """Print the live dashboard"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("🚀 SignEase Live Debug Dashboard")
    print("=" * 80)
    print(f"⏰ Time: {timestamp}")
    print()
    
    # Backend Status
    print("🔧 BACKEND STATUS")
    print("-" * 40)
    backend_data, backend_ok = get_backend_status()
    
    if backend_ok:
        print(f"✅ Status: {backend_data['status']}")
        print(f"🤖 Model: {backend_data['model_type']}")
        print(f"🎮 Device: {backend_data['device']}")
        print(f"📊 Model Loaded: {backend_data['model_loaded']}")
        
        if 'gpu_info' in backend_data and backend_data['gpu_info']:
            gpu = backend_data['gpu_info']
            print(f"💾 GPU: {gpu['gpu_name']}")
            print(f"🔥 GPU Memory: {gpu['gpu_memory_allocated']} / {gpu['gpu_memory_total']}")
    else:
        print(f"❌ Backend Error: {backend_data['error']}")
    
    print()
    
    # Model Information
    print("📊 MODEL INFORMATION")
    print("-" * 40)
    model_data, model_ok = get_model_info()
    
    if model_ok:
        print(f"🎯 Accuracy: {model_data['accuracy']}")
        print(f"⚙️ Parameters: {model_data['parameters']:,}")
        print(f"🏗️ Architecture: {model_data['architecture']}")
        print(f"📚 Training Samples: {model_data['training_samples']}")
        print(f"🔤 Classes: {len(model_data['classes'])}")
    else:
        print(f"❌ Model Error: {model_data['error']}")
    
    print()
    
    # Prediction Test
    print("🧪 PREDICTION TEST")
    print("-" * 40)
    pred_data, pred_ok = test_prediction()
    
    if pred_ok:
        print(f"✅ Prediction: {pred_data['prediction']}")
        print(f"📈 Confidence: {pred_data['confidence']:.4f}")
        print(f"⚡ Response Time: {pred_data['response_time']:.2f}ms")
        print(f"🎮 Device Used: {pred_data['device_used']}")
        
        if 'top3_predictions' in pred_data:
            print("🏆 Top 3 Predictions:")
            for i, (gesture, conf) in enumerate(list(pred_data['top3_predictions'].items())[:3], 1):
                print(f"   {i}. {gesture}: {conf:.4f}")
    else:
        print(f"❌ Prediction Error: {pred_data['error']}")
    
    print()
    
    # System URLs
    print("🌐 SYSTEM URLS")
    print("-" * 40)
    print("🔗 Backend API: http://localhost:5000")
    print("🔗 Frontend App: http://localhost:5173")
    print("🔗 Health Check: http://localhost:5000/health")
    print()
    
    # Instructions
    print("📋 INSTRUCTIONS")
    print("-" * 40)
    print("1. Open http://localhost:5173 in your browser")
    print("2. Allow camera permissions when prompted")
    print("3. Click the 'Start Recognition' button")
    print("4. Hold your hand in front of the camera")
    print("5. Make ASL gestures (A, B, C, etc.)")
    print("6. Watch this dashboard for real-time activity")
    print()
    print("Press Ctrl+C to stop monitoring")

def main():
    """Main monitoring loop"""
    try:
        while True:
            clear_screen()
            print_dashboard()
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")

if __name__ == "__main__":
    main()