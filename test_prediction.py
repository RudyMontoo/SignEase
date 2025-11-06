#!/usr/bin/env python3
"""
Test script to verify SignEase MVP gesture prediction
"""

import requests
import json
import numpy as np

# Sample hand landmarks for testing (21 points × 3 coordinates = 63 values)
# This represents a simple hand pose
sample_landmarks = [
    # Wrist
    0.5, 0.5, 0.0,
    # Thumb
    0.4, 0.4, 0.0, 0.35, 0.35, 0.0, 0.3, 0.3, 0.0, 0.25, 0.25, 0.0,
    # Index finger
    0.6, 0.3, 0.0, 0.65, 0.25, 0.0, 0.7, 0.2, 0.0, 0.75, 0.15, 0.0,
    # Middle finger
    0.6, 0.2, 0.0, 0.65, 0.15, 0.0, 0.7, 0.1, 0.0, 0.75, 0.05, 0.0,
    # Ring finger
    0.55, 0.25, 0.0, 0.6, 0.2, 0.0, 0.65, 0.15, 0.0, 0.7, 0.1, 0.0,
    # Pinky
    0.5, 0.3, 0.0, 0.55, 0.25, 0.0, 0.6, 0.2, 0.0, 0.65, 0.15, 0.0
]

def test_prediction():
    """Test gesture prediction API"""
    url = "http://localhost:5000/predict"
    
    payload = {
        "landmarks": sample_landmarks
    }
    
    try:
        print("🧪 Testing SignEase MVP Gesture Prediction...")
        print(f"📡 Sending request to: {url}")
        print(f"📊 Landmarks shape: {len(sample_landmarks)} values")
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ PREDICTION SUCCESS!")
            print(f"🤟 Predicted Gesture: {result.get('gesture', 'N/A')}")
            print(f"🎯 Confidence: {result.get('confidence', 0):.2%}")
            print(f"⚡ Inference Time: {result.get('inference_time_ms', 0):.2f}ms")
            
            alternatives = result.get('alternatives', [])
            if alternatives:
                print(f"\n🔄 Top 3 Alternatives:")
                for i, alt in enumerate(alternatives[:3], 1):
                    print(f"   {i}. {alt['gesture']} ({alt['confidence']:.2%})")
            
            return True
        else:
            print(f"❌ PREDICTION FAILED!")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION FAILED!")
        print("Backend server is not running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Backend Health Check PASSED")
            print(f"   Model Loaded: {health.get('model_loaded', False)}")
            print(f"   GPU Available: {health.get('gpu_available', False)}")
            print(f"   Device: {health.get('device', 'N/A')}")
            return True
        else:
            print("❌ Backend Health Check FAILED")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SignEase MVP - System Test")
    print("=" * 50)
    
    # Test health first
    health_ok = test_health()
    print()
    
    if health_ok:
        # Test prediction
        prediction_ok = test_prediction()
        
        if prediction_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("SignEase MVP is ready for demo! 🚀")
        else:
            print("\n⚠️  Prediction test failed")
    else:
        print("\n❌ Backend is not healthy")
    
    print("\n" + "=" * 50)