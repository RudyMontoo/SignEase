#!/usr/bin/env python3
"""
Direct API Test - Test the backend API directly
"""

import requests
import json

def test_api():
    print("🧪 Direct API Test")
    print("=" * 30)
    
    # Test landmarks for letter "A" (fist with thumb up)
    test_landmarks = [
        # Wrist
        0.5, 0.5, 0.0,
        # Thumb (extended)
        0.45, 0.4, 0.01, 0.42, 0.35, 0.02, 0.39, 0.3, 0.03, 0.36, 0.25, 0.04,
        # Index (folded)
        0.52, 0.45, 0.01, 0.54, 0.5, 0.02, 0.56, 0.55, 0.03, 0.58, 0.6, 0.04,
        # Middle (folded)
        0.55, 0.45, 0.01, 0.57, 0.5, 0.02, 0.59, 0.55, 0.03, 0.61, 0.6, 0.04,
        # Ring (folded)
        0.58, 0.45, 0.01, 0.6, 0.5, 0.02, 0.62, 0.55, 0.03, 0.64, 0.6, 0.04,
        # Pinky (folded)
        0.61, 0.45, 0.01, 0.63, 0.5, 0.02, 0.65, 0.55, 0.03, 0.67, 0.6, 0.04
    ]
    
    try:
        print("📡 Sending request to backend...")
        response = requests.post(
            "http://localhost:5000/predict",
            json={"landmarks": test_landmarks},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Confidence: {data['confidence']:.4f}")
            print(f"   Time: {data['inference_time_ms']:.2f}ms")
            print(f"   Device: {data['device_used']}")
            
            if 'top3_predictions' in data:
                print("   Top 3 predictions:")
                for i, (gesture, conf) in enumerate(list(data['top3_predictions'].items())[:3], 1):
                    print(f"     {i}. {gesture}: {conf:.4f}")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health():
    print("\n🏥 Health Check")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is healthy!")
            print(f"   Model: {data['model_type']}")
            print(f"   Device: {data['device']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SignEase Backend API Test")
    print("=" * 40)
    
    # Test health first
    health_ok = test_health()
    
    if health_ok:
        # Test prediction
        api_ok = test_api()
        
        if api_ok:
            print("\n🎉 Backend is working perfectly!")
            print("   The issue is likely in the frontend.")
            print("   Make sure to:")
            print("   1. Click 'Start Recognition' button")
            print("   2. Allow camera permissions")
            print("   3. Hold your hand in front of camera")
        else:
            print("\n❌ API prediction failed")
    else:
        print("\n❌ Backend is not responding")
        print("   Please start the backend server:")
        print("   python production_backend.py")