#!/usr/bin/env python3
"""
Test Frontend-Backend Connection
"""

import requests
import json

def test_backend():
    print("🔧 Testing Backend Connection")
    print("=" * 40)
    
    # Test health
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   Model: {data['model_type']}")
            print(f"   Device: {data['device']}")
        else:
            print(f"❌ Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health error: {e}")
        return False
    
    # Test prediction with simple data
    print("\n🧪 Testing Prediction:")
    test_landmarks = [0.5, 0.5, 0.0] + [0.4 + i*0.01 for i in range(60)]
    
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json={"landmarks": test_landmarks},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction Success:")
            print(f"   Gesture: {data['prediction']}")
            print(f"   Confidence: {data['confidence']:.4f}")
            print(f"   Time: {data['inference_time_ms']:.2f}ms")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_cors():
    print("\n🌐 Testing CORS Headers")
    print("=" * 40)
    
    try:
        response = requests.options("http://localhost:5000/predict")
        print(f"OPTIONS request: {response.status_code}")
        
        headers = response.headers
        print("CORS Headers:")
        for header in ['Access-Control-Allow-Origin', 'Access-Control-Allow-Methods', 'Access-Control-Allow-Headers']:
            value = headers.get(header, 'Not set')
            print(f"   {header}: {value}")
            
    except Exception as e:
        print(f"❌ CORS test error: {e}")

def main():
    print("🚀 Frontend-Backend Connection Test")
    print("=" * 50)
    
    backend_ok = test_backend()
    test_cors()
    
    print("\n" + "=" * 50)
    if backend_ok:
        print("✅ Backend is working correctly!")
        print("\nIf frontend still shows no results, the issue is likely:")
        print("1. MediaPipe not detecting hands in browser")
        print("2. Frontend not sending requests")
        print("3. Browser console errors")
        print("4. CORS issues")
        print("\nNext steps:")
        print("1. Open browser developer tools (F12)")
        print("2. Check Console tab for errors")
        print("3. Check Network tab for API requests")
        print("4. Make sure you clicked 'Start Recognition'")
    else:
        print("❌ Backend has issues - fix backend first")

if __name__ == "__main__":
    main()