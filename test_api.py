#!/usr/bin/env python3
"""Test the SignEase API endpoints"""

import requests
import json
import numpy as np

# Test data - sample landmarks for letter 'A'
sample_landmarks = np.random.rand(63).tolist()  # 21 landmarks * 3 coordinates

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health endpoint...")
    response = requests.get('http://localhost:5000/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict():
    """Test prediction endpoint"""
    print("🔍 Testing /predict endpoint...")
    
    data = {
        'landmarks': sample_landmarks,
        'handedness': 'Right'
    }
    
    response = requests.post(
        'http://localhost:5000/predict',
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_metrics():
    """Test metrics endpoint"""
    print("🔍 Testing /metrics endpoint...")
    response = requests.get('http://localhost:5000/metrics')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == '__main__':
    print("🧪 SignEase API Test Suite")
    print("=" * 40)
    
    try:
        test_health()
        test_predict()
        test_metrics()
        print("✅ All API tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"❌ Test failed: {e}")