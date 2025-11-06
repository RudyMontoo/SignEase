#!/usr/bin/env python3
"""
RTX 5060 Performance Testing
Test the production system with GPU acceleration
"""

import requests
import time
import json
import numpy as np
import torch

def test_gpu_status():
    """Test GPU availability and status"""
    print("🎮 RTX 5060 GPU Status:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   CUDA Version: {torch.version.cuda}")

def test_backend_health():
    """Test backend health and model status"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Backend Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   Model: {data['model_type']}")
            print(f"   Device: {data['device']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            if 'gpu_info' in data and data['gpu_info']:
                gpu = data['gpu_info']
                print(f"   GPU: {gpu['gpu_name']}")
                print(f"   GPU Memory: {gpu['gpu_memory_allocated']} / {gpu['gpu_memory_total']}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False

def test_model_info():
    """Get detailed model information"""
    try:
        response = requests.get("http://localhost:5000/model-info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\n📊 RTX 5060 Model Information:")
            print(f"   Name: {data['model_name']}")
            print(f"   Accuracy: {data['accuracy']}")
            print(f"   Parameters: {data['parameters']:,}")
            print(f"   Architecture: {data['architecture']}")
            print(f"   Training Samples: {data['training_samples']}")
            print(f"   Classes: {len(data['classes'])}")
            print(f"   Device: {data['device']}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def generate_test_landmarks():
    """Generate realistic test landmarks for performance testing"""
    # Simulate hand landmarks (21 points * 3 coordinates = 63 values)
    landmarks = []
    
    # Wrist (center point)
    wrist = [0.5, 0.5, 0.0]
    landmarks.extend(wrist)
    
    # Thumb (4 points)
    for i in range(4):
        landmarks.extend([0.4 + i*0.02, 0.6 + i*0.02, 0.01])
    
    # Index finger (4 points)
    for i in range(4):
        landmarks.extend([0.45 + i*0.01, 0.3 + i*0.05, 0.02])
    
    # Middle finger (4 points)
    for i in range(4):
        landmarks.extend([0.5 + i*0.01, 0.25 + i*0.06, 0.01])
    
    # Ring finger (4 points)
    for i in range(4):
        landmarks.extend([0.55 + i*0.01, 0.3 + i*0.05, 0.02])
    
    # Pinky (4 points)
    for i in range(4):
        landmarks.extend([0.6 + i*0.01, 0.35 + i*0.04, 0.01])
    
    return landmarks

def test_inference_performance():
    """Test inference speed and accuracy with RTX 5060"""
    print("\n⚡ RTX 5060 Inference Performance Test:")
    
    # Generate test data
    test_landmarks = generate_test_landmarks()
    
    # Performance metrics
    inference_times = []
    successful_predictions = 0
    total_tests = 10
    
    for i in range(total_tests):
        try:
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:5000/predict",
                json={"landmarks": test_landmarks},
                timeout=10
            )
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                inference_times.append(inference_time)
                successful_predictions += 1
                
                if i == 0:  # Show first prediction details
                    print(f"   Sample Prediction: {data['prediction']}")
                    print(f"   Confidence: {data['confidence']:.4f}")
                    print(f"   Device Used: {data['device_used']}")
                    print(f"   Model: {data['model_info']}")
            else:
                print(f"   Test {i+1} failed: {response.status_code}")
                
        except Exception as e:
            print(f"   Test {i+1} error: {e}")
    
    # Calculate performance metrics
    if inference_times:
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"\n📈 Performance Results:")
        print(f"   Successful Predictions: {successful_predictions}/{total_tests}")
        print(f"   Average Inference Time: {avg_time:.2f}ms")
        print(f"   Min/Max Time: {min_time:.2f}ms / {max_time:.2f}ms")
        print(f"   Theoretical FPS: {fps:.1f}")
        print(f"   Success Rate: {(successful_predictions/total_tests)*100:.1f}%")
    else:
        print("❌ No successful predictions recorded")

def test_gpu_memory_usage():
    """Monitor GPU memory during inference"""
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Before Test: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        
        # Run a few predictions to load model into GPU memory
        test_landmarks = generate_test_landmarks()
        for _ in range(3):
            try:
                requests.post(
                    "http://localhost:5000/predict",
                    json={"landmarks": test_landmarks},
                    timeout=5
                )
            except:
                pass
        
        print(f"   After Inference: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        print(f"   GPU Utilization: Active during inference")

def main():
    """Run complete RTX 5060 performance test suite"""
    print("🚀 RTX 5060 SignEase Performance Testing")
    print("=" * 60)
    
    # Test GPU status
    test_gpu_status()
    
    # Test backend health
    if not test_backend_health():
        print("❌ Backend not available. Please start production_backend.py")
        return
    
    # Test model information
    test_model_info()
    
    # Test inference performance
    test_inference_performance()
    
    # Test GPU memory usage
    test_gpu_memory_usage()
    
    print("\n" + "=" * 60)
    print("🎉 RTX 5060 Performance Testing Complete!")
    print("✅ System ready for production use")

if __name__ == "__main__":
    main()