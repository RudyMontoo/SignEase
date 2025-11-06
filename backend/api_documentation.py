#!/usr/bin/env python3
"""
SignEase API Documentation & Testing
===================================

Comprehensive API documentation and testing utilities for the SignEase ASL Recognition API.
"""

import requests
import json
import numpy as np
import time
from typing import Dict, List, Any

class SignEaseAPITester:
    """Test suite for SignEase API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test /health endpoint"""
        print("🔍 Testing /health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            result = {
                'endpoint': '/health',
                'method': 'GET',
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                result['data'] = data
                result['model_loaded'] = data.get('model_loaded', False)
                result['gpu_available'] = data.get('gpu_available', False)
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = {
                'endpoint': '/health',
                'method': 'GET',
                'success': False,
                'error': str(e)
            }
            self.test_results.append(result)
            return result
    
    def test_predict_endpoint(self) -> Dict[str, Any]:
        """Test /predict endpoint with sample data"""
        print("🔍 Testing /predict endpoint...")
        
        # Generate sample landmarks (21 points × 3 coordinates = 63 values)
        sample_landmarks = np.random.rand(63).tolist()
        
        payload = {
            'landmarks': sample_landmarks,
            'handedness': 'Right',
            'alternatives': 3
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            result = {
                'endpoint': '/predict',
                'method': 'POST',
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                result['data'] = data
                result['gesture'] = data.get('gesture')
                result['confidence'] = data.get('confidence')
                result['inference_time_ms'] = data.get('inference_time_ms')
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = {
                'endpoint': '/predict',
                'method': 'POST',
                'success': False,
                'error': str(e)
            }
            self.test_results.append(result)
            return result
    
    def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test /metrics endpoint"""
        print("🔍 Testing /metrics endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            
            result = {
                'endpoint': '/metrics',
                'method': 'GET',
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                result['data'] = data
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = {
                'endpoint': '/metrics',
                'method': 'GET',
                'success': False,
                'error': str(e)
            }
            self.test_results.append(result)
            return result
    
    def test_model_info_endpoint(self) -> Dict[str, Any]:
        """Test /model/info endpoint"""
        print("🔍 Testing /model/info endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=5)
            
            result = {
                'endpoint': '/model/info',
                'method': 'GET',
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                result['data'] = data
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = {
                'endpoint': '/model/info',
                'method': 'GET',
                'success': False,
                'error': str(e)
            }
            self.test_results.append(result)
            return result
    
    def test_invalid_request(self) -> Dict[str, Any]:
        """Test error handling with invalid request"""
        print("🔍 Testing error handling...")
        
        # Test with invalid landmarks
        payload = {
            'landmarks': [1, 2, 3],  # Invalid: should be 63 values
            'handedness': 'Right'
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            result = {
                'endpoint': '/predict (invalid)',
                'method': 'POST',
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 400,  # Should return 400 for bad request
                'expected_error': True
            }
            
            if response.status_code == 400:
                data = response.json()
                result['error_message'] = data.get('message', 'No error message')
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = {
                'endpoint': '/predict (invalid)',
                'method': 'POST',
                'success': False,
                'error': str(e)
            }
            self.test_results.append(result)
            return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests"""
        print("🧪 SignEase API Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all tests
        self.test_health_endpoint()
        self.test_predict_endpoint()
        self.test_metrics_endpoint()
        self.test_model_info_endpoint()
        self.test_invalid_request()
        
        total_time = time.time() - start_time
        
        # Calculate summary
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get('success', False)])
        
        summary = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            'total_time_seconds': total_time,
            'results': self.test_results
        }
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Failed: {total_tests - successful_tests}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Total Time: {total_time:.2f}s")
        
        if summary['success_rate'] == 100:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed. Check results for details.")
        
        return summary

def print_api_documentation():
    """Print comprehensive API documentation"""
    
    print("📚 SignEase ASL Recognition API Documentation")
    print("=" * 60)
    print()
    
    print("🌐 Base URL: http://localhost:5000")
    print()
    
    print("📡 Available Endpoints:")
    print("-" * 30)
    
    # Health endpoint
    print("1. GET /health")
    print("   Description: Check API health and status")
    print("   Response: JSON with system status, GPU info, model status")
    print("   Example: curl http://localhost:5000/health")
    print()
    
    # Predict endpoint
    print("2. POST /predict")
    print("   Description: Predict ASL gesture from hand landmarks")
    print("   Content-Type: application/json")
    print("   Request Body:")
    print("   {")
    print('     "landmarks": [x1,y1,z1, x2,y2,z2, ...],  // 63 values (21 points × 3 coords)')
    print('     "handedness": "Right",                     // Optional')
    print('     "alternatives": 3                          // Optional, default 3')
    print("   }")
    print("   Response:")
    print("   {")
    print('     "gesture": "A",')
    print('     "confidence": 0.95,')
    print('     "alternatives": [{"gesture": "A", "confidence": 0.95}, ...],')
    print('     "inference_time_ms": 45.2')
    print("   }")
    print()
    
    # Metrics endpoint
    print("3. GET /metrics")
    print("   Description: Get API performance metrics")
    print("   Response: JSON with performance statistics")
    print("   Example: curl http://localhost:5000/metrics")
    print()
    
    # Model info endpoint
    print("4. GET /model/info")
    print("   Description: Get model information and statistics")
    print("   Response: JSON with model details and performance stats")
    print("   Example: curl http://localhost:5000/model/info")
    print()
    
    # Model classes endpoint
    print("5. GET /model/classes")
    print("   Description: Get available gesture classes")
    print("   Response: JSON with list of supported gestures")
    print("   Example: curl http://localhost:5000/model/classes")
    print()
    
    print("🔧 Error Handling:")
    print("-" * 20)
    print("- 400 Bad Request: Invalid input data")
    print("- 404 Not Found: Endpoint not found")
    print("- 500 Internal Server Error: Server error")
    print("- 503 Service Unavailable: Model not loaded")
    print()
    
    print("⚡ Performance:")
    print("-" * 15)
    print("- Inference time: <50ms per request")
    print("- GPU acceleration: CUDA enabled")
    print("- Concurrent requests: Supported")
    print("- Request validation: Comprehensive")
    print()

def main():
    """Main function to run API tests and show documentation"""
    
    # Show documentation
    print_api_documentation()
    
    # Ask user if they want to run tests
    try:
        run_tests = input("Run API tests? (y/N): ").lower().strip()
        if run_tests == 'y':
            tester = SignEaseAPITester()
            results = tester.run_all_tests()
            
            # Save results to file
            with open('api_test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Test results saved to api_test_results.json")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()