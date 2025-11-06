#!/usr/bin/env python3
"""
Load Testing Utility
====================

Comprehensive load testing for the SignEase ASL Recognition API.
Tests concurrent requests, performance under load, and memory usage.
"""

import time
import numpy as np
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import statistics

class LoadTester:
    """Load testing utility for SignEase API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results = []
    
    def generate_test_landmarks(self) -> List[float]:
        """Generate realistic test landmarks"""
        # Generate 21 landmarks with x,y,z coordinates (63 values total)
        landmarks = []
        
        # Simulate hand landmarks with realistic ranges
        for i in range(21):
            x = np.random.uniform(0.2, 0.8)  # Normalized x coordinate
            y = np.random.uniform(0.2, 0.8)  # Normalized y coordinate
            z = np.random.uniform(-0.1, 0.1)  # Depth coordinate
            landmarks.extend([x, y, z])
        
        return landmarks
    
    def single_request_test(self, session_id: int = 0) -> Dict[str, Any]:
        """Test a single prediction request"""
        import requests
        
        landmarks = self.generate_test_landmarks()
        payload = {
            'landmarks': landmarks,
            'handedness': 'Right'
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to ms
            
            result = {
                'session_id': session_id,
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_time_ms': response_time,
                'timestamp': start_time
            }
            
            if response.status_code == 200:
                data = response.json()
                result.update({
                    'gesture': data.get('gesture'),
                    'confidence': data.get('confidence'),
                    'inference_time_ms': data.get('inference_time_ms'),
                    'cached': data.get('cached', False)
                })
            else:
                result['error'] = response.text
            
            return result
            
        except Exception as e:
            end_time = time.time()
            return {
                'session_id': session_id,
                'success': False,
                'response_time_ms': (end_time - start_time) * 1000,
                'timestamp': start_time,
                'error': str(e)
            }
    
    def concurrent_load_test(self, num_requests: int = 50, max_workers: int = 10) -> Dict[str, Any]:
        """Test concurrent requests"""
        print(f"🔥 Running concurrent load test: {num_requests} requests with {max_workers} workers")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            futures = [executor.submit(self.single_request_test, i) for i in range(num_requests)]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'success': False,
                        'error': str(e),
                        'response_time_ms': 0
                    })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time_ms'] for r in successful_requests]
            inference_times = [r.get('inference_time_ms', 0) for r in successful_requests if r.get('inference_time_ms')]
            cached_requests = [r for r in successful_requests if r.get('cached', False)]
            
            analysis = {
                'total_requests': num_requests,
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': (len(successful_requests) / num_requests) * 100,
                'total_time_seconds': total_time,
                'requests_per_second': num_requests / total_time,
                'cached_requests': len(cached_requests),
                'cache_hit_rate': (len(cached_requests) / len(successful_requests)) * 100,
                'response_times': {
                    'min_ms': min(response_times),
                    'max_ms': max(response_times),
                    'avg_ms': statistics.mean(response_times),
                    'median_ms': statistics.median(response_times),
                    'p95_ms': np.percentile(response_times, 95),
                    'p99_ms': np.percentile(response_times, 99)
                }
            }
            
            if inference_times:
                analysis['inference_times'] = {
                    'min_ms': min(inference_times),
                    'max_ms': max(inference_times),
                    'avg_ms': statistics.mean(inference_times),
                    'median_ms': statistics.median(inference_times)
                }
        else:
            analysis = {
                'total_requests': num_requests,
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'success_rate': 0,
                'total_time_seconds': total_time,
                'error': 'All requests failed'
            }
        
        return analysis
    
    def sustained_load_test(self, duration_seconds: int = 60, requests_per_second: int = 5) -> Dict[str, Any]:
        """Test sustained load over time"""
        print(f"⏱️  Running sustained load test: {duration_seconds}s at {requests_per_second} req/s")
        
        start_time = time.time()
        results = []
        request_interval = 1.0 / requests_per_second
        
        def make_request():
            result = self.single_request_test()
            results.append(result)
        
        # Schedule requests
        threads = []
        while time.time() - start_time < duration_seconds:
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)
            
            time.sleep(request_interval)
        
        # Wait for all requests to complete
        for thread in threads:
            thread.join(timeout=10)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        
        if successful_requests:
            response_times = [r['response_time_ms'] for r in successful_requests]
            
            analysis = {
                'duration_seconds': actual_duration,
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'success_rate': (len(successful_requests) / len(results)) * 100,
                'actual_rps': len(results) / actual_duration,
                'avg_response_time_ms': statistics.mean(response_times),
                'p95_response_time_ms': np.percentile(response_times, 95)
            }
        else:
            analysis = {
                'duration_seconds': actual_duration,
                'total_requests': len(results),
                'successful_requests': 0,
                'success_rate': 0,
                'error': 'All requests failed'
            }
        
        return analysis
    
    def memory_stress_test(self, num_requests: int = 100) -> Dict[str, Any]:
        """Test memory usage under load"""
        print(f"🧠 Running memory stress test: {num_requests} requests")
        
        import psutil
        import requests
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**2  # MB
        
        # Check initial GPU memory
        initial_gpu_memory = None
        try:
            import torch
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        except:
            pass
        
        # Run requests and monitor memory
        memory_samples = []
        
        for i in range(num_requests):
            # Make request
            result = self.single_request_test(i)
            
            # Sample memory every 10 requests
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024**2
                sample = {
                    'request_number': i,
                    'system_memory_mb': current_memory,
                    'memory_increase_mb': current_memory - initial_memory
                }
                
                if initial_gpu_memory is not None:
                    try:
                        current_gpu = torch.cuda.memory_allocated() / 1024**2
                        sample['gpu_memory_mb'] = current_gpu
                        sample['gpu_increase_mb'] = current_gpu - initial_gpu_memory
                    except:
                        pass
                
                memory_samples.append(sample)
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024**2
        final_gpu_memory = None
        
        if initial_gpu_memory is not None:
            try:
                final_gpu_memory = torch.cuda.memory_allocated() / 1024**2
            except:
                pass
        
        analysis = {
            'total_requests': num_requests,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - initial_memory,
            'memory_samples': memory_samples
        }
        
        if initial_gpu_memory is not None and final_gpu_memory is not None:
            analysis.update({
                'initial_gpu_memory_mb': initial_gpu_memory,
                'final_gpu_memory_mb': final_gpu_memory,
                'gpu_memory_increase_mb': final_gpu_memory - initial_gpu_memory
            })
        
        return analysis
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive load testing suite"""
        print("🚀 Starting comprehensive load testing suite")
        print("=" * 60)
        
        results = {}
        
        try:
            # Test 1: Basic functionality
            print("\n1. Basic functionality test...")
            basic_result = self.single_request_test()
            results['basic_test'] = basic_result
            
            if not basic_result['success']:
                print("❌ Basic test failed. Skipping other tests.")
                return results
            
            # Test 2: Concurrent load
            print("\n2. Concurrent load test...")
            results['concurrent_load'] = self.concurrent_load_test(num_requests=20, max_workers=5)
            
            # Test 3: High concurrency
            print("\n3. High concurrency test...")
            results['high_concurrency'] = self.concurrent_load_test(num_requests=50, max_workers=10)
            
            # Test 4: Sustained load
            print("\n4. Sustained load test...")
            results['sustained_load'] = self.sustained_load_test(duration_seconds=30, requests_per_second=3)
            
            # Test 5: Memory stress
            print("\n5. Memory stress test...")
            results['memory_stress'] = self.memory_stress_test(num_requests=50)
            
            # Overall assessment
            results['assessment'] = self._assess_performance(results)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _assess_performance(self, results: Dict) -> Dict[str, Any]:
        """Assess overall performance based on test results"""
        assessment = {
            'overall_status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check concurrent load performance
        if 'concurrent_load' in results:
            concurrent = results['concurrent_load']
            if concurrent.get('success_rate', 0) < 95:
                assessment['issues'].append(f"Low success rate in concurrent load: {concurrent.get('success_rate', 0):.1f}%")
            
            if concurrent.get('response_times', {}).get('avg_ms', 0) > 100:
                assessment['issues'].append(f"High average response time: {concurrent.get('response_times', {}).get('avg_ms', 0):.1f}ms")
        
        # Check memory usage
        if 'memory_stress' in results:
            memory = results['memory_stress']
            if memory.get('memory_increase_mb', 0) > 100:
                assessment['issues'].append(f"High memory increase: {memory.get('memory_increase_mb', 0):.1f}MB")
                assessment['recommendations'].append("Consider implementing memory cleanup")
        
        # Set overall status
        if len(assessment['issues']) > 2:
            assessment['overall_status'] = 'needs_attention'
        elif len(assessment['issues']) > 0:
            assessment['overall_status'] = 'acceptable'
        
        return assessment

def main():
    """Main function to run load tests"""
    tester = LoadTester()
    
    print("🧪 SignEase API Load Testing Suite")
    print("=" * 50)
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {filename}")
    
    # Print summary
    if 'assessment' in results:
        assessment = results['assessment']
        print(f"\n📊 Overall Status: {assessment['overall_status'].upper()}")
        
        if assessment['issues']:
            print("⚠️  Issues found:")
            for issue in assessment['issues']:
                print(f"   - {issue}")
        
        if assessment['recommendations']:
            print("💡 Recommendations:")
            for rec in assessment['recommendations']:
                print(f"   - {rec}")
    
    print("\n✅ Load testing completed!")

if __name__ == "__main__":
    main()