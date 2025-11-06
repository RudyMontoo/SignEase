#!/usr/bin/env python3
"""
Performance Testing Script
=========================

Test the performance optimizations of the SignEase ASL Recognition API.
"""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

from performance.load_tester import LoadTester

def main():
    """Main performance testing function"""
    print("🚀 SignEase Performance Testing")
    print("=" * 50)
    
    # Check if server is running
    tester = LoadTester()
    
    print("1. Testing basic connectivity...")
    basic_test = tester.single_request_test()
    
    if not basic_test['success']:
        print("❌ Server not responding. Make sure the backend is running:")
        print("   python start_backend.py")
        return
    
    print("✅ Server is responding")
    print(f"   Response time: {basic_test['response_time_ms']:.1f}ms")
    print(f"   Gesture: {basic_test.get('gesture', 'N/A')}")
    print(f"   Confidence: {basic_test.get('confidence', 0):.3f}")
    
    # Run performance tests
    print("\n2. Running performance optimization tests...")
    
    # Test concurrent requests
    print("\n   🔥 Concurrent load test (20 requests, 5 workers)...")
    concurrent_result = tester.concurrent_load_test(num_requests=20, max_workers=5)
    
    print(f"      Success rate: {concurrent_result.get('success_rate', 0):.1f}%")
    print(f"      Avg response time: {concurrent_result.get('response_times', {}).get('avg_ms', 0):.1f}ms")
    print(f"      P95 response time: {concurrent_result.get('response_times', {}).get('p95_ms', 0):.1f}ms")
    print(f"      Requests/second: {concurrent_result.get('requests_per_second', 0):.1f}")
    
    if concurrent_result.get('cache_hit_rate', 0) > 0:
        print(f"      Cache hit rate: {concurrent_result.get('cache_hit_rate', 0):.1f}%")
    
    # Test memory usage
    print("\n   🧠 Memory stress test (30 requests)...")
    memory_result = tester.memory_stress_test(num_requests=30)
    
    print(f"      Memory increase: {memory_result.get('memory_increase_mb', 0):.1f}MB")
    if 'gpu_memory_increase_mb' in memory_result:
        print(f"      GPU memory increase: {memory_result.get('gpu_memory_increase_mb', 0):.1f}MB")
    
    # Performance assessment
    print("\n3. Performance Assessment:")
    print("   " + "="*30)
    
    # Check if performance meets targets
    avg_response = concurrent_result.get('response_times', {}).get('avg_ms', 0)
    success_rate = concurrent_result.get('success_rate', 0)
    memory_increase = memory_result.get('memory_increase_mb', 0)
    
    issues = []
    if avg_response > 50:
        issues.append(f"Average response time ({avg_response:.1f}ms) exceeds target (50ms)")
    
    if success_rate < 95:
        issues.append(f"Success rate ({success_rate:.1f}%) below target (95%)")
    
    if memory_increase > 50:
        issues.append(f"Memory increase ({memory_increase:.1f}MB) is high")
    
    if not issues:
        print("   ✅ All performance targets met!")
        print("   🚀 System is optimized and ready for production")
    else:
        print("   ⚠️  Performance issues detected:")
        for issue in issues:
            print(f"      - {issue}")
    
    print(f"\n📊 Performance Summary:")
    print(f"   Target response time: <50ms (Actual: {avg_response:.1f}ms)")
    print(f"   Target success rate: >95% (Actual: {success_rate:.1f}%)")
    print(f"   Target concurrent requests: >10 (Tested: 5 workers)")
    print(f"   Memory management: {'✅ Good' if memory_increase < 50 else '⚠️ Needs attention'}")
    
    print("\n✅ Performance testing completed!")

if __name__ == "__main__":
    main()