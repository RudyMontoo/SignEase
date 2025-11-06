#!/usr/bin/env python3
"""
Real-time System Monitor
Monitor SignEase system activity in real-time
"""

import requests
import time
import json
import threading
from datetime import datetime
import torch

class SignEaseMonitor:
    def __init__(self):
        self.running = False
        self.request_count = 0
        self.last_prediction = None
        self.last_confidence = 0
        self.backend_status = "Unknown"
        
    def check_backend_status(self):
        """Check backend status"""
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.backend_status = f"✅ {data['model_type']} on {data['device']}"
                return True
            else:
                self.backend_status = f"❌ HTTP {response.status_code}"
                return False
        except Exception as e:
            self.backend_status = f"❌ {str(e)[:50]}"
            return False
    
    def monitor_gpu(self):
        """Monitor GPU usage"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**2
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU: {memory_used:.1f}MB / {memory_total:.1f}GB"
        return "GPU: Not available"
    
    def print_status(self):
        """Print current system status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        gpu_info = self.monitor_gpu()
        
        print(f"\r[{timestamp}] Backend: {self.backend_status} | {gpu_info} | Requests: {self.request_count} | Last: {self.last_prediction} ({self.last_confidence:.3f})", end="", flush=True)
    
    def monitor_requests(self):
        """Monitor incoming requests by checking backend logs"""
        # This will be updated when we see actual requests
        pass
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.running = True
        print("🔍 SignEase Real-time Monitor Started")
        print("=" * 80)
        print("Monitoring:")
        print("  • Backend API status")
        print("  • GPU memory usage") 
        print("  • Request count")
        print("  • Latest predictions")
        print("  • System performance")
        print()
        print("Now go to http://localhost:5173 and start using the system!")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)
        
        try:
            while self.running:
                # Check backend status
                self.check_backend_status()
                
                # Print status line
                self.print_status()
                
                # Sleep for a short interval
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            self.running = False
    
    def test_prediction_endpoint(self):
        """Test the prediction endpoint with sample data"""
        print("\n🧪 Testing prediction endpoint...")
        
        # Sample landmarks for testing
        test_landmarks = [0.5, 0.5, 0.0] + [0.4 + i*0.01 for i in range(60)]
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:5000/predict",
                json={"landmarks": test_landmarks},
                timeout=10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                self.request_count += 1
                self.last_prediction = data['prediction']
                self.last_confidence = data['confidence']
                
                print(f"✅ Test successful!")
                print(f"   Prediction: {data['prediction']}")
                print(f"   Confidence: {data['confidence']:.4f}")
                print(f"   Response time: {(end_time - start_time)*1000:.2f}ms")
                print(f"   Device: {data['device_used']}")
                return True
            else:
                print(f"❌ Test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False

def main():
    monitor = SignEaseMonitor()
    
    print("🚀 SignEase System Monitor")
    print("=" * 50)
    
    # Initial backend check
    if monitor.check_backend_status():
        print(f"Backend Status: {monitor.backend_status}")
        
        # Test prediction endpoint
        if monitor.test_prediction_endpoint():
            print("\n✅ System is ready for testing!")
        else:
            print("\n❌ Prediction endpoint has issues")
    else:
        print(f"❌ Backend not responding: {monitor.backend_status}")
        print("Please make sure backend is running: python production_backend.py")
        return
    
    print(f"GPU Status: {monitor.monitor_gpu()}")
    print()
    
    # Start real-time monitoring
    monitor.start_monitoring()

if __name__ == "__main__":
    main()