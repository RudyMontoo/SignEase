#!/usr/bin/env python3
"""
SignEase System Status Dashboard
Real-time monitoring of RTX 5060 production system
"""

import requests
import time
import torch
import psutil
import json
from datetime import datetime

def get_system_status():
    """Get comprehensive system status"""
    status = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': {},
        'backend': {},
        'frontend': {},
        'system': {}
    }
    
    # GPU Status
    if torch.cuda.is_available():
        status['gpu'] = {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            'memory_allocated': f"{torch.cuda.memory_allocated(0) / 1024**2:.1f}MB",
            'cuda_version': torch.version.cuda,
            'temperature': 'N/A'  # Would need nvidia-ml-py for this
        }
    else:
        status['gpu'] = {'available': False}
    
    # Backend Status
    try:
        response = requests.get("http://localhost:5000/health", timeout=3)
        if response.status_code == 200:
            backend_data = response.json()
            status['backend'] = {
                'status': 'online',
                'model_loaded': backend_data.get('model_loaded', False),
                'model_type': backend_data.get('model_type', 'Unknown'),
                'device': backend_data.get('device', 'Unknown'),
                'url': 'http://localhost:5000'
            }
        else:
            status['backend'] = {'status': 'error', 'code': response.status_code}
    except:
        status['backend'] = {'status': 'offline'}
    
    # Frontend Status
    try:
        response = requests.get("http://localhost:5173", timeout=3)
        status['frontend'] = {
            'status': 'online' if response.status_code == 200 else 'error',
            'url': 'http://localhost:5173'
        }
    except:
        status['frontend'] = {'status': 'offline'}
    
    # System Resources
    status['system'] = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 'N/A'
    }
    
    return status

def print_status_dashboard():
    """Print formatted status dashboard"""
    status = get_system_status()
    
    print("\n" + "="*80)
    print("🚀 SIGNEASE RTX 5060 PRODUCTION SYSTEM STATUS")
    print("="*80)
    print(f"⏰ Timestamp: {status['timestamp']}")
    
    # GPU Status
    print(f"\n🎮 RTX 5060 GPU:")
    if status['gpu']['available']:
        print(f"   ✅ Status: Online")
        print(f"   📱 Device: {status['gpu']['name']}")
        print(f"   💾 Memory: {status['gpu']['memory_allocated']} / {status['gpu']['memory_total']}")
        print(f"   🔧 CUDA: {status['gpu']['cuda_version']}")
    else:
        print(f"   ❌ Status: Not Available")
    
    # Backend Status
    print(f"\n🔧 Backend API:")
    if status['backend']['status'] == 'online':
        print(f"   ✅ Status: {status['backend']['status'].upper()}")
        print(f"   🤖 Model: {status['backend']['model_type']}")
        print(f"   📍 Device: {status['backend']['device']}")
        print(f"   🔗 URL: {status['backend']['url']}")
        print(f"   📊 Model Loaded: {status['backend']['model_loaded']}")
    else:
        print(f"   ❌ Status: {status['backend']['status'].upper()}")
    
    # Frontend Status
    print(f"\n🌐 Frontend App:")
    if status['frontend']['status'] == 'online':
        print(f"   ✅ Status: {status['frontend']['status'].upper()}")
        print(f"   🔗 URL: {status['frontend']['url']}")
    else:
        print(f"   ❌ Status: {status['frontend']['status'].upper()}")
    
    # System Resources
    print(f"\n💻 System Resources:")
    print(f"   🔥 CPU Usage: {status['system']['cpu_percent']:.1f}%")
    print(f"   💾 Memory Usage: {status['system']['memory_percent']:.1f}%")
    print(f"   💿 Disk Usage: {status['system']['disk_percent']}")
    
    # Overall Status
    overall_status = "🟢 FULLY OPERATIONAL" if (
        status['gpu']['available'] and 
        status['backend']['status'] == 'online' and 
        status['frontend']['status'] == 'online'
    ) else "🟡 PARTIAL" if (
        status['backend']['status'] == 'online' or 
        status['frontend']['status'] == 'online'
    ) else "🔴 OFFLINE"
    
    print(f"\n🎯 Overall Status: {overall_status}")
    print("="*80)

def test_end_to_end():
    """Test complete end-to-end functionality"""
    print("\n🧪 End-to-End System Test:")
    
    # Test landmarks data
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
            print(f"   ✅ Prediction: {data['prediction']}")
            print(f"   📊 Confidence: {data['confidence']:.4f}")
            print(f"   ⚡ Response Time: {(end_time - start_time)*1000:.2f}ms")
            print(f"   🎮 GPU Inference: {data['device_used']}")
            return True
        else:
            print(f"   ❌ API Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Test Failed: {e}")
        return False

def main():
    """Main dashboard function"""
    try:
        while True:
            print_status_dashboard()
            
            # Run end-to-end test
            test_success = test_end_to_end()
            
            if test_success:
                print("\n🎉 System is ready for production use!")
            else:
                print("\n⚠️ System issues detected")
            
            print("\nPress Ctrl+C to exit, or wait 30 seconds for refresh...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")

if __name__ == "__main__":
    main()