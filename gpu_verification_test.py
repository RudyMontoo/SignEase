#!/usr/bin/env python3
"""
Comprehensive GPU verification test for SignEase MVP
"""

import torch
import time
import sys

def verify_gpu_availability():
    """Verify GPU availability and basic functionality"""
    print("=== SignEase MVP - GPU Verification Test ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ GPU not available - will use CPU for inference")
        return False
    
    # GPU Information
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    
    print(f"✅ GPU detected: {device_name}")
    print(f"   Device count: {device_count}")
    print(f"   Current device: {current_device}")
    print(f"   Total memory: {device_props.total_memory / 1e9:.1f} GB")
    print(f"   Compute capability: {device_props.major}.{device_props.minor}")
    
    return True

def test_gpu_operations():
    """Test basic GPU operations for ML inference"""
    print("\n=== GPU Operations Test ===")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping GPU tests - CUDA not available")
        return False
    
    try:
        # Test basic tensor operations
        print("Testing basic tensor operations...")
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✅ Matrix multiplication (100x100): {gpu_time*1000:.2f}ms")
        
        # Test larger operations (more realistic for ML)
        print("Testing ML-sized operations...")
        x_large = torch.randn(1000, 512, device='cuda')
        y_large = torch.randn(512, 256, device='cuda')
        
        start_time = time.time()
        z_large = torch.mm(x_large, y_large)
        torch.cuda.synchronize()
        large_gpu_time = time.time() - start_time
        
        print(f"✅ Large matrix multiplication (1000x512 @ 512x256): {large_gpu_time*1000:.2f}ms")
        
        # Test neural network operations
        print("Testing neural network operations...")
        input_tensor = torch.randn(32, 21, device='cuda')  # Batch of 32, 21 landmarks
        linear_layer = torch.nn.Linear(21, 64).cuda()
        
        start_time = time.time()
        output = linear_layer(input_tensor)
        torch.cuda.synchronize()
        nn_time = time.time() - start_time
        
        print(f"✅ Neural network forward pass (32x21 -> 32x64): {nn_time*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

def test_memory_management():
    """Test GPU memory management"""
    print("\n=== GPU Memory Management Test ===")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping memory tests - CUDA not available")
        return False
    
    try:
        # Check initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {initial_memory / 1e6:.1f} MB")
        
        # Allocate some tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000, device='cuda')
            tensors.append(tensor)
        
        allocated_memory = torch.cuda.memory_allocated()
        print(f"After allocation: {allocated_memory / 1e6:.1f} MB")
        print(f"Memory used: {(allocated_memory - initial_memory) / 1e6:.1f} MB")
        
        # Clear tensors
        del tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        print(f"After cleanup: {final_memory / 1e6:.1f} MB")
        
        print("✅ Memory management working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def performance_comparison():
    """Compare CPU vs GPU performance"""
    print("\n=== CPU vs GPU Performance Comparison ===")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping comparison - GPU not available")
        return
    
    try:
        # Test size relevant to gesture recognition
        size = 512
        iterations = 10
        
        # CPU test
        print(f"Testing CPU performance ({iterations} iterations)...")
        cpu_times = []
        for _ in range(iterations):
            x_cpu = torch.randn(size, size)
            y_cpu = torch.randn(size, size)
            
            start_time = time.time()
            z_cpu = torch.mm(x_cpu, y_cpu)
            cpu_times.append(time.time() - start_time)
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        print(f"CPU average time: {avg_cpu_time*1000:.2f}ms")
        
        # GPU test
        print(f"Testing GPU performance ({iterations} iterations)...")
        gpu_times = []
        for _ in range(iterations):
            x_gpu = torch.randn(size, size, device='cuda')
            y_gpu = torch.randn(size, size, device='cuda')
            
            start_time = time.time()
            z_gpu = torch.mm(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start_time)
        
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        print(f"GPU average time: {avg_gpu_time*1000:.2f}ms")
        
        # Calculate speedup
        speedup = avg_cpu_time / avg_gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✅ GPU provides significant performance benefit")
        else:
            print("⚠️  GPU speedup is minimal - CPU might be sufficient")
            
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")

def main():
    """Run all GPU verification tests"""
    print("Starting GPU verification for SignEase MVP...\n")
    
    # Run tests
    gpu_available = verify_gpu_availability()
    gpu_ops_ok = test_gpu_operations()
    memory_ok = test_memory_management()
    
    # Performance comparison
    performance_comparison()
    
    # Summary
    print("\n" + "="*50)
    print("GPU VERIFICATION SUMMARY")
    print("="*50)
    
    if gpu_available and gpu_ops_ok and memory_ok:
        print("🎉 GPU VERIFICATION SUCCESSFUL!")
        print("✅ CUDA available and working")
        print("✅ GPU operations functional")
        print("✅ Memory management working")
        print("✅ Ready for GPU-accelerated inference")
        
        print("\n📋 Recommendations for SignEase MVP:")
        print("• Use GPU for model training (if time permits)")
        print("• GPU inference will provide faster response times")
        print("• Monitor GPU memory usage during development")
        print("• Have CPU fallback ready for deployment")
        
    else:
        print("⚠️  GPU verification had issues:")
        print(f"   GPU Available: {'✅' if gpu_available else '❌'}")
        print(f"   GPU Operations: {'✅' if gpu_ops_ok else '❌'}")
        print(f"   Memory Management: {'✅' if memory_ok else '❌'}")
        
        print("\n📋 Fallback plan:")
        print("• Use CPU-based inference (still viable for MVP)")
        print("• Focus on model optimization for CPU performance")
        print("• Consider cloud GPU for training if needed")
    
    print(f"\nEnvironment: Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
    print("Ready to proceed with SignEase MVP development!")

if __name__ == "__main__":
    main()