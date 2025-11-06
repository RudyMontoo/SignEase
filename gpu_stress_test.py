#!/usr/bin/env python3
"""
GPU Stress Test - Ensure RTX 5060 reaches 100% utilization
"""

import torch
import time
import numpy as np
from tqdm import tqdm

def gpu_stress_test():
    """Test GPU utilization to 100%"""
    print("🔥 GPU STRESS TEST - RTX 5060")
    print("=" * 40)
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create large tensors to stress GPU
    print("\n🚀 Creating large tensors on GPU...")
    
    # Use most of GPU memory (RTX 5060 has 8GB)
    batch_size = 512
    seq_len = 1024
    hidden_size = 1024
    
    # Create tensors
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    y = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    
    print(f"Tensor X: {x.shape} ({x.element_size() * x.nelement() / 1024**3:.2f} GB)")
    print(f"Tensor Y: {y.shape} ({y.element_size() * y.nelement() / 1024**3:.2f} GB)")
    
    # GPU intensive operations
    print("\n💪 Running GPU intensive operations...")
    print("Check Task Manager - GPU should hit 100%!")
    
    start_time = time.time()
    
    for i in tqdm(range(1000), desc="GPU Stress"):
        # Matrix multiplication (very GPU intensive)
        z = torch.matmul(x, y.transpose(-2, -1))
        
        # More operations
        z = torch.relu(z)
        z = torch.softmax(z, dim=-1)
        z = torch.matmul(z, y)
        
        # Keep GPU busy
        torch.cuda.synchronize()
        
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {i}: {elapsed:.1f}s - GPU should be at 100%!")
    
    total_time = time.time() - start_time
    print(f"\n✅ Stress test complete: {total_time:.1f}s")
    print("GPU utilization should have been 100% during this test!")

if __name__ == "__main__":
    gpu_stress_test()