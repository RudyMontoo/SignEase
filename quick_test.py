#!/usr/bin/env python3
"""Quick test of core functionality"""

print("🧪 Quick Test")
print("-" * 20)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV: {e}")

try:
    import mediapipe as mp
    print(f"✅ MediaPipe {mp.__version__}")
except Exception as e:
    print(f"❌ MediaPipe: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")

print("\n🎯 Core dependencies ready for ASL training!")