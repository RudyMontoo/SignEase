#!/usr/bin/env python3
"""
Comprehensive test for SignEase MVP with MediaPipe support
"""

import sys

def test_all_dependencies():
    """Test all required dependencies including MediaPipe"""
    print("=== SignEase MVP - Complete Dependency Test ===")
    print(f"Python version: {sys.version}")
    print()
    
    success_count = 0
    total_tests = 8
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - OK")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        success_count += 1
    except ImportError as e:
        print(f"❌ PyTorch - FAILED: {e}")
    
    # Test MediaPipe (should work now!)
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe {mp.__version__} - OK")
        print(f"   Solutions available: {len(dir(mp.solutions))}")
        success_count += 1
    except ImportError as e:
        print(f"❌ MediaPipe - FAILED: {e}")
    
    # Test OpenCV
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} - OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ OpenCV - FAILED: {e}")
    
    # Test Flask
    try:
        import flask
        print(f"✅ Flask {flask.__version__} - OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ Flask - FAILED: {e}")
    
    # Test Flask-CORS
    try:
        import flask_cors
        print(f"✅ Flask-CORS - OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ Flask-CORS - FAILED: {e}")
    
    # Test scikit-learn
    try:
        import sklearn
        print(f"✅ scikit-learn {sklearn.__version__} - OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ scikit-learn - FAILED: {e}")
    
    # Test pandas
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__} - OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ pandas - FAILED: {e}")
    
    # Test numpy
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__} - OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ numpy - FAILED: {e}")
    
    print(f"\n=== Dependency Test Results: {success_count}/{total_tests} ===")
    return success_count == total_tests

def test_mediapipe_functionality():
    """Test MediaPipe hand detection functionality"""
    print("\n=== MediaPipe Functionality Test ===")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe Hands initialized successfully")
        
        # Create a test image with a simple hand-like shape
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple hand-like shape for testing
        cv2.circle(test_image, (320, 240), 50, (255, 255, 255), -1)  # Palm
        cv2.circle(test_image, (300, 200), 15, (255, 255, 255), -1)  # Finger 1
        cv2.circle(test_image, (320, 190), 15, (255, 255, 255), -1)  # Finger 2
        cv2.circle(test_image, (340, 200), 15, (255, 255, 255), -1)  # Finger 3
        cv2.circle(test_image, (360, 210), 15, (255, 255, 255), -1)  # Finger 4
        cv2.circle(test_image, (280, 260), 15, (255, 255, 255), -1)  # Thumb
        
        # Process the test image
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        print("✅ MediaPipe image processing working")
        
        # Check if landmarks were detected (may not detect on simple test image)
        if results.multi_hand_landmarks:
            print(f"✅ Hand landmarks detected: {len(results.multi_hand_landmarks)} hands")
            for hand_landmarks in results.multi_hand_landmarks:
                print(f"   Landmarks count: {len(hand_landmarks.landmark)}")
        else:
            print("ℹ️  No hand landmarks detected (expected for simple test image)")
        
        hands.close()
        print("✅ MediaPipe Hands closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe functionality test failed: {e}")
        return False

def test_pytorch_functionality():
    """Test PyTorch basic operations"""
    print("\n=== PyTorch Functionality Test ===")
    
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x @ y
        
        print("✅ PyTorch tensor operations working")
        print(f"   Tensor shape: {z.shape}")
        print(f"   Device: {z.device}")
        
        # Test if CUDA is available and working
        if torch.cuda.is_available():
            try:
                x_gpu = torch.randn(3, 3).cuda()
                y_gpu = torch.randn(3, 3).cuda()
                z_gpu = x_gpu @ y_gpu
                print("✅ PyTorch GPU operations working")
                print(f"   GPU tensor device: {z_gpu.device}")
            except Exception as e:
                print(f"⚠️  GPU operations failed: {e}")
        else:
            print("ℹ️  CUDA not available - using CPU only")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch functionality test failed: {e}")
        return False

def test_flask_setup():
    """Test Flask basic setup"""
    print("\n=== Flask Setup Test ===")
    
    try:
        from flask import Flask, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/test')
        def test_route():
            return jsonify({"status": "success", "message": "Flask working"})
        
        print("✅ Flask app created successfully")
        print("✅ CORS configured")
        print("✅ Test route defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask setup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting comprehensive SignEase MVP dependency tests...\n")
    
    # Run all tests
    deps_ok = test_all_dependencies()
    mediapipe_ok = test_mediapipe_functionality()
    pytorch_ok = test_pytorch_functionality()
    flask_ok = test_flask_setup()
    
    # Summary
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    
    if deps_ok and mediapipe_ok and pytorch_ok and flask_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ SignEase MVP environment is fully ready")
        print("✅ MediaPipe compatibility issue RESOLVED")
        print("✅ Ready to proceed with development")
    else:
        print("⚠️  Some tests failed:")
        print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
        print(f"   MediaPipe: {'✅' if mediapipe_ok else '❌'}")
        print(f"   PyTorch: {'✅' if pytorch_ok else '❌'}")
        print(f"   Flask: {'✅' if flask_ok else '❌'}")
    
    print("\n=== Environment Summary ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Virtual Environment: signease-py311")
    print("Ready for SignEase MVP development!")

if __name__ == "__main__":
    main()