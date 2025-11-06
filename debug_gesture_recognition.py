#!/usr/bin/env python3
"""
Debug Gesture Recognition System
Comprehensive debugging for SignEase gesture recognition
"""

import requests
import json
import time
import cv2
import mediapipe as mp
import numpy as np

def test_backend_api():
    """Test backend API functionality"""
    print("🔧 Testing Backend API")
    print("-" * 40)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend Health:")
            print(f"   Status: {data['status']}")
            print(f"   Model: {data['model_type']}")
            print(f"   Device: {data['device']}")
            print(f"   Model Loaded: {data['model_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
        # Test model info
        response = requests.get("http://localhost:5000/model-info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Model Info:")
            print(f"   Accuracy: {data['accuracy']}")
            print(f"   Parameters: {data['parameters']:,}")
            print(f"   Classes: {len(data['classes'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend API error: {e}")
        return False

def test_mediapipe_detection():
    """Test MediaPipe hand detection with live camera"""
    print("\n🤖 Testing MediaPipe Hand Detection")
    print("-" * 40)
    
    try:
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        
        # Initialize camera
        cap = cv2.VideoCapture(1)  # Use camera 1 as detected earlier
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        print("✅ MediaPipe initialized")
        print("✅ Camera opened")
        print("\n🎥 Live Hand Detection Test:")
        print("   - Hold your hand in front of the camera")
        print("   - Make different ASL gestures")
        print("   - Press 'q' to quit, 's' to test prediction")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw landmarks if detected
            if results.multi_hand_landmarks:
                detection_count += 1
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks for API test
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Show detection info
                    cv2.putText(frame, f"Hand Detected! ({len(landmarks)} values)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Detections: {detection_count}/{frame_count}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to test prediction", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('MediaPipe Hand Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and results.multi_hand_landmarks:
                # Test prediction with current landmarks
                landmarks = []
                for landmark in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                print(f"\n🧪 Testing prediction with {len(landmarks)} landmarks...")
                test_prediction_api(landmarks)
        
        cap.release()
        cv2.destroyAllWindows()
        
        detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
        print(f"\n📊 Detection Results:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Hands detected: {detection_count}")
        print(f"   Detection rate: {detection_rate:.1f}%")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ MediaPipe test error: {e}")
        return False

def test_prediction_api(landmarks):
    """Test prediction API with real landmarks"""
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json={"landmarks": landmarks},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction: {data['prediction']}")
            print(f"   📊 Confidence: {data['confidence']:.4f}")
            print(f"   ⚡ Time: {data['inference_time_ms']:.2f}ms")
            print(f"   🎮 Device: {data['device_used']}")
            
            if 'top3_predictions' in data:
                print("   🏆 Top 3:")
                for gesture, conf in list(data['top3_predictions'].items())[:3]:
                    print(f"      {gesture}: {conf:.4f}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Prediction API error: {e}")

def test_frontend_mediapipe():
    """Test if frontend MediaPipe setup matches backend expectations"""
    print("\n🌐 Frontend MediaPipe Configuration Check")
    print("-" * 40)
    
    # Check if MediaPipe CDN is accessible
    try:
        import urllib.request
        
        mediapipe_urls = [
            "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js",
            "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands_solution_packed_assets.data",
            "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"
        ]
        
        for url in mediapipe_urls:
            try:
                response = urllib.request.urlopen(url, timeout=5)
                if response.status == 200:
                    print(f"✅ {url.split('/')[-1]} - Accessible")
                else:
                    print(f"❌ {url.split('/')[-1]} - Status {response.status}")
            except Exception as e:
                print(f"❌ {url.split('/')[-1]} - Error: {e}")
                
    except Exception as e:
        print(f"❌ CDN check error: {e}")

def generate_test_landmarks():
    """Generate test landmarks for API testing"""
    print("\n🧪 Testing with Generated Landmarks")
    print("-" * 40)
    
    # Generate realistic hand landmarks (21 points * 3 coordinates)
    landmarks = []
    
    # Wrist (center)
    landmarks.extend([0.5, 0.5, 0.0])
    
    # Thumb (4 points)
    for i in range(4):
        landmarks.extend([0.4 + i*0.02, 0.6 + i*0.02, 0.01])
    
    # Index finger (4 points) - pointing up for letter "A" approximation
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
    
    print(f"Generated {len(landmarks)} landmark values")
    test_prediction_api(landmarks)

def main():
    """Run comprehensive debugging"""
    print("🚀 SignEase Gesture Recognition Debug Tool")
    print("=" * 60)
    
    # Test backend API
    backend_ok = test_backend_api()
    
    if not backend_ok:
        print("\n❌ Backend issues detected. Please fix backend first.")
        return
    
    # Test frontend MediaPipe CDN
    test_frontend_mediapipe()
    
    # Test with generated landmarks
    generate_test_landmarks()
    
    # Test live MediaPipe detection
    print("\n" + "=" * 60)
    print("🎥 Starting Live Hand Detection Test")
    print("This will help identify if the issue is with:")
    print("  • Camera access")
    print("  • MediaPipe hand detection")
    print("  • Landmark extraction")
    print("  • API communication")
    print("\nPress Enter to continue or Ctrl+C to skip...")
    
    try:
        input()
        mediapipe_ok = test_mediapipe_detection()
        
        if mediapipe_ok:
            print("\n✅ MediaPipe detection working!")
        else:
            print("\n❌ MediaPipe detection issues found")
            
    except KeyboardInterrupt:
        print("\nSkipped live detection test")
    
    print("\n" + "=" * 60)
    print("🎯 Debug Summary:")
    print(f"   Backend API: {'✅ Working' if backend_ok else '❌ Issues'}")
    print("   Check the frontend console for MediaPipe errors")
    print("   Make sure to click 'Start Recognition' in the web app")
    print("   Hold your hand clearly in front of the camera")
    print("   Try different lighting conditions")

if __name__ == "__main__":
    main()