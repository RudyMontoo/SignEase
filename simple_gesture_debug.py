#!/usr/bin/env python3
"""
Simple Gesture Debug - Direct test with your camera
"""

import cv2
import mediapipe as mp
import requests
import json
import time

def main():
    print("🔍 Simple Gesture Debug")
    print("=" * 50)
    print("This will show you exactly what's happening:")
    print("1. Camera feed with hand detection")
    print("2. Real-time API calls to backend")
    print("3. Prediction results")
    print()
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,  # Lower for easier detection
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize camera (use camera 1 as detected earlier)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return
    
    print("✅ Camera opened")
    print("✅ MediaPipe initialized")
    print()
    print("Controls:")
    print("  SPACE - Send current gesture to API")
    print("  Q - Quit")
    print("  A - Auto-predict mode (sends every 2 seconds)")
    print()
    
    auto_predict = False
    last_auto_time = 0
    prediction_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks and info
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Show detection info
                cv2.putText(frame, "Hand Detected!", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Landmarks: {len(hand_landmarks.landmark)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Auto-predict mode
                if auto_predict and time.time() - last_auto_time > 2:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    print(f"\n🤖 Auto-prediction #{prediction_count + 1}")
                    test_prediction(landmarks)
                    prediction_count += 1
                    last_auto_time = time.time()
        else:
            cv2.putText(frame, "No hand detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Show your hand to the camera", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show mode
        mode_text = "AUTO" if auto_predict else "MANUAL"
        cv2.putText(frame, f"Mode: {mode_text}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show controls
        cv2.putText(frame, "SPACE=Test, A=Auto, Q=Quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Gesture Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and results.multi_hand_landmarks:
            # Manual prediction
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            print(f"\n🎯 Manual prediction #{prediction_count + 1}")
            test_prediction(landmarks)
            prediction_count += 1
        elif key == ord('a'):
            auto_predict = not auto_predict
            print(f"\n🔄 Auto-predict mode: {'ON' if auto_predict else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Debug completed! Total predictions: {prediction_count}")

def test_prediction(landmarks):
    """Test prediction with landmarks"""
    try:
        print(f"   📡 Sending {len(landmarks)} landmarks to API...")
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:5000/predict",
            json={"landmarks": landmarks},
            timeout=5
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS!")
            print(f"      Prediction: {data['prediction']}")
            print(f"      Confidence: {data['confidence']:.4f} ({data['confidence']*100:.1f}%)")
            print(f"      Response time: {(end_time - start_time)*1000:.1f}ms")
            print(f"      Device: {data['device_used']}")
            
            # Show top 3
            if 'top3_predictions' in data:
                print(f"      Top 3:")
                for gesture, conf in list(data['top3_predictions'].items())[:3]:
                    print(f"        {gesture}: {conf:.4f}")
        else:
            print(f"   ❌ API Error: HTTP {response.status_code}")
            print(f"      Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")

if __name__ == "__main__":
    main()