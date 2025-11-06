#!/usr/bin/env python3
"""
Quick Gesture Test - Test if your gestures work with the backend
"""

import cv2
import mediapipe as mp
import requests
import json

def main():
    print("🎯 Quick ASL Gesture Test")
    print("=" * 40)
    print("This will test your gestures directly with the backend")
    print("Make sure the backend is running on localhost:5000")
    print()
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3,  # Lower threshold for better detection
        min_tracking_confidence=0.3
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize camera
    cap = cv2.VideoCapture(1)  # Use camera 1 as detected
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    print("📹 Camera opened - Make ASL gestures!")
    print("Controls:")
    print("  SPACE - Test current gesture")
    print("  Q - Quit")
    print("  R - Reset")
    
    last_prediction = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks and get prediction
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Show detection
                cv2.putText(frame, "Hand Detected! Press SPACE to test", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if last_prediction:
                    cv2.putText(frame, f"Last: {last_prediction}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show instructions
        cv2.putText(frame, "SPACE=Test, Q=Quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('ASL Gesture Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and results.multi_hand_landmarks:  # Space key
            # Test prediction
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            try:
                response = requests.post(
                    "http://localhost:5000/predict",
                    json={"landmarks": landmarks},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data['prediction']
                    confidence = data['confidence']
                    last_prediction = f"{prediction} ({confidence:.2f})"
                    
                    print(f"🎯 Gesture: {prediction} (Confidence: {confidence:.4f})")
                    
                    # Show top 3
                    if 'top3_predictions' in data:
                        print("   Top 3:")
                        for gesture, conf in list(data['top3_predictions'].items())[:3]:
                            print(f"     {gesture}: {conf:.4f}")
                else:
                    print(f"❌ API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Request failed: {e}")
        elif key == ord('r'):
            last_prediction = None
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Test completed!")

if __name__ == "__main__":
    main()