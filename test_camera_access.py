#!/usr/bin/env python3
"""
Camera Access Test for SignEase
Test camera availability and permissions
"""

import cv2
import numpy as np
import time
import sys

def test_camera_access():
    """Test camera access and display feed"""
    print("🎥 SignEase Camera Access Test")
    print("=" * 50)
    
    # Try to access camera
    print("📹 Attempting to access camera...")
    
    try:
        # Try different camera indices
        camera_found = False
        cap = None
        
        for camera_id in range(3):  # Try cameras 0, 1, 2
            print(f"   Trying camera {camera_id}...")
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                # Test if we can read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {camera_id} is working!")
                    camera_found = True
                    break
                else:
                    cap.release()
            else:
                if cap:
                    cap.release()
        
        if not camera_found:
            print("❌ No working camera found")
            print("\n🔧 Troubleshooting:")
            print("   • Make sure your camera is connected")
            print("   • Close other applications using the camera")
            print("   • Check camera privacy settings")
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Camera Properties:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        print(f"\n🎬 Starting camera feed...")
        print("   Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Calculate actual FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"   Actual FPS: {actual_fps:.1f}")
            
            # Add text overlay
            cv2.putText(frame, f"SignEase Camera Test", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Press 'q' to quit", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('SignEase Camera Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting camera test")
                break
            elif key == ord('s'):
                filename = f"camera_test_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Camera test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def check_opencv_version():
    """Check OpenCV version and capabilities"""
    print(f"\n🔧 OpenCV Information:")
    print(f"   Version: {cv2.__version__}")
    
    # Check available backends
    backends = []
    backend_names = {
        cv2.CAP_DSHOW: "DirectShow (Windows)",
        cv2.CAP_MSMF: "Media Foundation (Windows)",
        cv2.CAP_V4L2: "Video4Linux2 (Linux)",
        cv2.CAP_GSTREAMER: "GStreamer"
    }
    
    for backend_id, name in backend_names.items():
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                backends.append(name)
                cap.release()
        except:
            pass
    
    print(f"   Available backends: {', '.join(backends) if backends else 'None detected'}")

def main():
    """Main function"""
    try:
        check_opencv_version()
        
        if test_camera_access():
            print("\n🎉 Camera is ready for SignEase!")
            print("   You can now use the web application with confidence.")
        else:
            print("\n⚠️ Camera issues detected.")
            print("   Please resolve camera access before using SignEase.")
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()