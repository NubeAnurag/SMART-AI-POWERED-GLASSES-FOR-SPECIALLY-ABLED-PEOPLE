#!/usr/bin/env python3
"""
Very simple camera test
"""

import cv2
import time

def main():
    print("🔍 Testing basic camera access...")
    
    # Try camera index 0 first
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera index 0 failed, trying index 1...")
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("❌ Both camera indices failed")
        print("💡 Please check camera permissions")
        return
    
    print("✅ Camera opened successfully!")
    print("📹 You should see a camera window now")
    print("⌨️  Press 'q' to quit")
    
    # Try to read a few frames
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame {i+1} read successfully - Size: {frame.shape}")
            cv2.imshow('Camera Test', frame)
            
            # Wait for key press or 1 second
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        else:
            print(f"❌ Failed to read frame {i+1}")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎯 Camera test completed!")

if __name__ == "__main__":
    main() 