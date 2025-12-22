#!/usr/bin/env python3
"""
Simple camera test script
"""

import cv2
import sys

def test_camera(index):
    """Test camera at given index"""
    print(f"Testing camera index {index}...")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"❌ Camera index {index} failed")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Could not read frame from camera {index}")
        cap.release()
        return False
    
    print(f"✅ Camera index {index} working - Frame size: {frame.shape}")
    cap.release()
    return True

def main():
    """Test different camera indices"""
    print("🔍 Testing camera access...")
    
    # Test common camera indices
    for i in range(5):
        if test_camera(i):
            print(f"\n🎯 Use camera index {i} in your config.py")
            print("Update SYSTEM_CONFIG['camera_index'] = {i}")
            return i
    
    print("\n❌ No camera found. Please check:")
    print("1. Camera is connected")
    print("2. Camera permissions are granted")
    print("3. No other app is using the camera")
    return None

if __name__ == "__main__":
    main() 