#!/usr/bin/env python3
"""
Test script for Human Detection and Identification System
"""

import sys
import cv2
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import face_recognition
        print("✅ face_recognition")
    except ImportError as e:
        print(f"❌ face_recognition: {e}")
        return False
    
    try:
        from deepface import DeepFace
        print("✅ deepface")
    except ImportError as e:
        print(f"❌ deepface: {e}")
        return False
    
    try:
        import dlib
        print("✅ dlib")
    except ImportError as e:
        print(f"❌ dlib: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ tensorflow")
    except ImportError as e:
        print(f"❌ tensorflow: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        return False
    
    return True

def test_webcam():
    """Test if webcam can be accessed"""
    print("\n📹 Testing webcam...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame from webcam")
        cap.release()
        return False
    
    print(f"✅ Webcam working - Frame size: {frame.shape}")
    cap.release()
    return True

def test_face_detection():
    """Test face detection functionality"""
    print("\n👤 Testing face detection...")
    
    try:
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test OpenCV face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("❌ Could not load face cascade classifier")
            return False
        
        print("✅ Face cascade classifier loaded")
        
        # Test face_recognition
        face_locations = face_recognition.face_locations(test_image)
        print("✅ face_recognition module working")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import KNOWN_PERSONS, SYSTEM_CONFIG, DISPLAY_CONFIG
        print(f"✅ Configuration loaded - {len(KNOWN_PERSONS)} known persons")
        return True
    except ImportError as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Human Detection and Identification System")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Webcam", test_webcam),
        ("Face Detection", test_face_detection),
        ("Configuration", test_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test passed")
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the system, run:")
        print("   python human_detection_system.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Try running the setup script:")
        print("   python setup.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 