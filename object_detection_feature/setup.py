#!/usr/bin/env python3
"""
===============================================================================
FEATURE: OBJECT/ENVIRONMENT ANALYSIS (Feature 3) - Setup Script
===============================================================================
This file belongs to the Object Detection and Environment Analysis feature module.

WORK:
- Automated setup script for the object detection feature
- Checks Python version compatibility (requires 3.8+)
- Installs all required dependencies from requirements.txt
- Tests imports of critical libraries (OpenCV, NumPy, Ultralytics, PyTorch)
- Downloads YOLOv8 model file on first run
- Tests webcam accessibility and permissions
- Provides helpful error messages and troubleshooting tips

PURPOSE:
This script should be run once before using the object detection feature
to ensure all dependencies are installed and the system is properly configured.

FUNCTIONS:
- check_python_version(): Verifies Python 3.8+ is installed
- install_requirements(): Installs packages from requirements.txt
- test_imports(): Verifies all critical libraries can be imported
- check_webcam(): Tests camera access and permissions
- download_yolo_model(): Downloads yolov8n.pt model file

USAGE:
    python3 setup.py

This ensures a smooth first-run experience for the object detection feature.

Author: DRDO Project
===============================================================================
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🔍 Testing imports...")
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
        
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_webcam():
    """Test webcam access"""
    print("\n📹 Testing webcam access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Webcam is accessible")
                return True
            else:
                print("⚠️  Webcam opened but couldn't read frame")
                return False
        else:
            print("❌ Could not open webcam")
            return False
    except Exception as e:
        print(f"❌ Webcam test error: {e}")
        return False

def download_yolo_model():
    """Download YOLO model on first run"""
    print("\n🤖 Downloading YOLO model (this may take a few minutes)...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading YOLO model: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up YOLO Object Detection Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Download YOLO model
    if not download_yolo_model():
        sys.exit(1)
    
    # Test webcam
    webcam_ok = check_webcam()
    if not webcam_ok:
        print("⚠️  Webcam test failed. You may need to:")
        print("   - Connect a webcam")
        print("   - Grant camera permissions")
        print("   - Close other applications using the webcam")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("  Basic version: python main.py")
    print("  Advanced version: python advanced_detector.py")
    print("\nFor help: python advanced_detector.py --help")

if __name__ == "__main__":
    main() 