#!/usr/bin/env python3
"""
Setup script for Human Detection and Identification System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        "opencv-python",
        "face-recognition", 
        "dlib",
        "numpy",
        "tensorflow",
        "keras",
        "Pillow",
        "deepface",
        "mtcnn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are available!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["known_faces", "logs", "saved_faces"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up Human Detection and Identification System")
    print("=" * 60)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Some dependencies are missing")
        return
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n🎯 To run the system:")
    print("   python human_detection_system.py")
    print("\n📖 Features:")
    print("   - Human detection using webcam")
    print("   - Gender classification (Male/Female)")
    print("   - Face recognition with known persons")
    print("   - Age estimation")
    print("   - Emotion detection")
    print("   - Real-time display with bounding boxes")
    print("\n⌨️  Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save current face")
    print("   - Press 'h' for help")

if __name__ == "__main__":
    main() 