#!/usr/bin/env python3
"""
Runner script for Human Detection and Identification System
"""

import sys
import os
import subprocess

def check_camera_permissions():
    """Check and guide user for camera permissions"""
    print("🔍 Checking camera permissions...")
    
    # On macOS, we need to check if camera access is granted
    if sys.platform == "darwin":
        print("📹 On macOS, please ensure camera permissions are granted:")
        print("   1. Go to System Preferences > Security & Privacy > Privacy")
        print("   2. Select 'Camera' from the left sidebar")
        print("   3. Make sure 'Terminal' or 'Python' is checked")
        print("   4. If not, click the lock icon and add it")
        print()
        
        # Try to open system preferences
        try:
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"], 
                         check=False)
            print("✅ Opened System Preferences for camera settings")
        except:
            pass

def main():
    """Main runner function"""
    print("🚀 Human Detection and Identification System")
    print("=" * 50)
    
    # Check camera permissions
    check_camera_permissions()
    
    print("🎯 Starting the system...")
    print("📋 Instructions:")
    print("   • Make sure your face is clearly visible to the camera")
    print("   • Good lighting will improve detection accuracy")
    print("   • Press 'q' to quit, 's' to save face, 'h' for help")
    print()
    
    # Try to run the system
    try:
        from human_detection_system_simple import main as run_system
        run_system()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python3 setup.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure all dependencies are installed")

if __name__ == "__main__":
    main() 