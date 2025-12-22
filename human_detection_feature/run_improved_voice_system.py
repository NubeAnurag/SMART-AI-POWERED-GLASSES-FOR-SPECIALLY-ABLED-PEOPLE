#!/usr/bin/env python3
"""
Runner script for Improved Voice-Enabled Human Detection and Identification System
"""

import sys
import os
import subprocess

def check_audio_permissions():
    """Check and guide user for audio permissions"""
    print("🔍 Checking audio permissions...")
    
    # On macOS, we need to check if microphone access is granted
    if sys.platform == "darwin":
        print("🎤 On macOS, please ensure microphone permissions are granted:")
        print("   1. Go to System Preferences > Security & Privacy > Privacy")
        print("   2. Select 'Microphone' from the left sidebar")
        print("   3. Make sure 'Terminal' or 'Python' is checked")
        print("   4. If not, click the lock icon and add it")
        print()
        
        # Try to open system preferences
        try:
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"], 
                         check=False)
            print("✅ Opened System Preferences for microphone settings")
        except:
            pass

def main():
    """Main runner function"""
    print("🎤 Improved Voice-Enabled Human Detection and Identification System")
    print("=" * 70)
    
    # Check audio permissions
    check_audio_permissions()
    
    print("🎯 Starting the improved voice-enabled system...")
    print("📋 Instructions:")
    print("   • Make sure your face is clearly visible to the camera")
    print("   • Good lighting will improve detection accuracy")
    print("   • Press 'v' to START recording (continuous)")
    print("   • Press 'p' to STOP recording and play audio")
    print("   • Press 'q' to quit, 's' to save face, 'h' for help")
    print()
    print("🎤 Voice Recording Flow:")
    print("   1. Look at camera (get recognized)")
    print("   2. Press 'v' → See '🎙️  Listening...' (red text)")
    print("   3. Speak your details (no time limit)")
    print("   4. Press 'p' → Recording stops, audio plays")
    print()
    
    # Try to run the system
    try:
        from improved_voice_system import main as run_system
        run_system()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python3 setup.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure all dependencies are installed")

if __name__ == "__main__":
    main() 