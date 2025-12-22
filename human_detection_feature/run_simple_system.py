#!/usr/bin/env python3
"""
Simple Face Recognition System Runner
"""

import sys
import os

def main():
    """Main runner function"""
    print("🎯 Simple Face Recognition System")
    print("=" * 50)
    
    print("🎯 This system will:")
    print("   • Take photos of people (front, left, right)")
    print("   • Let you input their name, age, and relationship")
    print("   • Recognize them when they appear")
    print("   • Speak information about them using English Siri-like male voice")
    print()
    print("📋 Controls:")
    print("   • Press 'a' to add a new person")
    print("   • Press 'p' to speak info about recognized person")
    print("   • Press 'q' to quit, 'h' for help")
    print()
    
    # Try to run the system
    try:
        from simple_face_system import main as run_system
        run_system()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python3 setup.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure all dependencies are installed")

if __name__ == "__main__":
    main() 