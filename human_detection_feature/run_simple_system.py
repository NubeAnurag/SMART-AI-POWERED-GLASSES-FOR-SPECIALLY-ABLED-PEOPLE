#!/usr/bin/env python3
"""
===============================================================================
FEATURE: HUMAN DETECTION AND IDENTIFICATION (Feature 2) - Runner Script
===============================================================================
This file belongs to the Human Detection and Identification feature module.

WORK:
- Entry point script to launch the simple face recognition system
- Provides user-friendly startup messages and instructions
- Handles import errors gracefully with helpful error messages
- Runs the SimpleFaceRecognitionSystem main function

PURPOSE:
This is a convenience wrapper script that makes it easy to start the
human detection feature independently. It provides clear instructions
and error handling.

USAGE:
    python3 run_simple_system.py

This script can be run standalone or is automatically called by the
unified_drdo_system.py when Feature 2 is selected.

Author: DRDO Project
===============================================================================
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