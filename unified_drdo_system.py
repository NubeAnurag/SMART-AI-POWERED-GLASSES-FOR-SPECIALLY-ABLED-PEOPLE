#!/usr/bin/env python3
"""
Unified DRDO System - Combining all three features:
1. OCR Text Recognition
2. Human Detection and Identification
3. Object/Environment Analysis (YOLOv8)
"""

import sys
import os
import cv2

# Add paths to import modules from different folders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'text_ocr_feature'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'human_detection_feature'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'object_detection_feature'))

def display_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print(" "*15 + "UNIFIED DRDO SYSTEM")
    print("="*60)
    print("\nAvailable Features:")
    print("  1. OCR Text Recognition (EasyOCR + Tesseract)")
    print("  2. Human Detection and Identification")
    print("  3. Object/Environment Analysis (YOLOv8)")
    print("  0. Exit")
    print("\n" + "="*60)
    print("Select a feature (1-3) or 0 to exit: ", end="")

def run_ocr_feature():
    """Run the OCR text recognition feature"""
    print("\n" + "="*60)
    print("Starting OCR Text Recognition Feature...")
    print("="*60)
    print("\nInstructions:")
    print("  - Press SPACE to capture and process text")
    print("  - Press ESC to exit and return to main menu")
    print("="*60 + "\n")
    
    try:
        # Import and run OCR feature
        from camera_ocr_live_ocr import main as ocr_main
        ocr_main()
    except ImportError as e:
        print(f"❌ Error importing OCR module: {e}")
        print("💡 Make sure you're in the correct directory and dependencies are installed")
    except Exception as e:
        print(f"❌ Error running OCR feature: {e}")
    finally:
        # Clean up any open windows
        cv2.destroyAllWindows()
        print("\n✅ OCR feature closed. Returning to main menu...")

def run_human_detection_feature():
    """Run the human detection and identification feature"""
    print("\n" + "="*60)
    print("Starting Human Detection and Identification Feature...")
    print("="*60)
    print("\nInstructions:")
    print("  - Press 'a' to add a new person")
    print("  - Press 'p' to speak info about recognized person")
    print("  - Press 'q' to exit and return to main menu")
    print("  - Press 'h' for help")
    print("="*60 + "\n")
    
    original_dir = os.getcwd()
    try:
        # Change to the human detection directory to access config and other files
        human_detection_dir = os.path.join(os.path.dirname(__file__), 'human_detection_feature')
        
        if os.path.exists(human_detection_dir):
            os.chdir(human_detection_dir)
        
        # Import and run human detection feature
        from simple_face_system import SimpleFaceRecognitionSystem
        
        system = SimpleFaceRecognitionSystem()
        system.start_detection()
        system.cleanup()
        
    except ImportError as e:
        print(f"❌ Error importing Human Detection module: {e}")
        print("💡 Make sure you're in the correct directory and dependencies are installed")
    except Exception as e:
        print(f"❌ Error running Human Detection feature: {e}")
    finally:
        # Return to original directory
        os.chdir(original_dir)
        # Clean up any open windows
        cv2.destroyAllWindows()
        print("\n✅ Human Detection feature closed. Returning to main menu...")

def run_object_detection_feature():
    """Run the object/environment analysis feature (YOLOv8)"""
    print("\n" + "="*60)
    print("Starting Object/Environment Analysis Feature (YOLOv8)...")
    print("="*60)
    print("\nInstructions:")
    print("  - Press 'q' to quit and return to main menu")
    print("  - Press 's' to save screenshot")
    print("  - Press 'p' to speak detections")
    print("  - Press 'c' to change confidence")
    print("  - Press 't' to toggle tracking")
    print("  - Press 'm' to toggle smoothing")
    print("  - Press 'b' to toggle confidence boost")
    print("="*60 + "\n")
    
    original_dir = os.getcwd()
    try:
        # Change to the object detection directory to access model files
        object_detection_dir = os.path.join(os.path.dirname(__file__), 'object_detection_feature')
        
        if os.path.exists(object_detection_dir):
            os.chdir(object_detection_dir)
        
        # Import and run object detection feature
        from main import ObjectDetector
        
        detector = ObjectDetector()
        detector.run()
        
    except ImportError as e:
        print(f"❌ Error importing Object Detection module: {e}")
        print("💡 Trying alternative import (smart_detector)...")
        try:
            from smart_detector import SmartObjectDetector
            detector = SmartObjectDetector()
            detector.run()
        except Exception as e2:
            print(f"❌ Error with alternative import: {e2}")
            print("💡 Make sure you're in the correct directory and dependencies are installed")
    except Exception as e:
        print(f"❌ Error running Object Detection feature: {e}")
    finally:
        # Return to original directory
        os.chdir(original_dir)
        # Clean up any open windows
        cv2.destroyAllWindows()
        print("\n✅ Object Detection feature closed. Returning to main menu...")

def main():
    """Main function with switch case menu"""
    print("\n" + "="*60)
    print(" "*15 + "WELCOME TO UNIFIED DRDO SYSTEM")
    print("="*60)
    print("\nThis system combines three powerful features:")
    print("  1. OCR Text Recognition - Extract text from images/video")
    print("  2. Human Detection - Recognize and identify people")
    print("  3. Object Detection - Analyze objects and environment using YOLOv8")
    
    while True:
        try:
            display_menu()
            choice = input().strip()
            
            if choice == '0':
                print("\n" + "="*60)
                print("Thank you for using Unified DRDO System!")
                print("="*60 + "\n")
                break
            
            elif choice == '1':
                run_ocr_feature()
            
            elif choice == '2':
                run_human_detection_feature()
            
            elif choice == '3':
                run_object_detection_feature()
            
            else:
                print("\n❌ Invalid choice! Please enter 1, 2, 3, or 0 to exit.")
                print("Press Enter to continue...")
                input()
        
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("Program interrupted by user. Exiting...")
            print("="*60 + "\n")
            cv2.destroyAllWindows()
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Press Enter to continue...")
            input()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

