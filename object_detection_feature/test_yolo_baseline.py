#!/usr/bin/env python3
"""
Test YOLO Baseline Performance
Test what standard YOLO can detect
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_yolo_baseline():
    """Test YOLO baseline performance"""
    print("🧪 Testing YOLO Baseline Performance")
    print("=" * 60)
    
    try:
        # Load YOLO model
        print("🔄 Loading YOLOv8s model...")
        model = YOLO("yolov8s.pt")
        print("✅ YOLOv8s model loaded successfully!")
        
        # Show COCO classes
        print(f"\n📊 COCO Classes Available ({len(model.names)}):")
        for i, name in enumerate(model.names.values()):
            if i < 20:  # Show first 20
                print(f"   {i:2d}: {name}")
            elif i == 20:
                print("   ... (and more)")
                break
        
        # Check for objects we want to detect
        target_objects = ["carpet", "printer", "keyboard", "monitor", "pen", "broom"]
        print(f"\n🎯 Checking for target objects:")
        for obj in target_objects:
            if obj in model.names.values():
                print(f"   ✅ {obj} - Available in COCO")
            else:
                print(f"   ❌ {obj} - NOT in COCO (needs custom training)")
        
        # Try webcam with better error handling
        print(f"\n📹 Testing Webcam Detection...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Webcam not accessible. Trying alternative...")
            
            # Try different webcam indices
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"✅ Webcam {i} opened successfully!")
                    break
                cap.release()
            
            if not cap.isOpened():
                print("❌ No webcam available. Creating test image instead...")
                return create_test_image_detection(model)
        
        print("✅ Webcam initialized!")
        print("🎮 Controls: Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read frame from webcam")
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, verbose=False, conf=0.25)
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        
                        # Color coding
                        if conf > 0.7:
                            color = (0, 255, 0)  # Green
                        elif conf > 0.5:
                            color = (0, 165, 255)  # Orange
                        else:
                            color = (0, 0, 255)  # Red
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        detection_count += 1
            
            # Add info overlay
            cv2.putText(frame, "YOLO Baseline Test", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('YOLO Baseline Test', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"yolo_baseline_test_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 YOLO Baseline Test Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {detection_count}")
        print(f"   Average detections per frame: {detection_count/max(frame_count, 1):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def create_test_image_detection(model):
    """Create a test image and run detection on it"""
    print("🖼️  Creating test image for detection...")
    
    # Create a simple test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(test_image, (100, 100), (300, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(test_image, (350, 150), (550, 250), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(test_image, (200, 300), (400, 400), (0, 0, 255), -1)  # Red rectangle
    
    # Add text
    cv2.putText(test_image, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_image, "No webcam available", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Run detection on test image
    results = model(test_image, verbose=False, conf=0.25)
    
    print("📊 Detection Results on Test Image:")
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                print(f"   Detected: {class_name} (confidence: {conf:.2f})")
        else:
            print("   No objects detected in test image")
    
    # Save test image
    cv2.imwrite("yolo_test_image.jpg", test_image)
    print("💾 Test image saved: yolo_test_image.jpg")
    
    return True

def main():
    """Main function"""
    print("🏠 YOLO Baseline Performance Test")
    print("=" * 60)
    
    success = test_yolo_baseline()
    
    if success:
        print(f"\n🎯 Baseline Test Complete!")
        print(f"💡 This shows what standard YOLO can detect")
        print(f"🚀 Next: Collect data for custom training")
    else:
        print(f"❌ Baseline test failed")

if __name__ == "__main__":
    main() 