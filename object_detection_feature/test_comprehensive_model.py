#!/usr/bin/env python3
"""
Test Comprehensive Multi-Object Detection Model
Test the trained model on webcam
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_comprehensive_model():
    """Test the comprehensive multi-object detection model"""
    print("🧪 Testing Comprehensive Multi-Object Detection Model...")
    print("=" * 60)
    
    # Check for trained model
    model_path = "comprehensive_training/comprehensive_v1/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("💡 Train the model first: python3 start_comprehensive_training.py")
        return False
    
    try:
        # Load the trained model
        model = YOLO(model_path)
        print("✅ Comprehensive model loaded successfully!")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return False
        
        print("📹 Testing comprehensive detection...")
        print("🎮 Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, verbose=False, conf=0.4)
            
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
                        
                        # Color coding based on confidence
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
            cv2.putText(frame, "Comprehensive Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Comprehensive Multi-Object Detection', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"comprehensive_test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Test Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {detection_count}")
        print(f"   Average detections per frame: {detection_count/max(frame_count, 1):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_comprehensive_model()
