#!/usr/bin/env python3
"""
===============================================================================
FEATURE: OBJECT/ENVIRONMENT ANALYSIS (Feature 3) - Advanced Detector
===============================================================================
This file belongs to the Object Detection and Environment Analysis feature module.

WORK:
- Advanced YOLOv8 object detector with configurable model sizes
- Command-line argument support for customization
- Multiple model size options (nano, small, medium, large, xlarge)
- Configurable confidence thresholds
- Real-time webcam processing with performance metrics
- Bounding box visualization with confidence scores
- FPS calculation and display

KEY CLASS: AdvancedObjectDetector

KEY FEATURES:
- Flexible model selection based on speed/accuracy needs
- Command-line interface for easy configuration
- Performance monitoring with FPS tracking
- Customizable detection thresholds

KEY METHODS:
- load_model(): Loads YOLOv8 model of specified size
- detect_objects(): Runs object detection on frame
- draw_detections(): Visualizes detections with bounding boxes
- calculate_fps(): Tracks processing speed

COMMAND-LINE USAGE:
    python advanced_detector.py --model s --confidence 0.5 --webcam 0

ARGUMENTS:
- --model: YOLO model size (n, s, m, l, x)
- --confidence: Detection confidence threshold (0.0-1.0)
- --webcam: Webcam device index

This version provides more control and customization options.

Author: DRDO Project
===============================================================================
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os

class AdvancedObjectDetector:
    def __init__(self, model_size='n', confidence=0.5, webcam_index=0):
        """
        Initialize the advanced YOLO object detector
        
        Args:
            model_size (str): YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence (float): Detection confidence threshold (0.0-1.0)
            webcam_index (int): Webcam device index
        """
        self.confidence = confidence
        self.model_size = model_size
        
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print(f"YOLOv8{model_size} model loaded successfully!")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(webcam_index)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open webcam at index {webcam_index}")
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_count = 0
        
        # Colors for different object classes
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        
        # Detection history for smoothing
        self.detection_history = {}
        
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on the frame with enhanced UI"""
        current_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    # Only show detections above confidence threshold
                    if conf > self.confidence:
                        current_detections.append(class_name)
                        
                        # Get color for this class
                        color = self.colors[cls]
                        color = (int(color[0]), int(color[1]), int(color[2]))
                        
                        # Draw bounding box with thickness based on confidence
                        thickness = max(1, int(conf * 3))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Create label text
                        label = f"{class_name}: {conf:.2f}"
                        
                        # Get text size
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Draw label background
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Update detection count
        self.detection_count = len(current_detections)
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw information text
        info_lines = [
            f"Model: YOLOv8{self.model_size.upper()}",
            f"FPS: {self.fps:.1f}",
            f"Confidence: {self.confidence:.2f}",
            f"Detections: {self.detection_count}",
            f"Resolution: {width}x{height}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_controls_help(self, frame):
        """Draw controls help at the bottom"""
        height = frame.shape[0]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 80), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        controls = [
            "Controls: 'q'=quit, 's'=screenshot, 'c'=change confidence, 'm'=change model"
        ]
        
        for i, line in enumerate(controls):
            y_pos = height - 60 + i * 20
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def change_confidence(self):
        """Change confidence threshold"""
        print(f"Current confidence: {self.confidence}")
        try:
            new_conf = float(input("Enter new confidence (0.1-1.0): "))
            if 0.1 <= new_conf <= 1.0:
                self.confidence = new_conf
                print(f"Confidence changed to: {self.confidence}")
            else:
                print("Confidence must be between 0.1 and 1.0")
        except ValueError:
            print("Invalid input. Confidence unchanged.")
    
    def change_model(self):
        """Change YOLO model size"""
        models = {'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large', 'x': 'xlarge'}
        print(f"Current model: YOLOv8{self.model_size.upper()} ({models[self.model_size]})")
        print("Available models: n(nano), s(small), m(medium), l(large), x(xlarge)")
        
        try:
            new_size = input("Enter new model size: ").lower()
            if new_size in models:
                print(f"Loading YOLOv8{new_size} model...")
                self.model = YOLO(f'yolov8{new_size}.pt')
                self.model_size = new_size
                print(f"Model changed to: YOLOv8{new_size.upper()}")
            else:
                print("Invalid model size. Model unchanged.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def run(self):
        """Main loop for object detection"""
        print("Starting advanced object detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'c' - Change confidence threshold")
        print("  'm' - Change model size")
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Draw detections on frame
                frame = self.draw_detections(frame, results)
                
                # Calculate and display FPS
                self.calculate_fps()
                
                # Draw information panel
                self.draw_info_panel(frame)
                
                # Draw controls help
                self.draw_controls_help(frame)
                
                # Show the frame
                cv2.imshow('Advanced YOLO Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
                elif key == ord('c'):
                    self.change_confidence()
                elif key == ord('m'):
                    self.change_model()
                    
        except KeyboardInterrupt:
            print("\nStopping object detection...")
        
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            print("Object detection stopped.")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced YOLO Object Detection')
    parser.add_argument('--model', '-m', default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Detection confidence threshold (0.1-1.0)')
    parser.add_argument('--webcam', '-w', type=int, default=0,
                       help='Webcam device index')
    
    args = parser.parse_args()
    
    try:
        detector = AdvancedObjectDetector(
            model_size=args.model,
            confidence=args.confidence,
            webcam_index=args.webcam
        )
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and accessible.")

if __name__ == "__main__":
    main() 