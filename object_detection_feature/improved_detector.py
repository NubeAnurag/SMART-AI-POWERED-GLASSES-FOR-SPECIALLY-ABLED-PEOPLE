#!/usr/bin/env python3
"""
===============================================================================
FEATURE: OBJECT/ENVIRONMENT ANALYSIS (Feature 3) - Improved Detector
===============================================================================
This file belongs to the Object Detection and Environment Analysis feature module.

WORK:
- Improved YOLOv8 object detector with enhanced features
- Temporal smoothing using frame history (deque)
- Confidence filtering and thresholding
- Multiple model size support via command-line arguments
- Real-time processing with FPS tracking
- Screenshot capability
- Text-to-speech integration for detected objects

KEY CLASS: ImprovedObjectDetector

KEY FEATURES:
- Frame history buffer for temporal smoothing
- Configurable model sizes and confidence thresholds
- Enhanced detection stability with frame averaging
- Better false positive filtering

KEY METHODS:
- initialize_webcam(): Sets up camera with error handling
- detect_objects(): Runs YOLO detection with filtering
- smooth_detections(): Applies temporal smoothing to reduce flickering
- draw_results(): Visualizes detections on frame

This is an improved version with better stability and fewer false positives.

Author: DRDO Project
===============================================================================
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import subprocess
import argparse
from collections import deque
import threading

class ImprovedObjectDetector:
    def __init__(self, model_size='s', confidence=0.4, nms_threshold=0.4, max_detections=20):
        """
        Initialize the improved YOLO object detector
        
        Args:
            model_size (str): YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence (float): Detection confidence threshold (0.0-1.0)
            nms_threshold (float): Non-maximum suppression threshold
            max_detections (int): Maximum number of detections to process
        """
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.model_size = model_size
        
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print(f"YOLOv8{model_size} model loaded successfully!")
        
        # Initialize webcam
        self.cap = None
        self.initialize_webcam()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Detection tracking for stability
        self.detection_history = deque(maxlen=10)  # Track last 10 frames
        self.stable_detections = {}
        
        # Colors for different object classes
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        
        # Speech tracking
        self.last_spoken_objects = set()
        self.speech_cooldown = 0
        
        # Advanced settings
        self.enable_tracking = True
        self.enable_smoothing = True
        self.enable_confidence_boost = True
        
    def speak_text(self, text):
        """Speak the given text using macOS say command"""
        try:
            subprocess.run(['say', text], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Could not speak: {text}")
    
    def initialize_webcam(self):
        """Initialize webcam with multiple attempts"""
        print("Initializing webcam...")
        
        for i in range(3):
            print(f"Trying webcam index {i}...")
            self.cap = cv2.VideoCapture(i)
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"✅ Webcam {i} initialized successfully!")
                    
                    # Set higher resolution for better detection
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    ret, frame = self.cap.read()
                    if ret:
                        return
                    else:
                        print(f"Webcam {i} opened but can't read frames")
                        self.cap.release()
                else:
                    print(f"Webcam {i} opened but frame is None")
                    self.cap.release()
            else:
                print(f"Could not open webcam {i}")
        
        raise ValueError("Could not initialize any webcam!")
    
    def apply_confidence_boost(self, detections):
        """Apply confidence boosting for common objects"""
        if not self.enable_confidence_boost:
            return detections
            
        # Boost confidence for common objects
        common_objects = {
            'person': 0.1,      # Boost person detection
            'chair': 0.05,      # Boost chair detection
            'laptop': 0.08,     # Boost laptop detection
            'cell phone': 0.1,  # Boost phone detection
            'cup': 0.05,        # Boost cup detection
            'bottle': 0.05,     # Boost bottle detection
            'book': 0.05,       # Boost book detection
            'tv': 0.08,         # Boost TV detection
            'remote': 0.05,     # Boost remote detection
            'keyboard': 0.05,   # Boost keyboard detection
            'mouse': 0.05,      # Boost mouse detection
        }
        
        boosted_detections = []
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Apply boost if object is in common list
            if class_name in common_objects:
                confidence += common_objects[class_name]
                confidence = min(confidence, 1.0)  # Cap at 1.0
                detection['confidence'] = confidence
            
            boosted_detections.append(detection)
        
        return boosted_detections
    
    def apply_temporal_smoothing(self, detections):
        """Apply temporal smoothing to reduce flickering"""
        if not self.enable_smoothing:
            return detections
        
        # Add current detections to history
        self.detection_history.append(detections)
        
        # If we don't have enough history, return current detections
        if len(self.detection_history) < 3:
            return detections
        
        # Count occurrences of each object in recent frames
        object_counts = {}
        for frame_detections in list(self.detection_history)[-5:]:  # Last 5 frames
            for det in frame_detections:
                class_name = det['class_name']
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
        
        # Only keep objects that appear in at least 2 of the last 5 frames
        stable_detections = []
        for detection in detections:
            class_name = detection['class_name']
            if object_counts.get(class_name, 0) >= 2:
                stable_detections.append(detection)
        
        return stable_detections
    
    def filter_overlapping_detections(self, detections):
        """Filter out overlapping detections using IoU"""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        for i, det1 in enumerate(detections):
            keep = True
            for j, det2 in enumerate(detections):
                if i != j:
                    # Calculate IoU
                    iou = self.calculate_iou(det1['bbox'], det2['bbox'])
                    if iou > 0.5:  # If overlap is more than 50%
                        # Keep the one with higher confidence
                        if det1['confidence'] < det2['confidence']:
                            keep = False
                            break
            
            if keep:
                filtered.append(det1)
        
        return filtered
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_detections(self, results):
        """Process and improve detections"""
        all_detections = []
        
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
                    
                    # Only process detections above confidence threshold
                    if conf > self.confidence:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_name': class_name,
                            'class_id': cls
                        }
                        all_detections.append(detection)
        
        # Apply improvements
        if self.enable_confidence_boost:
            all_detections = self.apply_confidence_boost(all_detections)
        
        # Filter overlapping detections
        all_detections = self.filter_overlapping_detections(all_detections)
        
        # Apply temporal smoothing
        if self.enable_smoothing:
            all_detections = self.apply_temporal_smoothing(all_detections)
        
        # Sort by confidence and limit detections
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        all_detections = all_detections[:self.max_detections]
        
        return all_detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame"""
        detected_objects = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            detected_objects.append(class_name)
            
            # Get color for this class
            color = self.colors[class_id]
            color = (int(color[0]), int(color[1]), int(color[2]))
            
            # Draw bounding box with thickness based on confidence
            thickness = max(1, int(conf * 4))
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
        
        return frame, detected_objects
    
    def draw_info_panel(self, frame):
        """Draw information panel on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw information text
        info_lines = [
            f"Model: YOLOv8{self.model_size.upper()}",
            f"FPS: {self.fps:.1f}",
            f"Confidence: {self.confidence:.2f}",
            f"Max Detections: {self.max_detections}",
            f"Resolution: {width}x{height}",
            f"Tracking: {'ON' if self.enable_tracking else 'OFF'}",
            f"Smoothing: {'ON' if self.enable_smoothing else 'OFF'}",
            f"Confidence Boost: {'ON' if self.enable_confidence_boost else 'OFF'}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 20
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """Main loop for object detection"""
        print("Starting improved object detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Speak detections")
        print("  'c' - Change confidence")
        print("  't' - Toggle tracking")
        print("  'm' - Toggle smoothing")
        print("  'b' - Toggle confidence boost")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Failed to grab frame")
                    break
                
                # Run YOLO detection with improved settings
                results = self.model(frame, verbose=False, conf=self.confidence, iou=self.nms_threshold)
                
                # Process detections with improvements
                detections = self.process_detections(results)
                
                # Draw detections on frame
                frame, detected_objects = self.draw_detections(frame, detections)
                
                # Calculate and display FPS
                self.calculate_fps()
                
                # Draw information panel
                self.draw_info_panel(frame)
                
                # Display instructions
                cv2.putText(frame, "q=quit s=save p=speak c=conf t=track m=smooth b=boost", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show the frame
                cv2.imshow('Improved YOLO Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
                elif key == ord('p'):
                    if detected_objects:
                        speech_text = f"I can see {', '.join(detected_objects)}"
                        print(f"Speaking: {speech_text}")
                        self.speak_text(speech_text)
                    else:
                        speech_text = "I don't see any objects right now"
                        print(f"Speaking: {speech_text}")
                        self.speak_text(speech_text)
                elif key == ord('c'):
                    try:
                        new_conf = float(input("Enter new confidence (0.1-1.0): "))
                        if 0.1 <= new_conf <= 1.0:
                            self.confidence = new_conf
                            print(f"Confidence changed to: {self.confidence}")
                    except ValueError:
                        print("Invalid input")
                elif key == ord('t'):
                    self.enable_tracking = not self.enable_tracking
                    print(f"Tracking: {'ON' if self.enable_tracking else 'OFF'}")
                elif key == ord('m'):
                    self.enable_smoothing = not self.enable_smoothing
                    print(f"Smoothing: {'ON' if self.enable_smoothing else 'OFF'}")
                elif key == ord('b'):
                    self.enable_confidence_boost = not self.enable_confidence_boost
                    print(f"Confidence Boost: {'ON' if self.enable_confidence_boost else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\nStopping object detection...")
        
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Object detection stopped.")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Improved YOLO Object Detection')
    parser.add_argument('--model', '-m', default='s', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--confidence', '-c', type=float, default=0.4,
                       help='Detection confidence threshold (0.1-1.0)')
    parser.add_argument('--nms', '-n', type=float, default=0.4,
                       help='Non-maximum suppression threshold (0.1-1.0)')
    parser.add_argument('--max-detections', '-d', type=int, default=20,
                       help='Maximum number of detections to process')
    
    args = parser.parse_args()
    
    try:
        detector = ImprovedObjectDetector(
            model_size=args.model,
            confidence=args.confidence,
            nms_threshold=args.nms,
            max_detections=args.max_detections
        )
        detector.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 