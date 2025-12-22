#!/usr/bin/env python3
"""
===============================================================================
FEATURE: OBJECT/ENVIRONMENT ANALYSIS (Feature 3) - Better Detector
===============================================================================
This file belongs to the Object Detection and Environment Analysis feature module.

WORK:
- Enhanced YOLOv8 object detector with better performance
- Optimized for speed and accuracy balance
- Improved webcam initialization and error handling
- Real-time object detection with bounding boxes
- FPS monitoring and display
- Text-to-speech announcements for detected objects

KEY CLASS: BetterObjectDetector

KEY FEATURES:
- Better performance optimization
- Improved error handling
- Cleaner code structure
- Enhanced user experience

KEY METHODS:
- initialize_webcam(): Robust camera initialization
- detect_objects(): Optimized detection pipeline
- draw_detections(): Efficient visualization
- speak_detections(): TTS output for objects

This version focuses on better performance and cleaner implementation.

Author: DRDO Project
===============================================================================
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import subprocess

class BetterObjectDetector:
    def __init__(self):
        """Initialize the better object detector"""
        print("Loading YOLO model...")
        self.model = YOLO('yolov8s.pt')
        print("YOLO model loaded successfully!")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam!")
        
        # Set higher resolution for better detection
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Colors for different object classes
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        
        # Enhanced detection settings
        self.confidence = 0.25  # Lower threshold for more detections
        self.nms_threshold = 0.3  # Lower NMS for better overlap handling
        
    def enhance_image(self, frame):
        """Apply image enhancement to improve detection"""
        enhanced = frame.copy()
        
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Apply bilateral filter for noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def boost_confidence(self, detections):
        """Boost confidence for commonly missed objects"""
        boosted = []
        
        # Confidence boost values for common objects
        boost_values = {
            'person': 0.2,        # +20% boost
            'chair': 0.15,        # +15% boost
            'laptop': 0.18,       # +18% boost
            'cell phone': 0.2,    # +20% boost
            'cup': 0.15,          # +15% boost
            'bottle': 0.15,       # +15% boost
            'book': 0.15,         # +15% boost
            'tv': 0.18,           # +18% boost
            'remote': 0.15,       # +15% boost
            'keyboard': 0.15,     # +15% boost
            'mouse': 0.15,        # +15% boost
            'bowl': 0.15,         # +15% boost
            'table': 0.15,        # +15% boost
            'sofa': 0.15,         # +15% boost
            'bed': 0.15,          # +15% boost
            'refrigerator': 0.18, # +18% boost
            'microwave': 0.18,    # +18% boost
            'oven': 0.18,         # +18% boost
            'sink': 0.18,         # +18% boost
            'clock': 0.15,        # +15% boost
            'vase': 0.15,         # +15% boost
            'potted plant': 0.15, # +15% boost
            'car': 0.18,          # +18% boost
            'truck': 0.18,        # +18% boost
            'bus': 0.18,          # +18% boost
            'motorcycle': 0.18,   # +18% boost
            'bicycle': 0.18,      # +18% boost
            'dog': 0.18,          # +18% boost
            'cat': 0.18,          # +18% boost
            'bird': 0.18,         # +18% boost
        }
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Apply boost if available
            if class_name in boost_values:
                confidence += boost_values[class_name]
                confidence = min(confidence, 1.0)  # Cap at 1.0
                detection['confidence'] = confidence
            
            boosted.append(detection)
        
        return boosted
    
    def detect_objects(self, frame):
        """Enhanced object detection with multiple strategies"""
        # Strategy 1: Enhanced image preprocessing
        enhanced_frame = self.enhance_image(frame)
        
        # Strategy 2: Run detection with lower confidence threshold
        results = self.model(enhanced_frame, verbose=False, 
                           conf=self.confidence, iou=self.nms_threshold)
        
        # Strategy 3: Process detections
        all_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    if conf > self.confidence:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_name': class_name,
                            'class_id': cls
                        }
                        all_detections.append(detection)
        
        # Strategy 4: Apply confidence boosting
        boosted_detections = self.boost_confidence(all_detections)
        
        # Strategy 5: Remove duplicates and sort by confidence
        unique_detections = self.remove_duplicates(boosted_detections)
        unique_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_detections[:30]  # Limit to top 30 detections
    
    def remove_duplicates(self, detections):
        """Remove duplicate detections of the same object"""
        unique = []
        for i, det1 in enumerate(detections):
            keep = True
            for j, det2 in enumerate(detections):
                if i != j and det1['class_name'] == det2['class_name']:
                    # Calculate overlap
                    iou = self.calculate_iou(det1['bbox'], det2['bbox'])
                    if iou > 0.4:  # If overlap > 40%
                        if det1['confidence'] < det2['confidence']:
                            keep = False
                            break
            
            if keep:
                unique.append(det1)
        
        return unique
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
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
    
    def speak_text(self, text):
        """Speak the given text using macOS say command"""
        try:
            subprocess.run(['say', text], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Could not speak: {text}")
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """Main loop for better object detection"""
        print("Starting better object detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'p' to speak detections")
        print("Enhanced detection with better preprocessing and confidence boosting!")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Run enhanced detection
                detections = self.detect_objects(frame)
                
                # Draw detections on frame
                frame, detected_objects = self.draw_detections(frame, detections)
                
                # Calculate and display FPS
                self.calculate_fps()
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display detection count
                cv2.putText(frame, f"Detections: {len(detections)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display confidence threshold
                cv2.putText(frame, f"Confidence: {self.confidence:.2f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display instructions
                cv2.putText(frame, "Press 'q'=quit, 's'=save, 'p'=speak", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow('Better Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"better_screenshot_{timestamp}.jpg"
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
                    
        except KeyboardInterrupt:
            print("\nStopping object detection...")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Object detection stopped.")

def main():
    """Main function to run the better object detector"""
    try:
        detector = BetterObjectDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and accessible.")

if __name__ == "__main__":
    main() 