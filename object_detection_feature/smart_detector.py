import cv2
import numpy as np
from ultralytics import YOLO
import time
import subprocess
import argparse
from collections import deque
import threading

class SmartObjectDetector:
    def __init__(self, model_size='s', confidence=0.4, nms_threshold=0.4, max_detections=20):
        """
        Initialize the smart YOLO object detector with environment understanding
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
        self.detection_history = deque(maxlen=10)
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
        
        # Environment understanding
        self.setup_environment_rules()
        
    def setup_environment_rules(self):
        """Setup rules for environment understanding and object relationships"""
        
        # Object categories for better grouping
        self.object_categories = {
            'people': ['person', 'man', 'woman', 'boy', 'girl'],
            'furniture': ['chair', 'table', 'bed', 'sofa', 'couch', 'desk', 'shelf'],
            'electronics': ['laptop', 'computer', 'tv', 'monitor', 'cell phone', 'phone', 'remote', 'keyboard', 'mouse'],
            'kitchen': ['bowl', 'cup', 'bottle', 'plate', 'fork', 'spoon', 'knife', 'sink', 'refrigerator', 'microwave', 'oven'],
            'clothing': ['shirt', 'pants', 'dress', 'hat', 'shoes', 'bag', 'backpack', 'suitcase'],
            'food': ['apple', 'banana', 'orange', 'pizza', 'hot dog', 'sandwich', 'cake'],
            'animals': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep'],
            'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
            'sports': ['baseball', 'basketball', 'tennis racket', 'skateboard'],
            'plants': ['potted plant', 'flower', 'tree']
        }
        
        # Relationship rules for natural speech
        self.relationship_rules = {
            'person': {
                'holding': ['cell phone', 'phone', 'cup', 'bowl', 'bottle', 'book', 'remote', 'keyboard', 'mouse'],
                'sitting_on': ['chair', 'sofa', 'couch', 'bed'],
                'using': ['laptop', 'computer', 'tv', 'monitor'],
                'wearing': ['shirt', 'pants', 'dress', 'hat', 'shoes'],
                'carrying': ['bag', 'backpack', 'suitcase']
            },
            'chair': {
                'has_person': ['person'],
                'near': ['table', 'desk', 'laptop', 'computer']
            },
            'table': {
                'has_on': ['laptop', 'computer', 'cup', 'bowl', 'bottle', 'book', 'phone', 'cell phone'],
                'near': ['chair', 'person']
            },
            'laptop': {
                'being_used_by': ['person'],
                'on': ['table', 'desk']
            }
        }
        
        # Spatial relationship thresholds
        self.spatial_thresholds = {
            'holding_distance': 100,  # pixels
            'near_distance': 150,     # pixels
            'on_distance': 50         # pixels
        }
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate centers
        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        return distance
    
    def check_spatial_relationship(self, bbox1, bbox2, relationship_type):
        """Check if two objects have a specific spatial relationship"""
        distance = self.calculate_distance(bbox1, bbox2)
        threshold = self.spatial_thresholds.get(relationship_type, 100)
        return distance <= threshold
    
    def analyze_object_relationships(self, detections):
        """Analyze relationships between detected objects"""
        relationships = []
        
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i != j:
                    obj1_name = det1['class_name']
                    obj2_name = det2['class_name']
                    
                    # Check if there's a defined relationship
                    if obj1_name in self.relationship_rules:
                        for relationship, related_objects in self.relationship_rules[obj1_name].items():
                            if obj2_name in related_objects:
                                # Check spatial relationship
                                if self.check_spatial_relationship(det1['bbox'], det2['bbox'], relationship):
                                    relationships.append({
                                        'subject': obj1_name,
                                        'object': obj2_name,
                                        'relationship': relationship,
                                        'confidence': min(det1['confidence'], det2['confidence'])
                                    })
        
        return relationships
    
    def create_natural_speech(self, detections, relationships):
        """Create natural speech based on object relationships"""
        if not detections:
            return "I don't see any objects right now"
        
        # Group objects by category
        categorized_objects = {}
        for det in detections:
            obj_name = det['class_name']
            category = self.get_object_category(obj_name)
            if category not in categorized_objects:
                categorized_objects[category] = []
            categorized_objects[category].append(obj_name)
        
        # Build natural speech
        speech_parts = []
        
        # Handle relationships first
        if relationships:
            for rel in relationships:
                if rel['relationship'] == 'holding':
                    speech_parts.append(f"a {rel['subject']} holding a {rel['object']}")
                elif rel['relationship'] == 'sitting_on':
                    speech_parts.append(f"a {rel['subject']} sitting on a {rel['object']}")
                elif rel['relationship'] == 'using':
                    speech_parts.append(f"a {rel['subject']} using a {rel['object']}")
                elif rel['relationship'] == 'wearing':
                    speech_parts.append(f"a {rel['subject']} wearing a {rel['object']}")
                elif rel['relationship'] == 'carrying':
                    speech_parts.append(f"a {rel['subject']} carrying a {rel['object']}")
                elif rel['relationship'] == 'on':
                    speech_parts.append(f"a {rel['object']} on a {rel['subject']}")
        
        # Add remaining objects without relationships
        used_objects = set()
        for rel in relationships:
            used_objects.add(rel['subject'])
            used_objects.add(rel['object'])
        
        remaining_objects = []
        for det in detections:
            if det['class_name'] not in used_objects:
                remaining_objects.append(det['class_name'])
        
        if remaining_objects:
            if len(remaining_objects) == 1:
                speech_parts.append(f"a {remaining_objects[0]}")
            else:
                speech_parts.append(f"{', '.join(['a ' + obj for obj in remaining_objects[:-1]])} and a {remaining_objects[-1]}")
        
        # Combine speech parts
        if len(speech_parts) == 1:
            return f"I can see {speech_parts[0]}"
        elif len(speech_parts) == 2:
            return f"I can see {speech_parts[0]} and {speech_parts[1]}"
        else:
            return f"I can see {', '.join(speech_parts[:-1])} and {speech_parts[-1]}"
    
    def get_object_category(self, object_name):
        """Get the category of an object"""
        for category, objects in self.object_categories.items():
            if object_name in objects:
                return category
        return 'other'
    
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
            'bowl': 0.05,       # Boost bowl detection
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
    
    def draw_detections(self, frame, detections, relationships):
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
        
        # Draw relationship lines
        for rel in relationships:
            # Find the two objects involved in the relationship
            obj1_bbox = None
            obj2_bbox = None
            
            for det in detections:
                if det['class_name'] == rel['subject']:
                    obj1_bbox = det['bbox']
                elif det['class_name'] == rel['object']:
                    obj2_bbox = det['bbox']
            
            if obj1_bbox and obj2_bbox:
                # Calculate centers
                x1_1, y1_1, x2_1, y2_1 = obj1_bbox
                x1_2, y1_2, x2_2, y2_2 = obj2_bbox
                
                center1_x = (x1_1 + x2_1) // 2
                center1_y = (y1_1 + y2_1) // 2
                center2_x = (x1_2 + x2_2) // 2
                center2_y = (y1_2 + y2_2) // 2
                
                # Draw relationship line
                cv2.line(frame, (center1_x, center1_y), (center2_x, center2_y), (0, 255, 255), 2)
                
                # Draw relationship label
                mid_x = (center1_x + center2_x) // 2
                mid_y = (center1_y + center2_y) // 2
                cv2.putText(frame, rel['relationship'], (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame, detected_objects
    
    def draw_info_panel(self, frame):
        """Draw information panel on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
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
            f"Confidence Boost: {'ON' if self.enable_confidence_boost else 'OFF'}",
            f"Smart Speech: ON"
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
        print("Starting smart object detection with environment understanding...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Speak detections (smart)")
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
                
                # Analyze object relationships
                relationships = self.analyze_object_relationships(detections)
                
                # Draw detections on frame
                frame, detected_objects = self.draw_detections(frame, detections, relationships)
                
                # Calculate and display FPS
                self.calculate_fps()
                
                # Draw information panel
                self.draw_info_panel(frame)
                
                # Display instructions
                cv2.putText(frame, "q=quit s=save p=speak c=conf t=track m=smooth b=boost", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show the frame
                cv2.imshow('Smart YOLO Object Detection', frame)
                
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
                    # Create smart speech
                    speech_text = self.create_natural_speech(detections, relationships)
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
    parser = argparse.ArgumentParser(description='Smart YOLO Object Detection with Environment Understanding')
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
        detector = SmartObjectDetector(
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