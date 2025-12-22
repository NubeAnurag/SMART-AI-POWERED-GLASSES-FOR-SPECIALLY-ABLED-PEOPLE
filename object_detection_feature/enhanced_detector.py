import cv2
import numpy as np
from ultralytics import YOLO
import time
import subprocess
import argparse
from collections import deque
import threading
import torch
import torchvision.transforms as transforms
from PIL import Image

class EnhancedObjectDetector:
    def __init__(self, confidence=0.3, nms_threshold=0.3, max_detections=30):
        """
        Enhanced object detector that addresses YOLO limitations
        """
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        print("Loading multiple detection models for enhanced performance...")
        
        # Load multiple YOLO models for ensemble detection
        self.models = {}
        self.load_models()
        
        # Initialize webcam
        self.cap = None
        self.initialize_webcam()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Detection tracking for stability
        self.detection_history = deque(maxlen=15)
        self.stable_detections = {}
        
        # Colors for different object classes
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        
        # Speech tracking
        self.last_spoken_objects = set()
        self.speech_cooldown = 0
        
        # Enhanced settings
        self.enable_ensemble = True
        self.enable_preprocessing = True
        self.enable_postprocessing = True
        self.enable_confidence_boosting = True
        self.enable_multi_scale = True
        
        # Preprocessing parameters
        self.setup_preprocessing()
        
    def load_models(self):
        """Load multiple YOLO models for ensemble detection"""
        try:
            # Load different model sizes for ensemble
            print("Loading YOLOv8n (nano) for fast detection...")
            self.models['nano'] = YOLO('yolov8n.pt')
            
            print("Loading YOLOv8s (small) for balanced detection...")
            self.models['small'] = YOLO('yolov8s.pt')
            
            print("Loading YOLOv8m (medium) for high accuracy...")
            self.models['medium'] = YOLO('yolov8m.pt')
            
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load all models: {e}")
            # Fallback to available models
            if 'small' in self.models:
                print("Using YOLOv8s as fallback")
            else:
                raise Exception("No models available!")
    
    def setup_preprocessing(self):
        """Setup advanced preprocessing parameters"""
        self.preprocessing_params = {
            'brightness_boost': 1.2,      # Increase brightness
            'contrast_boost': 1.3,        # Increase contrast
            'sharpness_boost': 1.5,       # Increase sharpness
            'noise_reduction': True,      # Apply noise reduction
            'edge_enhancement': True,     # Enhance edges
            'multi_scale_factors': [0.8, 1.0, 1.2, 1.5],  # Multi-scale detection
            'rotation_angles': [-15, -10, -5, 0, 5, 10, 15]  # Rotation augmentation
        }
        
        # Image enhancement kernels
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.edge_kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    
    def enhance_image(self, frame):
        """Apply advanced image preprocessing to improve detection"""
        if not self.enable_preprocessing:
            return frame
        
        enhanced = frame.copy()
        
        # Convert to float for processing
        enhanced = enhanced.astype(np.float32) / 255.0
        
        # Brightness and contrast enhancement
        enhanced = enhanced * self.preprocessing_params['brightness_boost']
        enhanced = np.clip(enhanced, 0, 1)
        
        # Contrast enhancement using CLAHE
        if len(enhanced.shape) == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply((lab[:,:,0] * 255).astype(np.uint8)) / 255.0
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Sharpness enhancement
        if self.preprocessing_params['sharpness_boost']:
            enhanced = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
            enhanced = np.clip(enhanced, 0, 1)
        
        # Edge enhancement
        if self.preprocessing_params['edge_enhancement']:
            edges = cv2.filter2D(enhanced, -1, self.edge_kernel)
            enhanced = enhanced + 0.1 * edges
            enhanced = np.clip(enhanced, 0, 1)
        
        # Noise reduction
        if self.preprocessing_params['noise_reduction']:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Convert back to uint8
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def multi_scale_detection(self, frame):
        """Perform detection at multiple scales for better small object detection"""
        if not self.enable_multi_scale:
            return self.run_single_detection(frame)
        
        all_detections = []
        
        for scale_factor in self.preprocessing_params['multi_scale_factors']:
            # Resize frame
            height, width = frame.shape[:2]
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Run detection on resized frame
            scale_detections = self.run_single_detection(resized_frame)
            
            # Scale bounding boxes back to original size
            for detection in scale_detections:
                detection['bbox'] = [
                    int(detection['bbox'][0] / scale_factor),
                    int(detection['bbox'][1] / scale_factor),
                    int(detection['bbox'][2] / scale_factor),
                    int(detection['bbox'][3] / scale_factor)
                ]
                all_detections.append(detection)
        
        return all_detections
    
    def run_single_detection(self, frame):
        """Run detection with a single model"""
        detections = []
        
        # Use the medium model for best accuracy
        model = self.models.get('medium', self.models.get('small'))
        results = model(frame, verbose=False, conf=self.confidence, iou=self.nms_threshold)
        
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
                            'class_id': cls,
                            'model': 'medium'
                        }
                        detections.append(detection)
        
        return detections
    
    def ensemble_detection(self, frame):
        """Combine detections from multiple models for better accuracy"""
        if not self.enable_ensemble:
            return self.run_single_detection(frame)
        
        all_detections = []
        
        # Run detection with each model
        for model_name, model in self.models.items():
            try:
                results = model(frame, verbose=False, conf=self.confidence, iou=self.nms_threshold)
                
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
                                    'class_id': cls,
                                    'model': model_name
                                }
                                all_detections.append(detection)
                                
            except Exception as e:
                print(f"Warning: Model {model_name} failed: {e}")
                continue
        
        return all_detections
    
    def apply_advanced_confidence_boosting(self, detections):
        """Apply advanced confidence boosting for better detection"""
        if not self.enable_confidence_boosting:
            return detections
        
        # Enhanced confidence boosting for common objects
        common_objects = {
            'person': 0.15,      # Boost person detection significantly
            'chair': 0.08,       # Boost chair detection
            'laptop': 0.12,      # Boost laptop detection
            'cell phone': 0.15,  # Boost phone detection
            'cup': 0.08,         # Boost cup detection
            'bottle': 0.08,      # Boost bottle detection
            'book': 0.08,        # Boost book detection
            'tv': 0.12,          # Boost TV detection
            'remote': 0.08,      # Boost remote detection
            'keyboard': 0.08,    # Boost keyboard detection
            'mouse': 0.08,       # Boost mouse detection
            'bowl': 0.08,        # Boost bowl detection
            'table': 0.08,       # Boost table detection
            'sofa': 0.08,        # Boost sofa detection
            'bed': 0.08,         # Boost bed detection
            'refrigerator': 0.10, # Boost refrigerator detection
            'microwave': 0.10,   # Boost microwave detection
            'oven': 0.10,        # Boost oven detection
            'sink': 0.10,        # Boost sink detection
            'clock': 0.08,       # Boost clock detection
            'vase': 0.08,        # Boost vase detection
            'potted plant': 0.08, # Boost plant detection
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
    
    def advanced_postprocessing(self, detections):
        """Apply advanced post-processing to improve detection quality"""
        if not self.enable_postprocessing:
            return detections
        
        # Remove duplicate detections with higher confidence
        filtered_detections = []
        for i, det1 in enumerate(detections):
            keep = True
            for j, det2 in enumerate(detections):
                if i != j and det1['class_name'] == det2['class_name']:
                    # Calculate overlap
                    iou = self.calculate_iou(det1['bbox'], det2['bbox'])
                    if iou > 0.3:  # Lower threshold for better detection
                        if det1['confidence'] < det2['confidence']:
                            keep = False
                            break
            
            if keep:
                filtered_detections.append(det1)
        
        # Sort by confidence and limit
        filtered_detections.sort(key=lambda x: x['confidence'], reverse=True)
        filtered_detections = filtered_detections[:self.max_detections]
        
        return filtered_detections
    
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
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame"""
        detected_objects = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            model_name = detection.get('model', 'unknown')
            
            detected_objects.append(class_name)
            
            # Get color for this class
            color = self.colors[class_id]
            color = (int(color[0]), int(color[1]), int(color[2]))
            
            # Draw bounding box with thickness based on confidence
            thickness = max(1, int(conf * 4))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Create label text with model info
            label = f"{class_name}: {conf:.2f} ({model_name})"
            
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
        cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw information text
        info_lines = [
            f"Enhanced Object Detection System",
            f"FPS: {self.fps:.1f}",
            f"Confidence: {self.confidence:.2f}",
            f"Max Detections: {self.max_detections}",
            f"Resolution: {width}x{height}",
            f"Ensemble: {'ON' if self.enable_ensemble else 'OFF'}",
            f"Preprocessing: {'ON' if self.enable_preprocessing else 'OFF'}",
            f"Multi-scale: {'ON' if self.enable_multi_scale else 'OFF'}",
            f"Confidence Boost: {'ON' if self.enable_confidence_boosting else 'OFF'}",
            f"Models: {len(self.models)} loaded"
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
        """Main loop for enhanced object detection"""
        print("Starting enhanced object detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Speak detections")
        print("  'c' - Change confidence")
        print("  'e' - Toggle ensemble detection")
        print("  'm' - Toggle multi-scale detection")
        print("  'b' - Toggle confidence boost")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Failed to grab frame")
                    break
                
                # Apply image enhancement
                enhanced_frame = self.enhance_image(frame)
                
                # Run enhanced detection
                if self.enable_ensemble:
                    detections = self.ensemble_detection(enhanced_frame)
                elif self.enable_multi_scale:
                    detections = self.multi_scale_detection(enhanced_frame)
                else:
                    detections = self.run_single_detection(enhanced_frame)
                
                # Apply advanced confidence boosting
                detections = self.apply_advanced_confidence_boosting(detections)
                
                # Apply advanced post-processing
                detections = self.advanced_postprocessing(detections)
                
                # Draw detections on frame
                frame, detected_objects = self.draw_detections(frame, detections)
                
                # Calculate and display FPS
                self.calculate_fps()
                
                # Draw information panel
                self.draw_info_panel(frame)
                
                # Display instructions
                cv2.putText(frame, "q=quit s=save p=speak c=conf e=ensemble m=multiscale b=boost", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show the frame
                cv2.imshow('Enhanced Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"enhanced_screenshot_{timestamp}.jpg"
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
                elif key == ord('e'):
                    self.enable_ensemble = not self.enable_ensemble
                    print(f"Ensemble Detection: {'ON' if self.enable_ensemble else 'OFF'}")
                elif key == ord('m'):
                    self.enable_multi_scale = not self.enable_multi_scale
                    print(f"Multi-scale Detection: {'ON' if self.enable_multi_scale else 'OFF'}")
                elif key == ord('b'):
                    self.enable_confidence_boosting = not self.enable_confidence_boosting
                    print(f"Confidence Boost: {'ON' if self.enable_confidence_boosting else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\nStopping object detection...")
        
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Object detection stopped.")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Object Detection with Multiple Models')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                       help='Detection confidence threshold (0.1-1.0)')
    parser.add_argument('--nms', '-n', type=float, default=0.3,
                       help='Non-maximum suppression threshold (0.1-1.0)')
    parser.add_argument('--max-detections', '-d', type=int, default=30,
                       help='Maximum number of detections to process')
    
    args = parser.parse_args()
    
    try:
        detector = EnhancedObjectDetector(
            confidence=args.confidence,
            nms_threshold=args.nms,
            max_detections=args.max_detections
        )
        detector.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 