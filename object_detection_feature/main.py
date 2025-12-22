import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import os
import subprocess

class ObjectDetector:
    def __init__(self):
        """Initialize the YOLO object detector"""
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model for speed
        print("YOLO model loaded successfully!")
        
        # Initialize webcam with better error handling
        self.cap = None
        self.initialize_webcam()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Colors for different object classes
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        
        # Speech tracking
        self.last_spoken_objects = set()
        self.speech_cooldown = 0
        
    def speak_text(self, text):
        """Speak the given text using macOS say command"""
        try:
            # Use macOS say command for text-to-speech
            subprocess.run(['say', text], check=True)
        except subprocess.CalledProcessError:
            print(f"Could not speak: {text}")
        except FileNotFoundError:
            print("Text-to-speech not available on this system")
    
    def speak_detections(self, detected_objects):
        """Speak the detected objects if they're new"""
        if not detected_objects:
            return
            
        # Create a set of current detections for comparison
        current_objects = set(detected_objects)
        
        # Only speak if objects have changed and cooldown has passed
        if (current_objects != self.last_spoken_objects and 
            time.time() - self.speech_cooldown > 2.0):  # 2 second cooldown
            
            # Create speech text
            if len(current_objects) == 1:
                speech_text = f"I can see a {list(current_objects)[0]}"
            elif len(current_objects) == 2:
                speech_text = f"I can see a {list(current_objects)[0]} and a {list(current_objects)[1]}"
            else:
                objects_list = list(current_objects)[:-1]
                last_object = list(current_objects)[-1]
                speech_text = f"I can see {', '.join(objects_list)}, and a {last_object}"
            
            # Speak the text
            self.speak_text(speech_text)
            
            # Update tracking
            self.last_spoken_objects = current_objects
            self.speech_cooldown = time.time()
    
    def initialize_webcam(self):
        """Initialize webcam with multiple attempts"""
        print("Initializing webcam...")
        
        # Try different webcam indices
        for i in range(3):  # Try indices 0, 1, 2
            print(f"Trying webcam index {i}...")
            self.cap = cv2.VideoCapture(i)
            
            if self.cap.isOpened():
                # Test if we can actually read a frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"✅ Webcam {i} initialized successfully!")
                    
                    # Set webcam properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test one more frame to ensure stability
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
        
        # If we get here, no webcam worked
        raise ValueError("Could not initialize any webcam! Please check:")
        print("1. Camera permissions in System Preferences > Security & Privacy > Camera")
        print("2. No other application is using the camera")
        print("3. Camera is properly connected")
        
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on the frame"""
        detected_objects = []
        
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
                    
                    # Only show detections with confidence > 0.5
                    if conf > 0.5:
                        detected_objects.append(class_name)
                        
                        # Get color for this class
                        color = self.colors[cls]
                        color = (int(color[0]), int(color[1]), int(color[2]))
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label text
                        label = f"{class_name}: {conf:.2f}"
                        
                        # Get text size
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Draw label background
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, detected_objects
    
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
        print("Starting object detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'p' to speak detections")
        print("If you see a black screen, please check camera permissions!")
        
        consecutive_failures = 0
        max_failures = 10
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"Failed to grab frame ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        print("Too many consecutive failures. Please check camera permissions.")
                        print("On macOS: System Preferences > Security & Privacy > Camera")
                        break
                    
                    time.sleep(0.1)  # Wait a bit before trying again
                    continue
                
                # Reset failure counter on successful frame
                consecutive_failures = 0
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Draw detections on frame
                frame, detected_objects = self.draw_detections(frame, results)
                
                # Calculate and display FPS
                self.calculate_fps()
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display instructions
                cv2.putText(frame, "Press 'q'=quit, 's'=save, 'p'=speak", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow('YOLO Object Detection', frame)
                
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
                elif key == ord('p'):
                    # Speak current detections
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
            # Clean up
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Object detection stopped.")

def main():
    """Main function to run the object detector"""
    try:
        detector = ObjectDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check camera permissions in System Preferences > Security & Privacy > Camera")
        print("2. Make sure no other app is using the camera")
        print("3. Try restarting your computer")
        print("4. Check if your camera is working in other applications")

if __name__ == "__main__":
    main() 