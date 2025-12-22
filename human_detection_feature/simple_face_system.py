import cv2
import numpy as np
import face_recognition
import os
import pickle
import time
from datetime import datetime
import pyttsx3
import threading
from config import SYSTEM_CONFIG, DISPLAY_CONFIG, EMOTION_DISPLAY

class SimpleFaceRecognitionSystem:
    def __init__(self):
        self.known_faces = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Text-to-Speech with Ellen voice
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS for English Siri-like male voice
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Find English male voice (Siri-like)
            selected_voice = None
            
            # Preferred English male voices
            preferred_voices = [
                'com.apple.speech.synthesis.voice.alex',      # Alex (English US Male)
                'com.apple.voice.compact.en-US.Alex',         # Alex compact
                'com.apple.speech.synthesis.voice.daniel',    # Daniel (English UK Male)
                'com.apple.voice.compact.en-GB.Daniel',       # Daniel compact
                'com.apple.speech.synthesis.voice.tom',       # Tom (English US Male)
                'com.apple.voice.compact.en-US.Tom'           # Tom compact
            ]
            
            # First try to find preferred voices
            for voice in voices:
                voice_id = voice.id.lower()
                if any(pref.lower() in voice_id for pref in preferred_voices):
                    selected_voice = voice.id
                    print(f"🎤 Using preferred male voice: {voice.name}")
                    break
            
            # If no preferred voice found, find any English male voice
            if not selected_voice:
                for voice in voices:
                    voice_id = voice.id.lower()
                    voice_name = voice.name.lower()
                    # Look for English male voices
                    if ('en' in voice_id and 'male' in voice_name) or \
                       ('en' in voice_id and 'alex' in voice_id) or \
                       ('en' in voice_id and 'daniel' in voice_id) or \
                       ('en' in voice_id and 'tom' in voice_id):
                        selected_voice = voice.id
                        print(f"🎤 Using English male voice: {voice.name}")
                        break
            
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice)
            else:
                print("⚠️  No English male voice found, using default")
        
        # Optimize TTS settings
        self.tts_engine.setProperty('rate', 140)      # Slower speed
        self.tts_engine.setProperty('volume', 1.0)    # Maximum volume
        self.tts_engine.setProperty('pitch', 1.0)     # Normal pitch
        
        # Load known faces database
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from database"""
        # Create directories if they don't exist
        for directory in ['known_faces', 'person_photos']:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Load person information first
        if os.path.exists('known_faces/person_info.pkl'):
            with open('known_faces/person_info.pkl', 'rb') as f:
                person_info = pickle.load(f)
                for name, info in person_info.items():
                    self.known_faces[name] = info
        
        # Load face encodings if they exist
        if os.path.exists('known_faces/encodings.pkl'):
            with open('known_faces/encodings.pkl', 'rb') as f:
                saved_encodings = pickle.load(f)
                for name, encoding in saved_encodings.items():
                    if name in self.known_faces:
                        self.known_faces[name]["encoding"] = encoding
        
        # Recreate encodings from photos for better recognition
        for person_dir in os.listdir('person_photos'):
            person_path = f"person_photos/{person_dir}"
            if os.path.isdir(person_path):
                person_id = person_dir
                encodings = []
                
                for photo in os.listdir(person_path):
                    photo_path = f"{person_path}/{photo}"
                    try:
                        image = face_recognition.load_image_file(photo_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            encodings.append(face_encodings[0])
                    except Exception as e:
                        print(f"Warning: Error processing {photo_path}: {e}")
                
                if encodings and person_id in self.known_faces:
                    self.known_faces[person_id]["encoding"] = encodings[0]
                    self.known_faces[person_id]["encodings"] = encodings
                    print(f"✅ Loaded {len(encodings)} encodings for {person_id}")
        
        print(f"📊 Loaded {len(self.known_faces)} known persons")
    
    def add_person_manually(self):
        """Add a person manually with photos and information"""
        print("\n👤 Adding New Person")
        print("=" * 30)
        
        # Get person details
        name = input("Enter person's name: ").strip()
        age = input("Enter person's age: ").strip()
        relationship = input("Enter relationship (friend, family, colleague, etc.): ").strip()
        
        # Create person directory
        person_dir = f"person_photos/{name.lower().replace(' ', '_')}"
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        print(f"\n📸 Taking photos for {name}")
        print("Please take 3 photos: front, left, right")
        print("Press 'c' to capture each photo")
        
        photos = []
        photo_names = ['front', 'left', 'right']
        
        for i, photo_name in enumerate(photo_names):
            print(f"\nTaking {photo_name} photo...")
            print("Position your face and press 'c' to capture")
            
            # Open camera for photo capture
            cap = cv2.VideoCapture(SYSTEM_CONFIG["camera_index"])
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Show instructions
                cv2.putText(frame, f"Position for {photo_name} photo", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Capture Photo', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Save photo
                    photo_path = f"{person_dir}/{photo_name}.jpg"
                    cv2.imwrite(photo_path, frame)
                    photos.append(photo_path)
                    print(f"✅ {photo_name} photo saved")
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            cap.release()
            cv2.destroyAllWindows()
        
        # Generate face encodings from photos
        encodings = []
        for photo_path in photos:
            try:
                image = face_recognition.load_image_file(photo_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings.append(face_encodings[0])
            except Exception as e:
                print(f"❌ Error processing {photo_path}: {e}")
        
        if not encodings:
            print("❌ No faces detected in photos")
            return
        
        # Use the first encoding as primary
        primary_encoding = encodings[0]
        
        # Add to known faces
        person_id = name.lower().replace(' ', '_')
        self.known_faces[person_id] = {
            "name": name,
            "age": age,
            "relationship": relationship,
            "encoding": primary_encoding,
            "photos": photos,
            "encodings": encodings  # Store all encodings for better recognition
        }
        
        # Save to files
        self.save_encodings()
        self.save_person_info()
        
        print(f"\n✅ Person '{name}' added successfully!")
        print(f"   Age: {age}")
        print(f"   Relationship: {relationship}")
        print(f"   Photos: {len(photos)} saved")
    
    def detect_human(self, frame):
        """Detect if there's a human face in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0, faces
    
    def recognize_person(self, face_encoding):
        """Recognize if the person is known"""
        if face_encoding is None:
            return None, 0
            
        best_match = None
        best_distance = 1.0
        
        for name, data in self.known_faces.items():
            if data.get("encoding") is not None:
                # Compare with all encodings for better accuracy
                encodings_to_check = data.get("encodings", [data["encoding"]])
                
                for encoding in encodings_to_check:
                    match = face_recognition.compare_faces([encoding], face_encoding, tolerance=SYSTEM_CONFIG["face_detection_confidence"])
                    distance = face_recognition.face_distance([encoding], face_encoding)[0]
                    
                    if match[0] and distance < best_distance:
                        best_match = name
                        best_distance = distance
        
        # If confidence is 0.5 or less, treat as unknown person
        confidence = 1 - best_distance
        if confidence <= SYSTEM_CONFIG["face_detection_confidence"]:
            return None, best_distance
        
        return best_match, best_distance
    
    def speak_person_info(self, person_name):
        """Speak information about the recognized person"""
        if person_name not in self.known_faces:
            return
        
        person_data = self.known_faces[person_name]
        
        # Clean up relationship text for better speech
        relationship = person_data['relationship']
        if relationship.lower() in ['u urself', 'yourself', 'me', 'myself']:
            relationship_text = "myself"
        elif relationship.lower() in ['friend', 'buddy', 'pal']:
            relationship_text = "friend"
        elif relationship.lower() in ['family', 'relative']:
            relationship_text = "family member"
        else:
            relationship_text = relationship
        
        # Create natural speech
        speech_text = f"This is {person_data['name']}, who is {person_data['age']} years old. They are my {relationship_text}."
        
        print(f"🗣️  Speaking: {speech_text}")
        
        # Speak in a separate thread to avoid blocking
        def speak():
            self.tts_engine.say(speech_text)
            self.tts_engine.runAndWait()
        
        threading.Thread(target=speak).start()
    
    def start_detection(self):
        """Start the face recognition system"""
        self.cap = cv2.VideoCapture(SYSTEM_CONFIG["camera_index"])
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Simple Face Recognition System Started!")
        print("Press 'q' to quit, 'a' to add person, 'p' to speak info")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect human faces
            is_human, face_locations = self.detect_human(frame)
            
            if is_human:
                # Get face encodings using face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)
                face_locations_recognition = face_recognition.face_locations(rgb_frame)
                
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations_recognition)):
                    # Recognize person
                    recognized_name, confidence = self.recognize_person(face_encoding)
                    
                    # Draw rectangle around face
                    top, right, bottom, left = face_location
                    box_color = DISPLAY_CONFIG["box_color"] if recognized_name else DISPLAY_CONFIG["unknown_color"]
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, DISPLAY_CONFIG["line_thickness"])
                    
                    # Prepare display text
                    if recognized_name:
                        person_data = self.known_faces[recognized_name]
                        display_text = f"I recognize this person!"
                        name_text = f"Name: {person_data['name']}"
                        age_text = f"Age: {person_data['age']}"
                        relationship_text = f"Relationship: {person_data['relationship']}"
                        confidence_text = f"Confidence: {1-confidence:.2f}"
                        
                        # Voice status
                        voice_text = "🗣️  Press 'p' to speak info"
                        voice_color = (0, 255, 255)  # Yellow
                    else:
                        display_text = "I don't know this person"
                        name_text = ""
                        age_text = ""
                        relationship_text = ""
                        confidence_text = ""
                        voice_text = "Press 'a' to add this person"
                        voice_color = (128, 128, 128)  # Gray
                    
                    # Display text on frame
                    y_position = top - 10
                    cv2.putText(frame, display_text, (left, y_position), 
                              cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                              DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                    
                    y_position -= 30
                    if name_text:
                        cv2.putText(frame, name_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                        y_position -= 25
                    
                    if age_text:
                        cv2.putText(frame, age_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                        y_position -= 25
                    
                    if relationship_text:
                        cv2.putText(frame, relationship_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                        y_position -= 25
                    
                    if confidence_text:
                        cv2.putText(frame, confidence_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                        y_position -= 25
                    
                    if voice_text:
                        cv2.putText(frame, voice_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  voice_color, DISPLAY_CONFIG["font_thickness"])
            else:
                # No human detected
                cv2.putText(frame, "No human detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate and display FPS
            if SYSTEM_CONFIG["display_fps"]:
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display frame
            cv2.imshow(DISPLAY_CONFIG["window_title"], frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.add_person_manually()
            elif key == ord('p'):
                # Speak info for recognized person
                if is_human and face_encodings:
                    face_encoding = face_encodings[0]
                    recognized_name, confidence = self.recognize_person(face_encoding)
                    if recognized_name:
                        print(f"🎤 Speaking info for: {recognized_name}")
                        self.speak_person_info(recognized_name)
                    else:
                        print("❌ Speaking: I don't know this person")
                        # Speak "I don't know this person" in a separate thread
                        def speak_unknown():
                            self.tts_engine.say("I don't know this person")
                            self.tts_engine.runAndWait()
                        threading.Thread(target=speak_unknown).start()
                else:
                    print("❌ No face detected to speak about")
            elif key == ord('h'):
                self.show_help()
        
        self.cleanup()
    
    def save_encodings(self):
        """Save face encodings to file"""
        encodings = {}
        for name, data in self.known_faces.items():
            if data["encoding"] is not None:
                encodings[name] = data["encoding"]
        
        with open('known_faces/encodings.pkl', 'wb') as f:
            pickle.dump(encodings, f)
    
    def save_person_info(self):
        """Save person information"""
        person_info = {}
        for name, data in self.known_faces.items():
            person_info[name] = {
                "name": data["name"],
                "age": data["age"],
                "relationship": data["relationship"],
                "photos": data.get("photos", [])
            }
        
        with open('known_faces/person_info.pkl', 'wb') as f:
            pickle.dump(person_info, f)
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("🎯 SIMPLE FACE RECOGNITION SYSTEM HELP")
        print("="*50)
        print("📹 Controls:")
        print("   'q' - Quit the application")
        print("   'a' - Add new person (take photos + input info)")
        print("   'p' - Speak information about recognized person")
        print("   'h' - Show this help message")
        print("\n🔍 Features:")
        print("   • Face detection using webcam")
        print("   • Manual person addition with photos")
        print("   • Face recognition with known persons")
        print("   • Voice output with English Siri-like male voice")
        print("\n👤 Adding a Person:")
        print("   1. Press 'a' to add new person")
        print("   2. Enter name, age, relationship")
        print("   3. Take 3 photos: front, left, right")
        print("   4. System saves and recognizes them")
        print("="*50)
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the simple face recognition system"""
    print("Initializing Simple Face Recognition System...")
    print("This system will:")
    print("- Detect faces using webcam")
    print("- Allow manual addition of persons with photos")
    print("- Recognize known persons")
    print("- Speak information using English Siri-like male voice")
    
    system = SimpleFaceRecognitionSystem()
    
    try:
        system.start_detection()
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main() 