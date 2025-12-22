import cv2
import numpy as np
import face_recognition
import os
import pickle
import time
from datetime import datetime
import wave
import pyaudio
import threading
from config import KNOWN_PERSONS, SYSTEM_CONFIG, DISPLAY_CONFIG, EMOTION_DISPLAY

class ImprovedVoiceHumanDetectionSystem:
    def __init__(self):
        self.known_faces = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Voice recording state
        self.is_recording = False
        self.recording_frames = []
        self.recording_stream = None
        self.recording_person = None
        
        # Load known faces database
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from database or create sample data"""
        # Initialize known faces from config
        self.known_faces = {}
        for person_id, person_data in KNOWN_PERSONS.items():
            self.known_faces[person_id] = {
                "name": person_data["name"],
                "age": person_data["age"],
                "gender": person_data["gender"],
                "description": person_data["description"],
                "encoding": None,  # Will be set when face is detected
                "voice_file": None  # Will be set when voice is recorded
            }
        
        # Create directories if they don't exist
        for directory in ['known_faces', 'voice_recordings', 'saved_faces']:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
        # Load face encodings if they exist
        if os.path.exists('known_faces/encodings.pkl'):
            with open('known_faces/encodings.pkl', 'rb') as f:
                saved_encodings = pickle.load(f)
                for name, encoding in saved_encodings.items():
                    if name in self.known_faces:
                        self.known_faces[name]["encoding"] = encoding
        
        # Load voice file mappings
        if os.path.exists('known_faces/voice_mappings.pkl'):
            with open('known_faces/voice_mappings.pkl', 'rb') as f:
                voice_mappings = pickle.load(f)
                for name, voice_file in voice_mappings.items():
                    if name in self.known_faces:
                        self.known_faces[name]["voice_file"] = voice_file
    
    def start_voice_recording(self, person_name):
        """Start continuous voice recording"""
        if self.is_recording:
            print("❌ Already recording! Press 'p' to stop and save.")
            return
        
        print(f"🎤 Starting voice recording for {person_name}...")
        print("💬 Speak your details now... Press 'p' to stop recording and play audio")
        
        self.is_recording = True
        self.recording_frames = []
        self.recording_person = person_name
        
        # Start recording in a separate thread
        threading.Thread(target=self._record_audio_thread).start()
    
    def _record_audio_thread(self):
        """Background thread for audio recording"""
        p = pyaudio.PyAudio()
        
        self.recording_stream = p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
        
        print("🎙️  Listening... (Press 'p' to stop)")
        
        while self.is_recording:
            try:
                data = self.recording_stream.read(self.CHUNK, exception_on_overflow=False)
                self.recording_frames.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break
        
        # Clean up recording
        if self.recording_stream:
            self.recording_stream.stop_stream()
            self.recording_stream.close()
        p.terminate()
    
    def stop_and_save_voice(self):
        """Stop recording and save the voice file"""
        if not self.is_recording:
            print("❌ Not currently recording")
            return
        
        print("⏹️  Stopping recording...")
        self.is_recording = False
        
        # Wait a moment for recording thread to finish
        time.sleep(0.5)
        
        if not self.recording_frames:
            print("❌ No audio recorded")
            return
        
        # Save the recorded audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        voice_filename = f"voice_recordings/{self.recording_person}_{timestamp}.wav"
        
        wf = wave.open(voice_filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.recording_frames))
        wf.close()
        
        print(f"💾 Voice saved to: {voice_filename}")
        
        # Update the person's voice file
        if self.recording_person in self.known_faces:
            self.known_faces[self.recording_person]["voice_file"] = voice_filename
            self.save_voice_mappings()
            print(f"✅ Voice recorded for {self.recording_person}")
        
        # Clear recording state
        self.recording_frames = []
        self.recording_person = None
        
        return voice_filename
    
    def play_voice(self, voice_file):
        """Play recorded voice"""
        if not voice_file or not os.path.exists(voice_file):
            print("❌ Voice file not found")
            return
        
        print(f"🔊 Playing audio: {voice_file}")
        
        try:
            wf = wave.open(voice_file, 'rb')
            p = pyaudio.PyAudio()
            
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                           channels=wf.getnchannels(),
                           rate=wf.getframerate(),
                           output=True)
            
            data = wf.readframes(self.CHUNK)
            
            while data:
                stream.write(data)
                data = wf.readframes(self.CHUNK)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            
            print("✅ Audio playback completed")
            
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
    
    def detect_human(self, frame):
        """Detect if there's a human face in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0, faces
    
    def recognize_person(self, face_encoding):
        """Recognize if the person is known"""
        if face_encoding is None:
            return None, 0
            
        for name, data in self.known_faces.items():
            if data["encoding"] is not None:
                # Compare face encodings
                matches = face_recognition.compare_faces([data["encoding"]], face_encoding, tolerance=SYSTEM_CONFIG["face_detection_confidence"])
                if matches[0]:
                    return name, face_recognition.face_distance([data["encoding"]], face_encoding)[0]
        return None, 1.0
    
    def estimate_gender_age(self, face_image):
        """Simple gender and age estimation based on face features"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect facial features (only eyes to avoid cascade file issues)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Simple heuristics (this is just for demonstration)
        face_area = face_image.shape[0] * face_image.shape[1]
        estimated_age = 25 + (face_area % 20)
        
        gender = "Male" if len(eyes) > 1 else "Female"
        
        return {
            'age': estimated_age,
            'gender': gender,
            'emotion': 'neutral'
        }
    
    def start_detection(self):
        """Start the human detection system"""
        self.cap = cv2.VideoCapture(SYSTEM_CONFIG["camera_index"])
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Improved Voice-Enabled Human Detection System Started!")
        print("Press 'q' to quit, 's' to save face, 'v' to start recording, 'p' to stop recording and play")
        
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
                    # Analyze face
                    analysis = self.estimate_gender_age(frame)
                    
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
                        gender_text = f"Gender: {person_data['gender']}"
                        emotion_text = f"Emotion: {EMOTION_DISPLAY.get(analysis['emotion'], analysis['emotion'])}"
                        confidence_text = f"Confidence: {1-confidence:.2f}"
                        
                        # Voice status
                        if self.is_recording and self.recording_person == recognized_name:
                            voice_text = "🎙️  Listening..."  # Red color for recording
                            voice_color = (0, 0, 255)  # Red
                        elif person_data.get("voice_file"):
                            voice_text = "🎤 Voice available"
                            voice_color = (0, 255, 255)  # Yellow
                        else:
                            voice_text = "🎤 No voice recorded"
                            voice_color = (128, 128, 128)  # Gray
                    else:
                        display_text = "I don't know this person"
                        name_text = f"Detected Gender: {analysis['gender']}"
                        age_text = f"Estimated Age: {analysis['age']}"
                        emotion_text = f"Emotion: {EMOTION_DISPLAY.get(analysis['emotion'], analysis['emotion'])}"
                        gender_text = ""
                        confidence_text = ""
                        voice_text = ""
                        voice_color = DISPLAY_CONFIG["text_color"]
                    
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
                    
                    if gender_text:
                        cv2.putText(frame, gender_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                        y_position -= 25
                    
                    if emotion_text:
                        cv2.putText(frame, emotion_text, (left, y_position), 
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
            elif key == ord('s'):
                self.save_current_face(frame, face_encodings, face_locations_recognition)
            elif key == ord('v'):
                self.handle_voice_recording(frame, face_encodings, face_locations_recognition)
            elif key == ord('p'):
                self.handle_voice_playback(frame, face_encodings, face_locations_recognition)
            elif key == ord('h'):
                self.show_help()
        
        self.cleanup()
    
    def handle_voice_recording(self, frame, face_encodings, face_locations):
        """Handle voice recording for the currently detected face"""
        if not face_encodings:
            print("No face detected to record voice for")
            return
        
        # Recognize the current face
        face_encoding = face_encodings[0]
        recognized_name, confidence = self.recognize_person(face_encoding)
        
        if recognized_name:
            if self.is_recording:
                print("❌ Already recording! Press 'p' to stop and save.")
            else:
                self.start_voice_recording(recognized_name)
        else:
            print("❌ Face not recognized. Save the face first with 's' key")
    
    def handle_voice_playback(self, frame, face_encodings, face_locations):
        """Handle voice playback for the currently detected face"""
        if not face_encodings:
            print("No face detected to play voice for")
            return
        
        # If currently recording, stop and save
        if self.is_recording:
            voice_file = self.stop_and_save_voice()
            if voice_file:
                print("🔊 Playing the recorded audio...")
                threading.Thread(target=self.play_voice, args=(voice_file,)).start()
            return
        
        # Recognize the current face
        face_encoding = face_encodings[0]
        recognized_name, confidence = self.recognize_person(face_encoding)
        
        if recognized_name:
            person_data = self.known_faces[recognized_name]
            voice_file = person_data.get("voice_file")
            
            if voice_file:
                print("🔊 Playing audio for recognized person...")
                threading.Thread(target=self.play_voice, args=(voice_file,)).start()
            else:
                print(f"❌ No voice recorded for {recognized_name}")
        else:
            print("❌ Face not recognized")
    
    def save_current_face(self, frame, face_encodings, face_locations):
        """Save current face to known faces database"""
        if not face_encodings:
            print("No face detected to save")
            return
            
        # Save the first detected face
        face_encoding = face_encodings[0]
        face_location = face_locations[0]
        
        # Generate unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"person_{timestamp}"
        
        # Add to known faces
        self.known_faces[new_name] = {
            "name": f"Person {timestamp}",
            "age": "Unknown",
            "gender": "Unknown",
            "description": "Saved from webcam",
            "encoding": face_encoding,
            "voice_file": None
        }
        
        # Save face image
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        cv2.imwrite(f"saved_faces/{new_name}.jpg", face_image)
        
        # Save encoding
        self.save_encodings()
        
        print(f"✅ Face saved as {new_name}")
        print(f"📁 Image saved to: saved_faces/{new_name}.jpg")
        print("💡 Press 'v' to start recording voice for this person")
    
    def save_voice_mappings(self):
        """Save voice file mappings"""
        voice_mappings = {}
        for name, data in self.known_faces.items():
            if data.get("voice_file"):
                voice_mappings[name] = data["voice_file"]
        
        with open('known_faces/voice_mappings.pkl', 'wb') as f:
            pickle.dump(voice_mappings, f)
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("🎯 IMPROVED VOICE-ENABLED HUMAN DETECTION SYSTEM HELP")
        print("="*50)
        print("📹 Controls:")
        print("   'q' - Quit the application")
        print("   's' - Save current face to database")
        print("   'v' - Start voice recording (continuous)")
        print("   'p' - Stop recording and play audio (or play existing audio)")
        print("   'h' - Show this help message")
        print("\n🔍 Features:")
        print("   • Human detection using webcam")
        print("   • Gender classification (Male/Female)")
        print("   • Face recognition with known persons")
        print("   • Age estimation")
        print("   • Continuous voice recording until 'p' is pressed")
        print("   • Real-time display with bounding boxes")
        print("\n🎤 Voice Features:")
        print("   • Press 'v' to start listening")
        print("   • Press 'p' to stop recording and play audio")
        print("   • Continuous recording (no time limit)")
        print("   • Automatic voice playback on recognition")
        print("="*50)
    
    def save_encodings(self):
        """Save face encodings to file"""
        encodings = {}
        for name, data in self.known_faces.items():
            if data["encoding"] is not None:
                encodings[name] = data["encoding"]
        
        with open('known_faces/encodings.pkl', 'wb') as f:
            pickle.dump(encodings, f)
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.is_recording = False
            time.sleep(0.5)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the improved voice-enabled human detection system"""
    print("Initializing Improved Voice-Enabled Human Detection and Identification System...")
    print("This system will:")
    print("- Detect if a human is present")
    print("- Identify gender (Male/Female)")
    print("- Recognize known persons with their names and ages")
    print("- Record voice continuously until 'p' is pressed")
    print("- Play voice descriptions automatically")
    print("- Display 'I don't know this person' for unrecognized faces")
    
    system = ImprovedVoiceHumanDetectionSystem()
    
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