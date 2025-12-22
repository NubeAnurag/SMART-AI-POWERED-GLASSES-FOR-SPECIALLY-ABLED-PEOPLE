#!/usr/bin/env python3
"""
===============================================================================
FEATURE: HUMAN DETECTION AND IDENTIFICATION (Feature 2) - Smart Voice System
===============================================================================
This file belongs to the Human Detection and Identification feature module.

WORK:
- Advanced voice-enabled system with speech recognition
- Voice commands for system control
- Speech-to-text using speech_recognition library
- Text-to-speech output with pyttsx3
- Interactive voice commands (e.g., "add person", "speak info")
- Dual-mode identification: Face recognition + Voice commands
- Natural language interaction with the system

KEY CLASS: SmartVoiceHumanDetectionSystem

KEY FEATURES:
- Voice command recognition
- Natural language interaction
- Speech-to-text for commands
- Text-to-speech for responses
- Interactive voice-based control

KEY METHODS:
- listen_for_command(): Recognizes voice commands
- process_command(): Executes recognized commands
- speak_response(): Provides audio feedback
- voice_interaction(): Handles voice-based interactions

PURPOSE:
This is the most advanced version with full voice interaction capabilities,
allowing users to control the system using voice commands.

Author: DRDO Project
===============================================================================
"""
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
import speech_recognition as sr
import pyttsx3
import re
from config import KNOWN_PERSONS, SYSTEM_CONFIG, DISPLAY_CONFIG, EMOTION_DISPLAY

class SmartVoiceHumanDetectionSystem:
    def __init__(self):
        self.known_faces = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Audio settings - optimized for speech recognition
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Lower rate for better speech recognition
        
        # Voice recording state
        self.is_recording = False
        self.recording_frames = []
        self.recording_stream = None
        self.recording_person = None
        
        # Speech recognition and TTS
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS for better voice quality
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Use Karen (Siri-like English female voice) specifically
            selected_voice = None
            
            for voice in voices:
                voice_id = voice.id.lower()
                if 'karen' in voice_id:
                    selected_voice = voice.id
                    break
            
            # If Karen not found, try other good English female voices
            if not selected_voice:
                preferred_voices = ['com.apple.voice.compact.en-AU.Karen', 'com.apple.speech.synthesis.voice.victoria', 'com.apple.speech.synthesis.voice.samantha']
                for voice in voices:
                    voice_id = voice.id.lower()
                    if any(pref in voice_id for pref in preferred_voices):
                        selected_voice = voice.id
                        break
            
            # If no preferred voice found, use the first available
            if not selected_voice and voices:
                selected_voice = voices[0].id
            
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice)
                print(f"🎤 Using voice: {selected_voice}")
        
        # Optimize TTS settings for better quality
        self.tts_engine.setProperty('rate', 140)      # Speed (words per minute) - slower and more natural
        self.tts_engine.setProperty('volume', 1.0)    # Volume (0.0 to 1.0) - maximum volume
        self.tts_engine.setProperty('pitch', 1.0)     # Pitch (0.5 to 2.0) - normal pitch
        
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
                "voice_file": None,  # Will be set when voice is recorded
                "extracted_info": None,  # Will store extracted voice information
                "spoken_summary": None  # Will store the summary to speak
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
        
        # Load voice file mappings and extracted info
        if os.path.exists('known_faces/voice_mappings.pkl'):
            with open('known_faces/voice_mappings.pkl', 'rb') as f:
                voice_mappings = pickle.load(f)
                for name, voice_file in voice_mappings.items():
                    if name in self.known_faces:
                        self.known_faces[name]["voice_file"] = voice_file
        
        # Load extracted information
        if os.path.exists('known_faces/extracted_info.pkl'):
            with open('known_faces/extracted_info.pkl', 'rb') as f:
                extracted_info = pickle.load(f)
                for name, info in extracted_info.items():
                    if name in self.known_faces:
                        self.known_faces[name]["extracted_info"] = info
                        self.known_faces[name]["spoken_summary"] = self.generate_spoken_summary(info)
    
    def extract_info_from_voice(self, audio_file):
        """Extract structured information from voice recording"""
        try:
            print("🔍 Extracting information from voice...")
            
            # Configure recognizer for better accuracy
            self.recognizer.energy_threshold = 300  # Lower threshold for better detection
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8  # Longer pause threshold
            
            # Convert audio file to format suitable for speech recognition
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
            
            # Try multiple speech recognition services for better accuracy
            text = None
            
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio, language='en-US')
                print(f"📝 Google Speech Recognition: {text}")
            except sr.UnknownValueError:
                print("❌ Google couldn't understand the audio")
            except sr.RequestError as e:
                print(f"❌ Google Speech Recognition service error: {e}")
            
            # If Google fails, try with different language settings
            if not text:
                try:
                    text = self.recognizer.recognize_google(audio, language='en-IN')
                    print(f"📝 Google Speech Recognition (India): {text}")
                except:
                    pass
            
            # If still no text, try with show_all=True to see alternatives
            if not text:
                try:
                    result = self.recognizer.recognize_google(audio, show_all=True)
                    if result and 'alternative' in result:
                        alternatives = result['alternative']
                        print("🔍 Speech recognition alternatives:")
                        for i, alt in enumerate(alternatives[:3]):
                            print(f"   {i+1}. {alt['transcript']} (confidence: {alt.get('confidence', 'N/A')})")
                        # Use the first alternative
                        if alternatives:
                            text = alternatives[0]['transcript']
                            print(f"📝 Using alternative: {text}")
                except Exception as e:
                    print(f"❌ Error getting alternatives: {e}")
            
            if not text:
                print("❌ Could not recognize any speech from the audio")
                return None
            
            # Ask user if the recognized text is correct
            print(f"\n🤔 Is this what you said?")
            print(f"   '{text}'")
            print("   Press 'y' if correct, or type the correct text:")
            
            user_input = input("   ").strip()
            
            if user_input.lower() == 'y' or user_input.lower() == 'yes':
                corrected_text = text
                print("✅ Using recognized text as is")
            else:
                corrected_text = user_input
                print(f"✅ Using corrected text: '{corrected_text}'")
            
            # Extract structured information from corrected text
            extracted_info = self.parse_voice_text(corrected_text)
            
            return extracted_info
            
        except Exception as e:
            print(f"❌ Error processing voice: {e}")
            return None
    
    def parse_voice_text(self, text):
        """Parse voice text to extract structured information with enhanced accuracy"""
        text = text.lower().strip()
        print(f"🔍 Processing text: '{text}'")
        
        # Initialize extracted info
        info = {
            "name": None,
            "age": None,
            "relationship": None,
            "how_they_know_me": None,
            "interests": [],
            "additional_info": []
        }
        
        # Enhanced name extraction with better patterns
        name_patterns = [
            r"my name is (\w+(?:\s+\w+){0,3})",  # "my name is John Doe Smith"
            r"i'm (\w+(?:\s+\w+){0,3})",         # "I'm John Doe Smith"
            r"i am (\w+(?:\s+\w+){0,3})",        # "I am John Doe Smith"
            r"call me (\w+(?:\s+\w+){0,3})",     # "call me John"
            r"this is (\w+(?:\s+\w+){0,3})",     # "this is John"
            r"hi my name is (\w+(?:\s+\w+){0,3})", # "hi my name is John"
            r"hello my name is (\w+(?:\s+\w+){0,3})" # "hello my name is John"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).title().strip()
                # Clean up common issues
                name = re.sub(r'\b(and|i|am|is|the|a|an)\b', '', name, flags=re.IGNORECASE).strip()
                if name and len(name.split()) <= 3:  # Limit to 3 words max
                    info["name"] = name
                    print(f"✅ Extracted name: {name}")
                    break
        
        # Enhanced age extraction
        age_patterns = [
            r"(\d+)\s*years?\s*old",           # "22 years old"
            r"age\s*(\d+)",                    # "age 22"
            r"i'm\s*(\d+)",                    # "I'm 22"
            r"i am\s*(\d+)",                   # "I am 22"
            r"(\d+)\s*year\s*old",             # "22 year old"
            r"my age is (\d+)"                 # "my age is 22"
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                age = int(match.group(1))
                if 1 <= age <= 120:  # Reasonable age range
                    info["age"] = age
                    print(f"✅ Extracted age: {age}")
                    break
        
        # Enhanced relationship extraction
        relationship_patterns = {
            "friend": [
                r"(\w+)\s*'s\s*friend",        # "anurag's friend"
                r"i am (\w+)\s*'s\s*friend",    # "I am anurag's friend"
                r"(\w+)\s*friend",              # "anurag friend"
                r"friend\s*of\s*(\w+)",         # "friend of anurag"
            ],
            "family": [
                r"(\w+)\s*'s\s*(son|daughter|father|mother|brother|sister)",
                r"i am (\w+)\s*'s\s*(son|daughter|father|mother|brother|sister)"
            ],
            "colleague": [
                r"(\w+)\s*'s\s*colleague",
                r"work\s+with\s*(\w+)",
                r"colleague\s*of\s*(\w+)"
            ],
            "classmate": [
                r"(\w+)\s*'s\s*classmate",
                r"study\s+with\s*(\w+)",
                r"classmate\s*of\s*(\w+)"
            ]
        }
        
        for relationship_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    info["relationship"] = relationship_type
                    print(f"✅ Extracted relationship: {relationship_type}")
                    break
            if info["relationship"]:
                break
        
        # Enhanced "how they know you" extraction
        know_patterns = [
            r"i am (\w+(?:\s+\w+){0,6})",                 # "I am working on this project for DRDO"
            r"i'm (\w+(?:\s+\w+){0,6})",                   # "I'm working on this project"
            r"we\s+(\w+(?:\s+\w+){0,4})\s+together",      # "we study together"
            r"we\s+(\w+(?:\s+\w+){0,4})",                  # "we work"
            r"(\w+(?:\s+\w+){0,4})\s+with\s+(\w+)",        # "study with anurag"
            r"(\w+(?:\s+\w+){0,4})\s+project",             # "work on project"
            r"(\w+(?:\s+\w+){0,4})\s+in\s+(\w+)",          # "study in graphic era"
            r"(\w+(?:\s+\w+){0,4})\s+at\s+(\w+)",          # "study at university"
        ]
        
        for pattern in know_patterns:
            match = re.search(pattern, text)
            if match:
                how_know = match.group(1).strip()
                # Clean up the text
                how_know = re.sub(r'\b(and|i|am|is|the|a|an)\b', '', how_know, flags=re.IGNORECASE).strip()
                if how_know and len(how_know) > 2:
                    info["how_they_know_me"] = how_know
                    print(f"✅ Extracted how they know you: {how_know}")
                    break
        
        # Enhanced interests extraction
        interest_patterns = [
            r"love\s+(\w+(?:\s+\w+){0,3})",              # "love AI"
            r"like\s+(\w+(?:\s+\w+){0,3})",               # "like machine learning"
            r"enjoy\s+(\w+(?:\s+\w+){0,3})",              # "enjoy programming"
            r"interested\s+in\s+(\w+(?:\s+\w+){0,3})",    # "interested in AI"
            r"passionate\s+about\s+(\w+(?:\s+\w+){0,3})", # "passionate about coding"
        ]
        
        for pattern in interest_patterns:
            match = re.search(pattern, text)
            if match:
                interest = match.group(1).strip()
                if interest and len(interest) > 2:
                    info["interests"].append(interest)
                    print(f"✅ Extracted interest: {interest}")
        
        # Clean up and validate extracted information
        if info["name"] and len(info["name"].split()) > 3:
            # If name is too long, take only first 2-3 words
            name_words = info["name"].split()[:3]
            info["name"] = ' '.join(name_words)
        
        # Remove empty or invalid entries
        for key in list(info.keys()):
            if isinstance(info[key], str) and (not info[key] or len(info[key]) < 2):
                info[key] = None
            elif isinstance(info[key], list) and not info[key]:
                info[key] = []
        
        return info
    
    def generate_spoken_summary(self, info):
        """Generate a natural spoken summary from extracted information"""
        if not info:
            return None
        
        summary_parts = []
        
        # Start with name - clean up the name extraction
        if info.get("name"):
            name = info['name'].strip()
            # Clean up the name further
            name = re.sub(r'\b(and|i|am|is|the|a|an)\b', '', name, flags=re.IGNORECASE).strip()
            if name and len(name.split()) <= 3:
                summary_parts.append(f"This is {name}")
        
        # Add age
        if info.get("age"):
            summary_parts.append(f"who is {info['age']} years old")
        
        # Add relationship
        if info.get("relationship"):
            if info["relationship"] == "friend":
                summary_parts.append("and is my friend")
            elif info["relationship"] == "family":
                summary_parts.append("and is family")
            elif info["relationship"] == "colleague":
                summary_parts.append("and is a colleague")
            elif info["relationship"] == "classmate":
                summary_parts.append("and is a classmate")
            elif info["relationship"] == "neighbor":
                summary_parts.append("and is a neighbor")
        
        # Add how they know you - clean up the text
        if info.get("how_they_know_me"):
            how_know = info['how_they_know_me'].strip()
            # Clean up common issues and make it more natural
            how_know = re.sub(r'\b(and|i|am|is|the|a|an)\b', '', how_know, flags=re.IGNORECASE).strip()
            if how_know and len(how_know) > 2:
                # Keep the original meaning, don't change "I am working" to "we work"
                if "working on" in how_know.lower() or "work on" in how_know.lower():
                    summary_parts.append(f"they are working on this project")
                elif "study" in how_know.lower():
                    summary_parts.append(f"we study together")
                elif "work" in how_know.lower() and "together" not in how_know.lower():
                    summary_parts.append(f"they are working on this project")
                else:
                    summary_parts.append(f"we {how_know} together")
        
        # Add interests
        if info.get("interests"):
            interests = info["interests"][:2]  # Limit to 2 interests
            if interests:
                # Clean up interests
                clean_interests = []
                for interest in interests:
                    # Remove common filler words
                    interest = re.sub(r'\b(and|i|am|is|the|a|an)\b', '', interest, flags=re.IGNORECASE).strip()
                    if interest and len(interest) > 3:  # Only add if meaningful
                        clean_interests.append(interest)
                
                if clean_interests:
                    summary_parts.append(f"they are interested in {', '.join(clean_interests)}")
        
        # Add additional info
        if info.get("additional_info"):
            additional = info["additional_info"][0]  # Take first additional info
            if additional and len(additional) > 10:  # Only add if substantial
                summary_parts.append(f"also, {additional}")
        
        if summary_parts:
            summary = ". ".join(summary_parts) + "."
            return summary
        
        return None
    
    def speak_summary(self, person_name):
        """Speak the summary for a recognized person"""
        if person_name not in self.known_faces:
            return
        
        person_data = self.known_faces[person_name]
        summary = person_data.get("spoken_summary")
        
        if summary:
            print(f"🗣️  Speaking: {summary}")
            self.tts_engine.say(summary)
            self.tts_engine.runAndWait()
        else:
            # Fallback to basic info
            basic_info = f"This is {person_data['name']}, {person_data['age']} years old, {person_data['gender']}"
            print(f"🗣️  Speaking: {basic_info}")
            self.tts_engine.say(basic_info)
            self.tts_engine.runAndWait()
    
    def start_voice_recording(self, person_name):
        """Start continuous voice recording"""
        if self.is_recording:
            print("❌ Already recording! Press 'p' to stop and save.")
            return
        
        print(f"🎤 Starting voice recording for {person_name}...")
        print("💬 Speak your details now... Press 'p' to stop recording and process")
        print("💡 Example: 'Hi, my name is Anurag Mandal, I'm 22 years old, I'm your friend and we study Computer Science together in Section A. I love AI and machine learning projects.'")
        
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
    
    def stop_and_process_voice(self):
        """Stop recording and process the voice for information extraction"""
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
        
        # Extract information from voice
        extracted_info = self.extract_info_from_voice(voice_filename)
        
        if extracted_info:
            print("📊 Extracted information:")
            for key, value in extracted_info.items():
                if value:
                    print(f"   {key}: {value}")
            
            # Generate spoken summary
            spoken_summary = self.generate_spoken_summary(extracted_info)
            if spoken_summary:
                print(f"🗣️  Generated summary: {spoken_summary}")
            
            # Update the person's information
            if self.recording_person in self.known_faces:
                self.known_faces[self.recording_person]["voice_file"] = voice_filename
                self.known_faces[self.recording_person]["extracted_info"] = extracted_info
                self.known_faces[self.recording_person]["spoken_summary"] = spoken_summary
                
                self.save_voice_mappings()
                self.save_extracted_info()
                
                print(f"✅ Voice processed and information saved for {self.recording_person}")
                
                # Speak the summary
                if spoken_summary:
                    print("🔊 Speaking the summary...")
                    self.tts_engine.say(spoken_summary)
                    self.tts_engine.runAndWait()
        else:
            print("❌ Could not extract information from voice")
        
        # Clear recording state
        self.recording_frames = []
        self.recording_person = None
    
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
            
        print("Smart Voice-Enabled Human Detection System Started!")
        print("Press 'q' to quit, 's' to save face, 'v' to start recording, 'p' to stop and process")
        
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
                        
                        # Display extracted voice information if available
                        if person_data.get("extracted_info"):
                            extracted = person_data["extracted_info"]
                            name_text = f"Name: {extracted.get('name', person_data['name'])}"
                            age_text = f"Age: {extracted.get('age', person_data['age'])}"
                            
                            # Show relationship if available
                            if extracted.get('relationship'):
                                relationship_text = f"Relationship: {extracted['relationship'].title()}"
                            else:
                                relationship_text = f"Gender: {person_data['gender']}"
                            
                            # Show how they know you if available
                            if extracted.get('how_they_know_me'):
                                how_know_text = f"Context: {extracted['how_they_know_me']}"
                            else:
                                how_know_text = ""
                            
                            # Show interests if available
                            if extracted.get('interests'):
                                interests_text = f"Interests: {', '.join(extracted['interests'][:2])}"
                            else:
                                interests_text = ""
                        else:
                            name_text = f"Name: {person_data['name']}"
                            age_text = f"Age: {person_data['age']}"
                            relationship_text = f"Gender: {person_data['gender']}"
                            how_know_text = ""
                            interests_text = ""
                        
                        emotion_text = f"Emotion: {EMOTION_DISPLAY.get(analysis['emotion'], analysis['emotion'])}"
                        confidence_text = f"Confidence: {1-confidence:.2f}"
                        
                        # Voice status
                        if self.is_recording and self.recording_person == recognized_name:
                            voice_text = "🎙️  Listening..."  # Red color for recording
                            voice_color = (0, 0, 255)  # Red
                        elif person_data.get("spoken_summary"):
                            voice_text = "🗣️  Smart voice available"
                            voice_color = (0, 255, 255)  # Yellow
                        elif person_data.get("voice_file"):
                            voice_text = "🎤 Voice available"
                            voice_color = (128, 128, 128)  # Gray
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
                    
                    if relationship_text:
                        cv2.putText(frame, relationship_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                        y_position -= 25
                    
                    if how_know_text:
                        cv2.putText(frame, how_know_text, (left, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG["font_scale"], 
                                  DISPLAY_CONFIG["text_color"], DISPLAY_CONFIG["font_thickness"])
                        y_position -= 25
                    
                    if interests_text:
                        cv2.putText(frame, interests_text, (left, y_position), 
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
                self.handle_voice_processing(frame, face_encodings, face_locations_recognition)
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
                print("❌ Already recording! Press 'p' to stop and process.")
            else:
                self.start_voice_recording(recognized_name)
        else:
            print("❌ Face not recognized. Save the face first with 's' key")
    
    def handle_voice_processing(self, frame, face_encodings, face_locations):
        """Handle voice processing for the currently detected face"""
        if not face_encodings:
            print("No face detected to process voice for")
            return
        
        # If currently recording, stop and process
        if self.is_recording:
            self.stop_and_process_voice()
            return
        
        # Recognize the current face
        face_encoding = face_encodings[0]
        recognized_name, confidence = self.recognize_person(face_encoding)
        
        if recognized_name:
            person_data = self.known_faces[recognized_name]
            spoken_summary = person_data.get("spoken_summary")
            
            if spoken_summary:
                print("🗣️  Speaking saved information for recognized person...")
                threading.Thread(target=self.speak_summary, args=(recognized_name,)).start()
            else:
                # If no spoken summary, create one from extracted info
                extracted_info = person_data.get("extracted_info")
                if extracted_info:
                    print("🗣️  Creating and speaking summary from extracted information...")
                    summary = self.generate_spoken_summary(extracted_info)
                    if summary:
                        print(f"🗣️  Speaking: {summary}")
                        self.tts_engine.say(summary)
                        self.tts_engine.runAndWait()
                    else:
                        print(f"❌ Could not generate summary for {recognized_name}")
                else:
                    print(f"❌ No voice information available for {recognized_name}")
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
            "voice_file": None,
            "extracted_info": None,
            "spoken_summary": None
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
    
    def save_extracted_info(self):
        """Save extracted voice information"""
        extracted_info = {}
        for name, data in self.known_faces.items():
            if data.get("extracted_info"):
                extracted_info[name] = data["extracted_info"]
        
        with open('known_faces/extracted_info.pkl', 'wb') as f:
            pickle.dump(extracted_info, f)
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print("🎯 SMART VOICE-ENABLED HUMAN DETECTION SYSTEM HELP")
        print("="*60)
        print("📹 Controls:")
        print("   'q' - Quit the application")
        print("   's' - Save current face to database")
        print("   'v' - Start voice recording (continuous)")
        print("   'p' - Stop recording and process voice (or speak summary)")
        print("   'h' - Show this help message")
        print("\n🔍 Features:")
        print("   • Human detection using webcam")
        print("   • Gender classification (Male/Female)")
        print("   • Face recognition with known persons")
        print("   • Age estimation")
        print("   • Voice-to-text processing")
        print("   • Information extraction from voice")
        print("   • Text-to-speech summaries")
        print("\n🎤 Smart Voice Features:")
        print("   • Press 'v' to start listening")
        print("   • Press 'p' to stop recording and extract information")
        print("   • System extracts: name, age, relationship, interests")
        print("   • System speaks natural summaries about recognized persons")
        print("   • No time limit on recording")
        print("\n💡 Example Voice Input:")
        print("   'Hi, my name is Anurag Mandal, I'm 22 years old,")
        print("    I'm your friend and we study Computer Science together")
        print("    in Section A. I love AI and machine learning projects.'")
        print("="*60)
    
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
    """Main function to run the smart voice-enabled human detection system"""
    print("Initializing Smart Voice-Enabled Human Detection and Identification System...")
    print("This system will:")
    print("- Detect if a human is present")
    print("- Identify gender (Male/Female)")
    print("- Recognize known persons with their names and ages")
    print("- Extract information from voice recordings")
    print("- Speak natural summaries about recognized persons")
    print("- Display 'I don't know this person' for unrecognized faces")
    
    system = SmartVoiceHumanDetectionSystem()
    
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