#!/usr/bin/env python3
"""
===============================================================================
FEATURE: HUMAN DETECTION AND IDENTIFICATION (Feature 2) - Debug Tool
===============================================================================
This file belongs to the Human Detection and Identification feature module.

WORK:
- Debugging utility for face recognition system
- Tests face encoding generation and matching
- Validates face recognition accuracy
- Helps diagnose issues with face encodings
- Compares face encodings from different photos
- Useful for troubleshooting recognition problems

PURPOSE:
This is a diagnostic tool used during development and troubleshooting to
verify that face encodings are being generated correctly and matching
properly.

USAGE:
    python3 debug_recognition.py

This tool helps identify and fix issues with face recognition accuracy.

Author: DRDO Project
===============================================================================
"""

import cv2
import face_recognition
import os
import pickle
import numpy as np
from config import SYSTEM_CONFIG

def debug_recognition():
    """Debug the face recognition system"""
    print("🔍 Debugging Face Recognition System")
    print("=" * 50)
    
    # Check if encodings file exists
    if os.path.exists('known_faces/encodings.pkl'):
        print("✅ Encodings file found")
        with open('known_faces/encodings.pkl', 'rb') as f:
            saved_encodings = pickle.load(f)
            print(f"📊 Found {len(saved_encodings)} saved encodings:")
            for name, encoding in saved_encodings.items():
                print(f"   - {name}: {type(encoding)}")
    else:
        print("❌ No encodings file found")
    
    # Check if person info exists
    if os.path.exists('known_faces/person_info.pkl'):
        print("✅ Person info file found")
        with open('known_faces/person_info.pkl', 'rb') as f:
            person_info = pickle.load(f)
            print(f"📊 Found {len(person_info)} persons:")
            for name, info in person_info.items():
                print(f"   - {name}: {info}")
    else:
        print("❌ No person info file found")
    
    # Check photos
    print("\n📸 Checking saved photos:")
    for person_dir in os.listdir('person_photos'):
        person_path = f"person_photos/{person_dir}"
        if os.path.isdir(person_path):
            photos = os.listdir(person_path)
            print(f"   - {person_dir}: {len(photos)} photos")
            for photo in photos:
                print(f"     * {photo}")
    
    # Test face detection from saved photos
    print("\n🔍 Testing face detection from saved photos:")
    for person_dir in os.listdir('person_photos'):
        person_path = f"person_photos/{person_dir}"
        if os.path.isdir(person_path):
            for photo in os.listdir(person_path):
                photo_path = f"{person_path}/{photo}"
                try:
                    image = face_recognition.load_image_file(photo_path)
                    face_encodings = face_recognition.face_encodings(image)
                    print(f"   ✅ {photo_path}: {len(face_encodings)} faces detected")
                except Exception as e:
                    print(f"   ❌ {photo_path}: Error - {e}")
    
    # Recreate encodings from photos
    print("\n🔄 Recreating encodings from photos...")
    known_faces = {}
    
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
                        print(f"   ✅ Added encoding from {photo}")
                except Exception as e:
                    print(f"   ❌ Error processing {photo}: {e}")
            
            if encodings:
                known_faces[person_id] = {
                    "encoding": encodings[0],  # Primary encoding
                    "encodings": encodings     # All encodings
                }
                print(f"   ✅ {person_id}: {len(encodings)} encodings saved")
    
    # Save updated encodings
    if known_faces:
        encodings_to_save = {}
        for name, data in known_faces.items():
            encodings_to_save[name] = data["encoding"]
        
        with open('known_faces/encodings.pkl', 'wb') as f:
            pickle.dump(encodings_to_save, f)
        print(f"\n✅ Saved {len(encodings_to_save)} encodings to file")
    
    # Test live recognition
    print("\n📹 Testing live recognition...")
    print("Position yourself in front of the camera and press 'q' to quit")
    
    cap = cv2.VideoCapture(SYSTEM_CONFIG["camera_index"])
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get face encodings
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            # Draw rectangle
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Try to recognize
            recognized_name = None
            best_distance = 1.0
            
            for name, data in known_faces.items():
                if data["encoding"] is not None:
                    # Compare with all encodings
                    for encoding in data.get("encodings", [data["encoding"]]):
                        match = face_recognition.compare_faces([encoding], face_encoding, tolerance=0.6)
                        distance = face_recognition.face_distance([encoding], face_encoding)[0]
                        
                        if match[0] and distance < best_distance:
                            recognized_name = name
                            best_distance = distance
            
            # Display result
            if recognized_name:
                cv2.putText(frame, f"Recognized: {recognized_name}", (left, top - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {1-best_distance:.2f}", (left, top - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"✅ Recognized: {recognized_name} (confidence: {1-best_distance:.2f})")
            else:
                cv2.putText(frame, "Unknown", (left, top - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Debug Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Debug completed!")

if __name__ == "__main__":
    debug_recognition() 