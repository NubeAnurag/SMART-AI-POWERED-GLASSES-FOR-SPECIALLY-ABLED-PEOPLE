#!/usr/bin/env python3
"""
===============================================================================
FEATURE: HUMAN DETECTION AND IDENTIFICATION (Feature 2) - Configuration
===============================================================================
This file belongs to the Human Detection and Identification feature module.

WORK:
- Contains all configuration settings for the human detection system
- Defines known persons database with names, ages, genders, descriptions
- System configuration: camera index, face detection confidence, FPS display
- Display settings: window title, colors, font settings, line thickness
- Emotion display mappings for better visualization

CONFIGURATION SECTIONS:
- KNOWN_PERSONS: Predefined database of known persons
- SYSTEM_CONFIG: Camera settings, detection thresholds, system behavior
- DISPLAY_CONFIG: UI appearance settings (colors, fonts, window title)
- EMOTION_DISPLAY: Emotion emoji mappings for visual feedback

This file centralizes all configuration, making it easy to modify system
behavior without changing the main code.

Author: DRDO Project
===============================================================================
"""
# Configuration file for Human Detection and Identification System

# Known persons database with hardcoded names and ages
KNOWN_PERSONS = {
    "anurag_mandal": {
        "name": "Anurag Mandal",
        "age": 22,
        "gender": "Male",
        "description": "System User"
    },
    "john_doe": {
        "name": "John Doe",
        "age": 28,
        "gender": "Male",
        "description": "Software Engineer"
    },
    "jane_smith": {
        "name": "Jane Smith",
        "age": 25,
        "gender": "Female", 
        "description": "Data Scientist"
    },
    "mike_wilson": {
        "name": "Mike Wilson",
        "age": 32,
        "gender": "Male",
        "description": "Project Manager"
    },
    "sarah_johnson": {
        "name": "Sarah Johnson",
        "age": 29,
        "gender": "Female",
        "description": "UX Designer"
    },
    "david_brown": {
        "name": "David Brown",
        "age": 35,
        "gender": "Male",
        "description": "Team Lead"
    },
    "emma_davis": {
        "name": "Emma Davis",
        "age": 27,
        "gender": "Female",
        "description": "Product Manager"
    }
}

# System configuration
SYSTEM_CONFIG = {
    "camera_index": 0,  # Webcam index (working index)
    "face_detection_confidence": 0.5,  # Face recognition confidence threshold
    "display_fps": True,  # Show FPS on screen
    "save_faces": True,  # Allow saving new faces
    "max_faces": 10,  # Maximum number of faces to detect simultaneously
}

# Display settings
DISPLAY_CONFIG = {
    "window_title": "Human Detection and Identification System",
    "text_color": (255, 255, 255),  # White text
    "box_color": (0, 255, 0),  # Green bounding box
    "unknown_color": (0, 0, 255),  # Red for unknown persons
    "font_scale": 0.6,
    "font_thickness": 2,
    "line_thickness": 2
}

# Emotion mapping for better display
EMOTION_DISPLAY = {
    "happy": "😊 Happy",
    "sad": "😢 Sad", 
    "angry": "😠 Angry",
    "fear": "😨 Fear",
    "surprise": "😲 Surprise",
    "disgust": "🤢 Disgust",
    "neutral": "😐 Neutral"
} 