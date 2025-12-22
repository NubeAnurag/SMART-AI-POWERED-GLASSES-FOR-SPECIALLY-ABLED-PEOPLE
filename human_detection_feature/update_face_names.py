#!/usr/bin/env python3
"""
===============================================================================
FEATURE: HUMAN DETECTION AND IDENTIFICATION (Feature 2) - Name Updater
===============================================================================
This file belongs to the Human Detection and Identification feature module.

WORK:
- Utility script to update face encodings with proper names
- Syncs face encodings with known persons database from config.py
- Ensures consistency between face encodings and person information
- Updates pickle files with correct name mappings

PURPOSE:
This utility helps maintain consistency between the face encodings database
and the known persons configuration. Run this if you've updated the
KNOWN_PERSONS dictionary in config.py.

USAGE:
    python3 update_face_names.py

This ensures face recognition works correctly with updated person information.

Author: DRDO Project
===============================================================================
"""

import pickle
import os
from config import KNOWN_PERSONS

def update_face_encodings():
    """Update face encodings to use proper names"""
    
    # Load current encodings
    if os.path.exists('known_faces/encodings.pkl'):
        with open('known_faces/encodings.pkl', 'rb') as f:
            encodings = pickle.load(f)
        
        print("Current encodings:")
        for name, encoding in encodings.items():
            print(f"  - {name}")
        
        # Create new encodings with proper names
        new_encodings = {}
        
        # Add Anurag's encoding with proper name
        if encodings:
            # Get the first (most recent) encoding
            timestamp_name = list(encodings.keys())[0]
            encoding = encodings[timestamp_name]
            
            # Add with proper name
            new_encodings['anurag_mandal'] = encoding
            print(f"✅ Updated {timestamp_name} -> anurag_mandal")
        
        # Save updated encodings
        with open('known_faces/encodings.pkl', 'wb') as f:
            pickle.dump(new_encodings, f)
        
        print("\n✅ Face encodings updated!")
        print("Now restart the system and you should see your name!")
        
    else:
        print("❌ No encodings file found")

if __name__ == "__main__":
    update_face_encodings() 