#!/usr/bin/env python3
"""
===============================================================================
FEATURE: TEXT OCR RECOGNITION (Feature 1)
===============================================================================
This file belongs to the OCR (Optical Character Recognition) feature module.

WORK:
- Real-time text extraction from camera feed or images
- Uses dual OCR engines: EasyOCR and Tesseract for better accuracy
- Supports English and Hindi (Devanagari) text recognition
- Preprocesses images (grayscale, contrast enhancement, CLAHE)
- Flags unrecognized characters for quality control
- Provides side-by-side comparison of both OCR results
- Text-to-speech output using pyttsx3 with Siri-like male voice
- Saves OCR output to text file
- Can work with live camera feed or static image files

KEY FUNCTIONS:
- preprocess_image(): Enhances image for better OCR accuracy
- flag_unrecognized(): Identifies characters that may be misread
- clean_text(): Removes empty lines and noise
- list_available_cameras(): Scans and lists available camera devices

CONTROLS:
- SPACE: Capture and process current frame for OCR
- ESC: Exit OCR feature and return to main menu

OUTPUTS:
- ocr_output.txt: Text file with OCR results from both engines
- debug_preprocessed.png: Preprocessed image for debugging
- Console output with recognized text
- Audio output via text-to-speech

Author: DRDO Project
===============================================================================
"""
# Python 3.13, OpenCV version will be printed at runtime
import cv2
import numpy as np
import easyocr
import tkinter as tk
from tkinter import messagebox
import re
import pytesseract
from PIL import Image
import subprocess
import time
from copy import deepcopy
import os
import requests
import pyttsx3

# NLP postprocessing removed - using raw OCR output only

# --- Camera Functions ---
def list_available_cameras():
    """List all available camera devices"""
    print("Scanning for available cameras...")
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: Available")
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"  Resolution: {width}x{height}")
            cap.release()
        else:
            print(f"Camera {i}: Not available")

# --- Image Processing Functions ---
def preprocess_image(frame):
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Slightly darken (make text stand out)
    dark = cv2.convertScaleAbs(gray, alpha=1.0, beta=-30)
    # Gentle contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced = clahe.apply(dark)
    cv2.imwrite('debug_preprocessed.png', enhanced)
    return enhanced

# Add a function to flag unrecognized characters
def flag_unrecognized(text):
    # English letters, numerals, and Hindi (Devanagari) letters
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
    # Devanagari Unicode block: \u0900-\u097F
    def is_allowed(c):
        return c in allowed or ('\u0900' <= c <= '\u097F')
    flagged = ''
    allowed_punct = "\n\r.,;:!?-()[]{}\"':/\\|@#$%^&*_+=~`<>"
    for c in text:
        if is_allowed(c) or c in allowed_punct:
            flagged += c
        else:
            flagged += f'[{c}]'  # flag unrecognized
    return flagged

# --- NLP Cleanup ---
def clean_text(text):
    # Remove lines with no letters or numbers
    lines = text.split('\n')
    cleaned = [line for line in lines if re.search(r'[A-Za-z0-9]', line)]
    return '\n'.join(cleaned)

# Remove PaddleOCR import and related code
# --- Main Live Camera OCR ---
def main():
    print(f"Python version: {subprocess.getoutput('python3 --version')}")
    print(f"OpenCV version: {cv2.__version__}")
    use_image_file = os.getenv('USE_IMAGE_FILE', '0') == '1'
    if use_image_file:
        print("[INFO] Using debug_preprocessed.png as input (Docker test mode)")
        frame = cv2.imread('debug_preprocessed.png')
        if frame is None:
            print("[ERROR] debug_preprocessed.png not found or cannot be read.")
            return
        preprocessed = frame  # Already preprocessed
        print("Running EasyOCR...")
        t0 = time.time()
        easy_text, tess_text = '', ''
        try:
            easyocr_start = time.time()
            reader = easyocr.Reader(['en', 'hi'])
            easy_result = reader.readtext(preprocessed)
            easy_text = '\n'.join([item[1] for item in easy_result])
            print(f"EasyOCR done in {time.time() - easyocr_start:.2f} seconds.")
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        print("Running Tesseract...")
        try:
            tess_start = time.time()
            pil_img = Image.fromarray(preprocessed)
            tess_text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6', lang='eng+hin')
            print(f"Tesseract done in {time.time() - tess_start:.2f} seconds.")
        except Exception as e:
            print(f"Tesseract failed: {e}")
        print(f"Total OCR time: {time.time() - t0:.2f} seconds.")
        flagged_easy = flag_unrecognized(clean_text(easy_text))
        flagged_tess = flag_unrecognized(clean_text(tess_text))
        print("\n===== EASYOCR OUTPUT =====\n")
        print(flagged_easy)
        print("\n===== TESSERACT OUTPUT =====\n")
        print(flagged_tess)
        print("\n==========================\n")
        with open('ocr_output.txt', 'w') as f:
            f.write("===== EASYOCR OUTPUT =====\n" + flagged_easy + "\n\n")
            f.write("===== TESSERACT OUTPUT =====\n" + flagged_tess + "\n\n")
        def show_ocr_results_side_by_side(easy_text, tess_text):
            easy_lines = easy_text.split('\n')
            tess_lines = tess_text.split('\n')
            max_lines = max(len(easy_lines), len(tess_lines))
            easy_lines += [''] * (max_lines - len(easy_lines))
            tess_lines += [''] * (max_lines - len(tess_lines))
            width = 600
            height = 30 * (max_lines + 2)
            img = np.ones((height, width, 3), dtype=np.uint8) * 30
            cv2.putText(img, 'EasyOCR', (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(img, 'Tesseract', (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            for i in range(max_lines):
                y = 60 + i*30
                cv2.putText(img, easy_lines[i], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(img, tess_lines[i], (310, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('OCR Results Side by Side (press any key to continue)', img)
            cv2.waitKey(0)
            cv2.destroyWindow('OCR Results Side by Side (press any key to continue)')
        show_ocr_results_side_by_side(flagged_easy, flagged_tess)
        subprocess.Popen(['open', 'ocr_output.txt'])
        # Speak the combined output with male Siri-like voice
        engine = pyttsx3.init()
        
        # Get available voices and set to male voice
        voices = engine.getProperty('voices')
        male_voice = None
        
        # Look for male voices (preferably Siri-like)
        for voice in voices:
            if 'male' in voice.name.lower() or 'daniel' in voice.name.lower() or 'alex' in voice.name.lower():
                male_voice = voice
                break
        
        # If no male voice found, use the first available voice
        if male_voice:
            engine.setProperty('voice', male_voice.id)
        else:
            # Use default voice but try to make it sound more like Siri
            engine.setProperty('rate', 150)  # Slightly slower rate
            engine.setProperty('volume', 0.9)  # High volume
        
        # Set speech properties for Siri-like quality
        engine.setProperty('rate', 150)  # Words per minute
        engine.setProperty('volume', 0.9)  # Volume level
        
        combined_text = f"EasyOCR output: {flagged_easy}. Tesseract output: {flagged_tess}."
        engine.say(combined_text)
        engine.runAndWait()
        return
    # Try different camera indices to find iPhone camera
    cap = None
    for camera_index in [0, 1, 2, 3]:  # Try multiple camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera opened successfully at index {camera_index}")
            break
    
    if not cap or not cap.isOpened():
        print("Cannot open any camera. Trying alternative method...")
        # Try using AVFoundation backend which might work better with iPhone
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Still cannot open camera. Please check camera permissions.")
            return
    # List available cameras first
    list_available_cameras()
    print("\nPress SPACE to capture, ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Live Feed - Press SPACE to Capture', frame)
        k = cv2.waitKey(1)
        if k%256 == 27:  # ESC pressed
            break
        elif k%256 == 32:  # SPACE pressed
            preprocessed = preprocess_image(frame)
            print("Running EasyOCR...")
            t0 = time.time()
            easy_text, tess_text = '', ''
            try:
                easyocr_start = time.time()
                reader = easyocr.Reader(['en', 'hi'])
                easy_result = reader.readtext(preprocessed)
                easy_text = '\n'.join([item[1] for item in easy_result])
                print(f"EasyOCR done in {time.time() - easyocr_start:.2f} seconds.")
            except Exception as e:
                print(f"EasyOCR failed: {e}")
            print("Running Tesseract...")
            try:
                tess_start = time.time()
                pil_img = Image.fromarray(preprocessed)
                tess_text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6', lang='eng+hin')
                print(f"Tesseract done in {time.time() - tess_start:.2f} seconds.")
            except Exception as e:
                print(f"Tesseract failed: {e}")
            print(f"Total OCR time: {time.time() - t0:.2f} seconds.")
            flagged_easy = flag_unrecognized(clean_text(easy_text))
            flagged_tess = flag_unrecognized(clean_text(tess_text))
            print("\n===== EASYOCR OUTPUT =====\n")
            print(flagged_easy)
            print("\n===== TESSERACT OUTPUT =====\n")
            print(flagged_tess)
            print("\n==========================\n")
            with open('ocr_output.txt', 'w') as f:
                f.write("===== EASYOCR OUTPUT =====\n" + flagged_easy + "\n\n")
                f.write("===== TESSERACT OUTPUT =====\n" + flagged_tess + "\n\n")
            # NLP postprocessing removed - showing raw OCR output only
            def show_ocr_results_side_by_side(easy_text, tess_text):
                easy_lines = easy_text.split('\n')
                tess_lines = tess_text.split('\n')
                max_lines = max(len(easy_lines), len(tess_lines))
                easy_lines += [''] * (max_lines - len(easy_lines))
                tess_lines += [''] * (max_lines - len(tess_lines))
                width = 600
                height = 30 * (max_lines + 2)
                img = np.ones((height, width, 3), dtype=np.uint8) * 30
                cv2.putText(img, 'EasyOCR', (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(img, 'Tesseract', (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                for i in range(max_lines):
                    y = 60 + i*30
                    cv2.putText(img, easy_lines[i], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(img, tess_lines[i], (310, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow('OCR Results Side by Side (press any key to continue)', img)
                cv2.waitKey(0)
                cv2.destroyWindow('OCR Results Side by Side (press any key to continue)')
            show_ocr_results_side_by_side(flagged_easy, flagged_tess)
            subprocess.Popen(['open', 'ocr_output.txt'])
            # Speak the combined output with male Siri-like voice
            engine = pyttsx3.init()
            
            # Get available voices and set to male voice
            voices = engine.getProperty('voices')
            male_voice = None
            
            # Look for male voices (preferably Siri-like)
            for voice in voices:
                if 'male' in voice.name.lower() or 'daniel' in voice.name.lower() or 'alex' in voice.name.lower():
                    male_voice = voice
                    break
            
            # If no male voice found, use the first available voice
            if male_voice:
                engine.setProperty('voice', male_voice.id)
            else:
                # Use default voice but try to make it sound more like Siri
                engine.setProperty('rate', 150)  # Slightly slower rate
                engine.setProperty('volume', 0.9)  # High volume
            
            # Set speech properties for Siri-like quality
            engine.setProperty('rate', 150)  # Words per minute
            engine.setProperty('volume', 0.9)  # Volume level
            
            combined_text = f"EasyOCR output: {flagged_easy}. Tesseract output: {flagged_tess}."
            engine.say(combined_text)
            engine.runAndWait()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 