#!/usr/bin/env python3
"""
===============================================================================
FEATURE: TEXT OCR RECOGNITION (Feature 1) - TTS Variant
===============================================================================
This file belongs to the OCR (Optical Character Recognition) feature module.

WORK:
- Alternative OCR implementation with enhanced Text-to-Speech integration
- Uses Photo Booth integration for better image capture on macOS
- Combines EasyOCR and Tesseract for dual-engine text recognition
- Enhanced TTS (Text-to-Speech) output with voice configuration
- GUI-based interface using tkinter for better user interaction
- Handles Photo Booth process management for seamless image capture

KEY FEATURES:
- Photo Booth integration for high-quality image capture
- Enhanced TTS voice selection and configuration
- GUI dialog boxes for user interaction
- Process management to handle Photo Booth lifecycle

This is an alternative implementation focusing on TTS enhancement and
Photo Booth integration, particularly useful for macOS systems.

Author: DRDO Project
===============================================================================
"""
import tkinter as tk
from tkinter import messagebox
import cv2
import pytesseract
from PIL import Image
import pyttsx3
import numpy as np
import tempfile
import os
import subprocess
import easyocr
import time

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Global variable to keep track of Photo Booth process
photo_booth_process = None

# Function to open Photo Booth
def open_photo_booth():
    global photo_booth_process
    # Open Photo Booth using macOS 'open' command
    subprocess.Popen(["open", "-a", "Photo Booth"])

def close_photo_booth():
    # Try to close Photo Booth (optional, user can close manually)
    subprocess.call(["osascript", "-e", 'quit app "Photo Booth"'])

def preprocess_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize (upscale for small text)
    scale_percent = 150  # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    # Save debug image for inspection
    cv2.imwrite('debug_preprocessed.png', gray)
    return gray

def capture_and_read_text():
    # Step 1: Open Photo Booth
    open_photo_booth()
    # Step 2: Show Continue button
    continue_btn = tk.Button(root, text="Continue", font=("Arial", 14), command=lambda: proceed_to_capture(continue_btn))
    continue_btn.pack(expand=True, pady=10)

def proceed_to_capture(continue_btn):
    # Remove the Continue button
    continue_btn.pack_forget()
    # Step 1: Capture image from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image.")
        return
    # Step 2: Wait for 3 seconds before closing Photo Booth and processing
    time.sleep(3)
    close_photo_booth()
    # Step 3: Show 'Ready to start processing?' popup
    messagebox.showinfo("Ready", "Ready to start processing?")
    # Step 4: Show 'Processing, please wait...' popup
    processing_popup = tk.Toplevel(root)
    processing_popup.title("Processing")
    tk.Label(processing_popup, text="Processing, please wait...", font=("Arial", 14)).pack(padx=20, pady=20)
    processing_popup.update()
    # Step 5: Preprocess image for better OCR
    preprocessed = preprocess_image(frame)
    cv2.imwrite('debug_preprocessed.png', preprocessed)
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(preprocessed)
        text = '\n'.join([item[1] for item in result])
    except Exception as e:
        processing_popup.destroy()
        messagebox.showerror("OCR Error", f"Failed to process image with OCR.\n{e}")
        return
    processing_popup.destroy()
    messagebox.showinfo("Detected Text", text if text.strip() else "(No text detected)")
    if text.strip():
        tts_engine.say(text)
        tts_engine.runAndWait()

# Set up GUI
root = tk.Tk()
root.title("Camera OCR to Speech")
root.geometry("350x200")

capture_btn = tk.Button(root, text="Start Capturing", command=capture_and_read_text, font=("Arial", 14))
capture_btn.pack(expand=True, pady=40)

root.mainloop() 