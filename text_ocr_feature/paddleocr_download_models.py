#!/usr/bin/env python3
"""
===============================================================================
FEATURE: TEXT OCR RECOGNITION (Feature 1) - Model Downloader
===============================================================================
This file belongs to the OCR (Optical Character Recognition) feature module.

WORK:
- Downloads PaddleOCR models for offline use
- Ensures all required OCR models are available locally
- Prevents runtime delays by pre-downloading models
- Creates dummy image and runs prediction to trigger downloads
- Supports English language OCR models

PURPOSE:
This utility script should be run once before using OCR features to ensure
all models are downloaded and cached locally. This improves performance
and allows offline operation.

USAGE:
    python3 paddleocr_download_models.py

Author: DRDO Project
===============================================================================
"""
from paddleocr import PaddleOCR
import numpy as np

# Step 1: Create a blank image (white, 3-channel)
dummy_img = np.ones((100, 300, 3), dtype=np.uint8) * 255

# Step 2: Initialize OCR (this will download all models for English)
print("Initializing PaddleOCR and downloading models (this may take several minutes)...")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# Step 3: Run predict to trigger all downloads
print("Running a dummy prediction to ensure all models are downloaded...")
ocr.predict(dummy_img)

print("PaddleOCR models downloaded and ready for offline use.") 