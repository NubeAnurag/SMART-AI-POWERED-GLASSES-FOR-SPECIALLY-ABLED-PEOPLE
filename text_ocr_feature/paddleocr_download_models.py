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