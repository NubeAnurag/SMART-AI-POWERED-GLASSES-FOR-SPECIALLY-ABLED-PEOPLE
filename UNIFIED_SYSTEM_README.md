# Unified DRDO System

This is a unified program that combines all three DRDO features into a single menu-driven application.

## Features

1. **OCR Text Recognition** - Extract text from images/video using EasyOCR and Tesseract
2. **Human Detection and Identification** - Recognize and identify people using face recognition
3. **Object/Environment Analysis** - Analyze objects and environment using YOLOv8

## How to Run

```bash
python3 unified_drdo_system.py
```

Or make it executable and run directly:
```bash
chmod +x unified_drdo_system.py
./unified_drdo_system.py
```

## Menu Options

- **1** - OCR Text Recognition Feature
  - Press **SPACE** to capture and process text
  - Press **ESC** to exit and return to main menu

- **2** - Human Detection and Identification Feature
  - Press **'a'** to add a new person
  - Press **'p'** to speak info about recognized person
  - Press **'q'** to exit and return to main menu
  - Press **'h'** for help

- **3** - Object/Environment Analysis Feature (YOLOv8)
  - Press **'q'** to quit and return to main menu
  - Press **'s'** to save screenshot
  - Press **'p'** to speak detections
  - Press **'c'** to change confidence
  - Press **'t'** to toggle tracking
  - Press **'m'** to toggle smoothing
  - Press **'b'** to toggle confidence boost

- **0** - Exit the program

## Requirements

Make sure all dependencies are installed for each feature:

1. **OCR Feature** dependencies (from `text_ocr_feature/requirements.txt`)
2. **Human Detection** dependencies (from `human_detection_feature/requirements.txt`)
3. **Object Detection** dependencies (from `object_detection_feature/requirements.txt`)

## Directory Structure

```
DDRDDO/
├── unified_drdo_system.py          # Main unified program
├── text_ocr_feature/               # OCR feature folder
├── human_detection_feature/        # Human detection feature folder
└── object_detection_feature/       # Object detection feature folder
```

## Notes

- Each feature runs independently and returns to the main menu when exited
- The program handles directory changes automatically for each feature
- All OpenCV windows are properly cleaned up when switching between features
- Press Ctrl+C to exit the program at any time

