# Real-Time Object Detection with YOLO

A real-time object detection application that uses your webcam to identify objects in the environment using YOLOv8.

## Features

- 🎥 **Real-time webcam feed** with object detection
- 🔍 **YOLOv8 model** for accurate object identification
- 📊 **Live FPS counter** for performance monitoring
- 🎯 **Bounding boxes** with object labels and confidence scores
- 📸 **Screenshot capability** to save detected objects
- 🎨 **Color-coded detection** for different object classes

## Supported Objects

YOLOv8 can detect 80+ different object classes including:
- People, animals (dogs, cats, birds, etc.)
- Vehicles (cars, trucks, buses, motorcycles)
- Common objects (chairs, tables, phones, laptops, books)
- Food items (pizza, hot dog, apple, banana)
- Sports equipment (baseball, tennis racket, skateboard)
- And many more!

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam connected to your computer
- Internet connection (for downloading YOLO model)

### Setup Steps

1. **Clone or download this project**
   ```bash
   # Navigate to the project directory
   cd "object identification drdo"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

## Usage

### Controls
- **'q' key**: Quit the application
- **'s' key**: Save a screenshot of the current frame
- **Close window**: Click the X button to exit

### What You'll See
- Live webcam feed with real-time object detection
- Green bounding boxes around detected objects
- Object labels with confidence scores (e.g., "person: 0.95")
- FPS counter in the top-left corner
- Instructions at the bottom of the screen

## Performance Tips

- **Hardware**: Better GPU/CPU will improve FPS
- **Resolution**: Lower webcam resolution for better performance
- **Model size**: The app uses YOLOv8n (nano) for speed
- **Confidence threshold**: Only objects with >50% confidence are shown

## Troubleshooting

### Common Issues

1. **"Could not open webcam"**
   - Make sure your webcam is connected and not in use by another application
   - Try changing the webcam index in the code (0, 1, 2, etc.)

2. **Low FPS**
   - Close other applications using the webcam
   - Reduce webcam resolution in the code
   - Ensure you have a decent GPU/CPU

3. **Model download issues**
   - Check your internet connection
   - The model will be downloaded automatically on first run

4. **Permission errors**
   - Make sure your application has permission to access the webcam
   - On macOS, check System Preferences > Security & Privacy > Camera

## Technical Details

- **Model**: YOLOv8n (nano version for speed)
- **Input Resolution**: 640x480 (configurable)
- **Detection Threshold**: 0.5 (50% confidence)
- **Supported Formats**: All common webcam formats

## Customization

You can modify the code to:
- Change detection confidence threshold
- Adjust webcam resolution
- Use different YOLO models (yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- Add custom object classes
- Implement recording functionality

## Dependencies

- `ultralytics`: YOLO implementation
- `opencv-python`: Computer vision and webcam access
- `numpy`: Numerical operations
- `Pillow`: Image processing
- `torch`: PyTorch backend for YOLO

## License

This project is open source and available under the MIT License. 