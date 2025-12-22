# Human Detection and Identification System

A comprehensive real-time human detection and identification system using webcam that can detect humans, classify gender, recognize known persons, estimate age, and detect emotions.

## 🎯 Features

- **Human Detection**: Detects if a human face is present in the webcam feed
- **Gender Classification**: Identifies if the person is Male or Female
- **Face Recognition**: Recognizes known persons with their hardcoded names and ages
- **Age Estimation**: Estimates age for unknown persons
- **Emotion Detection**: Detects emotions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **Real-time Display**: Shows all information with colored bounding boxes
- **Face Saving**: Ability to save new faces to the database

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam
- macOS (tested on macOS 24.0.0)
- Camera permissions enabled for Terminal/Python

### Installation

1. **Clone or download the project files**

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Or install dependencies manually**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system**:
   ```bash
   python3 run_system.py
   ```
   
   Or run the simplified version directly:
   ```bash
   python3 human_detection_system_simple.py
   ```

## 🎮 Controls

- **'q'** - Quit the application
- **'s'** - Save current face to database
- **'h'** - Show help message

## 📊 System Behavior

### For Known Persons:
- Shows "I recognize this person!"
- Displays hardcoded name and age
- Shows gender and emotion
- Green bounding box
- Confidence level

### For Unknown Persons:
- Shows "I don't know this person"
- Displays detected gender and estimated age
- Shows emotion with emoji
- Red bounding box

### When No Human Detected:
- Shows "No human detected" message

## ⚙️ Configuration

The system uses `config.py` for configuration:

### Known Persons Database
```python
KNOWN_PERSONS = {
    "john_doe": {
        "name": "John Doe",
        "age": 28,
        "gender": "Male",
        "description": "Software Engineer"
    },
    # Add more persons here...
}
```

### System Settings
```python
SYSTEM_CONFIG = {
    "camera_index": 0,  # Webcam index
    "face_detection_confidence": 0.6,  # Recognition threshold
    "display_fps": True,  # Show FPS
    "save_faces": True,  # Allow saving faces
    "max_faces": 10,  # Max faces to detect
}
```

## 📁 Project Structure

```
human detection and identification drdo/
├── human_detection_system.py  # Main application
├── config.py                  # Configuration file
├── setup.py                   # Setup script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── known_faces/              # Face encodings database
├── saved_faces/              # Saved face images
└── logs/                     # System logs
```

## 🔧 Technical Details

### Libraries Used
- **OpenCV**: Computer vision and webcam handling
- **face_recognition**: Face detection and recognition
- **DeepFace**: Age, gender, and emotion analysis
- **dlib**: Face detection and encoding
- **TensorFlow/Keras**: Deep learning models
- **NumPy**: Numerical computations

### Models
- **Face Detection**: Haar Cascade Classifier
- **Face Recognition**: dlib's face recognition model
- **Age/Gender/Emotion**: DeepFace pre-trained models

## 🎨 Display Features

- **Color-coded bounding boxes**:
  - 🟢 Green: Recognized person
  - 🔴 Red: Unknown person
- **Emotion emojis**: 😊 Happy, 😢 Sad, 😠 Angry, etc.
- **Real-time FPS counter**
- **Overlay text with person information**

## 📝 Adding Known Persons

1. **Edit `config.py`**:
   ```python
   "new_person": {
       "name": "New Person Name",
       "age": 30,
       "gender": "Male",
       "description": "Description"
   }
   ```

2. **Run the system and press 's'** when the person is in front of the camera

3. **The face encoding will be automatically saved**

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not working**:
   - Check if webcam is connected
   - Ensure camera permissions are granted for Terminal/Python
   - On macOS: System Preferences > Security & Privacy > Privacy > Camera
   - Try changing `camera_index` in config.py

2. **Dependencies installation issues**:
   - On macOS, you might need to install Xcode command line tools:
     ```bash
     xcode-select --install
     ```

3. **Face recognition not working**:
   - Ensure good lighting
   - Face should be clearly visible
   - Try adjusting `face_detection_confidence` in config.py

4. **Performance issues**:
   - Reduce `max_faces` in config.py
   - Disable FPS display by setting `display_fps: False`

### Error Messages

- **"Could not open webcam"**: Check webcam connection
- **"No face detected"**: Ensure face is visible and well-lit
- **"Error in face analysis"**: Try adjusting lighting or position

## 🔒 Privacy and Security

- All face data is stored locally
- No data is transmitted to external servers
- Face images are saved only when explicitly requested
- System can be run completely offline

## 📈 Performance Tips

- Ensure good lighting conditions
- Keep face clearly visible to camera
- Close unnecessary applications to free up CPU
- Use a modern computer for better performance

## 🤝 Contributing

Feel free to improve the system by:
- Adding more known persons to the database
- Adjusting detection parameters
- Improving the UI/UX
- Adding new features

## 📄 License

This project is for educational and research purposes.

---

**Note**: This system is designed for the DRDO human detection and identification project. The hardcoded names and ages are for demonstration purposes and should be updated with actual known persons as needed. 