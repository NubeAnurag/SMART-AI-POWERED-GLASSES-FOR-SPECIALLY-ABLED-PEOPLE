# 🧠 Smart Voice-Enabled Human Detection and Identification System

## 🎯 Overview

This advanced system combines face recognition with intelligent voice processing to:
- **Detect humans** using webcam
- **Recognize known persons** and display their information
- **Extract structured information** from voice recordings
- **Speak natural summaries** about recognized persons
- **Learn relationships** and personal details through voice

## 🚀 Quick Start

### 1. Installation
```bash
python3 setup.py
```

### 2. Run the System
```bash
python3 run_smart_voice_system.py
```

### 3. Grant Permissions (macOS)
- **Camera**: System Preferences > Security & Privacy > Privacy > Camera
- **Microphone**: System Preferences > Security & Privacy > Privacy > Microphone

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save current face to database |
| `v` | Start voice recording (continuous) |
| `p` | Stop recording and process voice (or speak summary) |
| `h` | Show help message |

## 🧠 Smart Voice Workflow

### Step 1: Face Recognition
1. Look at the camera
2. System detects and recognizes your face
3. Displays your basic information (name, age, gender)

### Step 2: Voice Recording
1. Press `v` to start recording
2. See "🎙️ Listening..." in red text
3. Speak your details (no time limit)
4. Press `p` to stop recording

### Step 3: Information Extraction
1. System converts speech to text
2. Extracts structured information:
   - **Name**: "my name is", "I'm", "I am"
   - **Age**: "22 years old", "age 22"
   - **Relationship**: friend, family, colleague, classmate
   - **How you know each other**: "we study together", "work together"
   - **Interests**: "love", "like", "enjoy", "interested in"

### Step 4: Natural Speech Output
1. System generates a natural summary
2. Speaks the summary using text-to-speech
3. Saves information for future recognition

## 💡 Example Voice Input

**Speak this:**
```
"Hi, my name is Anurag Mandal, I'm 22 years old, 
I'm your friend and we study Computer Science together 
in Section A. I love AI and machine learning projects."
```

**System extracts:**
- Name: Anurag Mandal
- Age: 22
- Relationship: friend
- How you know each other: study Computer Science together
- Interests: AI and machine learning projects

**System speaks:**
```
"This is Anurag Mandal who is 22 years old and is my friend. 
we study Computer Science together together. 
they are interested in AI and machine learning projects."
```

## 🔧 Technical Features

### Voice Processing
- **Speech Recognition**: Google Speech Recognition API
- **Text-to-Speech**: pyttsx3 with natural voice
- **Information Extraction**: Pattern matching and NLP techniques
- **Continuous Recording**: No time limits, press 'p' to stop

### Face Recognition
- **Face Detection**: OpenCV Haar Cascades
- **Face Encoding**: dlib face_recognition library
- **Gender/Age Estimation**: Computer vision heuristics
- **Real-time Processing**: Live video feed analysis

### Data Storage
- **Face Encodings**: `known_faces/encodings.pkl`
- **Voice Files**: `voice_recordings/`
- **Extracted Info**: `known_faces/extracted_info.pkl`
- **Voice Mappings**: `known_faces/voice_mappings.pkl`

## 📊 Information Extraction Patterns

### Name Detection
- "my name is [name]"
- "I'm [name]"
- "I am [name]"
- "call me [name]"
- "this is [name]"

### Age Detection
- "[number] years old"
- "age [number]"
- "I'm [number]"
- "I am [number]"

### Relationship Detection
- **Friend**: "friend", "buddy", "pal", "mate"
- **Family**: "son", "daughter", "father", "mother", "brother", "sister"
- **Colleague**: "colleague", "coworker", "work together"
- **Classmate**: "classmate", "study together", "same class"
- **Neighbor**: "neighbor", "live nearby"

### Interest Detection
- "love [interest]"
- "like [interest]"
- "enjoy [interest]"
- "interested in [interest]"
- "passionate about [interest]"

## 🎯 Use Cases

### Personal Assistant
- Remember people you meet
- Learn about their interests
- Build relationship database
- Natural conversation starter

### Security System
- Recognize authorized persons
- Voice-based authentication
- Personal information recall
- Access control

### Social Networking
- Remember meeting details
- Track relationships
- Personal information management
- Conversation memory

## 🔍 Troubleshooting

### Camera Issues
```bash
# Test camera access
python3 test_camera.py

# Check camera permissions
# System Preferences > Security & Privacy > Privacy > Camera
```

### Microphone Issues
```bash
# Test microphone
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print('Microphone available')"

# Check microphone permissions
# System Preferences > Security & Privacy > Privacy > Microphone
```

### Speech Recognition Issues
- Ensure clear speech
- Good microphone quality
- Quiet environment
- Internet connection (for Google Speech API)

### Text-to-Speech Issues
- Check system volume
- Verify pyttsx3 installation
- Test with: `python3 -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait()"`

## 📁 File Structure

```
human detection and identification drdo/
├── smart_voice_system.py          # Main system file
├── run_smart_voice_system.py      # Runner script
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── known_faces/
│   ├── encodings.pkl             # Face encodings
│   ├── voice_mappings.pkl        # Voice file mappings
│   └── extracted_info.pkl        # Extracted voice information
├── voice_recordings/              # Audio files
└── saved_faces/                   # Face images
```

## 🎨 Customization

### Adding New Information Types
Edit `parse_voice_text()` in `smart_voice_system.py`:
```python
# Add new patterns
new_patterns = [
    r"pattern to match",
    r"another pattern"
]

# Extract information
for pattern in new_patterns:
    match = re.search(pattern, text)
    if match:
        info["new_field"] = match.group(1)
        break
```

### Modifying Speech Output
Edit `generate_spoken_summary()` in `smart_voice_system.py`:
```python
# Customize summary format
if info.get("new_field"):
    summary_parts.append(f"custom text {info['new_field']}")
```

### Changing Voice Settings
Edit TTS configuration in `__init__()`:
```python
self.tts_engine.setProperty('rate', 150)      # Speed (words per minute)
self.tts_engine.setProperty('volume', 0.9)    # Volume (0.0 to 1.0)
```

## 🔒 Privacy and Security

### Data Storage
- All data stored locally
- No cloud processing (except Google Speech API)
- Voice files in WAV format
- Face encodings encrypted in pickle files

### Permissions
- Camera access required for face detection
- Microphone access required for voice recording
- No internet required for face recognition
- Internet required for speech recognition

### Data Management
- Delete voice files: `rm voice_recordings/*`
- Delete face data: `rm known_faces/*.pkl`
- Reset system: Delete all data files

## 🚀 Performance Tips

### Optimal Settings
- Good lighting for face detection
- Clear speech for voice recognition
- Quiet environment for recording
- High-quality microphone

### System Requirements
- Python 3.8+
- Webcam with 720p+ resolution
- Microphone with noise cancellation
- 4GB+ RAM for smooth operation

### Troubleshooting Performance
- Reduce video resolution if laggy
- Close other applications
- Check CPU/memory usage
- Restart system if needed

## 📝 License

This project is for educational and personal use. Please respect privacy and obtain consent before recording anyone's voice or face.

---

**🎯 Ready to experience the future of human-computer interaction!** 