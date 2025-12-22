# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
python3 setup.py
```

### 2. Grant Camera Permissions (macOS)
- Go to **System Preferences > Security & Privacy > Privacy**
- Select **Camera** from the left sidebar
- Make sure **Terminal** or **Python** is checked
- If not, click the lock icon and add it

### 3. Run the System
```bash
python3 run_system.py
```

## 🎯 What You'll See

- **Green box**: Recognized person with name and age
- **Red box**: Unknown person with estimated gender/age
- **"No human detected"**: When no face is visible

## ⌨️ Controls

- **'q'**: Quit the application
- **'s'**: Save current face to database
- **'h'**: Show help

## 📊 System Features

✅ **Human Detection**: Detects if a human face is present  
✅ **Gender Classification**: Identifies Male/Female  
✅ **Face Recognition**: Recognizes known persons  
✅ **Age Display**: Shows hardcoded ages for known persons  
✅ **Real-time Display**: Live video with overlays  
✅ **Face Saving**: Save new faces to database  

## 🔧 Known Persons (Hardcoded)

The system comes with 6 pre-configured persons:
- John Doe (28, Male)
- Jane Smith (25, Female)  
- Mike Wilson (32, Male)
- Sarah Johnson (29, Female)
- David Brown (35, Male)
- Emma Davis (27, Female)

## 💡 Tips for Best Results

- Ensure good lighting
- Keep face clearly visible
- Look directly at the camera
- Stay within frame boundaries

---

**Note**: This is a simplified version that works with current dependencies. For advanced features like emotion detection, additional setup may be required. 