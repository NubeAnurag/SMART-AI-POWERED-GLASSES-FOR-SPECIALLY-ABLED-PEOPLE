# 🎤 Voice-Enabled Human Detection System Guide

## 🚀 How to Use Voice Features

### **Step 1: Run the Voice-Enabled System**
```bash
python3 run_voice_system.py
```

### **Step 2: Grant Microphone Permissions (macOS)**
- Go to **System Preferences > Security & Privacy > Privacy**
- Select **Microphone** from the left sidebar
- Make sure **Terminal** or **Python** is checked
- If not, click the lock icon and add it

### **Step 3: Record Your Voice Description**

1. **Look at the camera** so the system recognizes you
2. **Press 'v'** to start voice recording
3. **Speak your details** for 5 seconds:
   ```
   "Hi, my name is Anurag Mandal, I'm 22 years old, 
   I'm your friend and we study together in Section A. 
   I love working on AI projects."
   ```
4. **Wait for confirmation** - you'll see:
   ```
   🎤 Recording voice for anurag_mandal...
   💬 Please speak your details (name, age, relationship, etc.)
   ⏱️  Recording for 5 seconds...
   ✅ Recording completed!
   💾 Voice saved to: voice_recordings/anurag_mandal_20250731_232510.wav
   ✅ Voice recorded for anurag_mandal
   ```

### **Step 4: Test Voice Playback**

1. **Look at the camera** again
2. **Press 'p'** to play your recorded voice
3. **Hear your description** being played back

## 🎮 **Voice Controls**

- **'v'** - Record voice for current face
- **'p'** - Play voice for current face
- **'s'** - Save current face to database
- **'q'** - Quit the application
- **'h'** - Show help

## 📊 **What You'll See**

### **With Voice Recorded:**
```
🟢 [Green Box]
I recognize this person!
Name: Anurag Mandal
Age: 22
Gender: Male
Emotion: 😐 Neutral
Confidence: 0.85
🎤 Voice available
```

### **Without Voice:**
```
🟢 [Green Box]
I recognize this person!
Name: Anurag Mandal
Age: 22
Gender: Male
Emotion: 😐 Neutral
Confidence: 0.85
🎤 No voice recorded
```

## 💡 **Voice Recording Tips**

- **Speak clearly** and at a normal pace
- **Include key details**: name, age, relationship, interests
- **5 seconds** is enough for a good description
- **Good microphone** quality improves results
- **Quiet environment** for better recording

## 📁 **Files Created**

- **Voice recordings**: `voice_recordings/anurag_mandal_YYYYMMDD_HHMMSS.wav`
- **Voice mappings**: `known_faces/voice_mappings.pkl`
- **Face images**: `saved_faces/person_YYYYMMDD_HHMMSS.jpg`
- **Face encodings**: `known_faces/encodings.pkl`

## 🎯 **Example Voice Scripts**

### **For Friends:**
```
"Hi, I'm Anurag Mandal, 22 years old. We're friends from college, 
we study Computer Science together in Section A. I love AI and 
machine learning projects."
```

### **For Family:**
```
"This is Anurag, I'm 22 years old, your son. I study Computer 
Science and I'm working on this human detection project. 
I love technology and coding."
```

### **For Colleagues:**
```
"Hello, I'm Anurag Mandal, 22 years old. We work together on 
AI projects. I'm a software developer and we're building this 
face recognition system."
```

## 🔧 **Troubleshooting**

### **Microphone not working:**
- Check microphone permissions
- Ensure Terminal has microphone access
- Try a different microphone

### **Voice not playing:**
- Check if voice file exists
- Ensure speakers are on
- Try pressing 'p' again

### **Recording issues:**
- Speak louder
- Reduce background noise
- Check microphone connection

---

**🎤 Now you can speak your personal details and the system will remember and play them back when it recognizes you!** 