# 🎙️ Voice AI Assistant - Raspberry Pi 4

A sleek, voice-powered AI assistant built for Raspberry Pi 4 with a CustomTkinter GUI, Gemini API for transcription and refinement, and txtai + TinyLlama for intelligent question answering from documents.

---

## 📌 Features

- 🎤 Voice recording from Raspberry Pi microphone
- 🧠 Audio transcription using **Gemini 1.5 Flash**
- ✨ Refined responses focusing on **Saveetha Engineering College**
- 📄 Document-based question answering using **txtai RAG + TinyLlama**
- 🎨 CustomTkinter GUI with real-time feedback and interactive output

---

## 🚀 Setup Instructions

### ✅ Requirements

- Raspberry Pi 4 (or equivalent)
- Python 3.9+
- Microphone
- Internet connection

### 📦 Install Dependencies

```bash
sudo apt update && sudo apt upgrade
sudo apt install python3-pip portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
pip install customtkinter sounddevice scipy pillow google-generativeai txtai[all]
```

### Demo Image

![demo1](https://github.com/codebyNJ/voiceai/blob/main/demo_images/demo1.png?raw=true)

### ⚙️ How It Works
- Start Recording: Press "Start" to begin audio capture

- Transcription: Gemini API transcribes and refines the text

- Document QA: txtai RAG answers based on uploaded document

- Output: Transcript and AI response appear in GUI

### Project Structure
```
voice-ai-assistant/
│
├── main.py                # Main script with GUI + backend integration
├── elements/              # Folder with UI assets (lines, icons, etc.)
│   └── line1.png
├── recording.wav          # Temporary audio file
├── Saveetha_Engineering_College.docx  # Knowledge source for txtai
└── README.md              # This file
```

### Team Members
- Nijeesh NJ - 212223100039 (Cyber Security)
- Saravanan PV - 212223230195 (AIDS)
- Oswald Shilo - 212223040139 (CSE)
- Harish - 212223230072 (AIDS)
- Nemaleshwar H - 212223230142 (AIDS)
