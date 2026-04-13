
# 🎤 AI-Powered Voice Bot for Customer Support Automation

## 📌 Project Overview
This project implements an end-to-end AI-powered voice bot capable of handling customer support queries using speech interaction.

---

## ⚙️ System Architecture
User Audio → Speech-to-Text (Whisper) → Intent Classification → Response → Text-to-Speech → Audio Output

---

## 🛠️ Tech Stack
- Whisper (ASR)
- Scikit-learn (ML Model)
- gTTS (Text-to-Speech)
- FastAPI (Backend)
- Python

---

## 🚀 Features
- Accepts voice input
- Converts speech to text
- Predicts intent
- Generates response
- Converts response to speech
- REST API support

---

## 🔌 API Endpoint

### POST /voicebot

Input: Audio file  
Output:

{
  "transcription": "I want a refund",
  "intent": "refund_request",
  "confidence": 0.29,
  "response": "Your refund will be processed soon.",
  "audio_file": "output.mp3"
}

---

## ⚙️ Setup

pip install fastapi uvicorn openai-whisper gTTS scikit-learn python-multipart

Run:
uvicorn app:app --reload

Open:
http://127.0.0.1:8000/docs

---

## 📂 Project Structure

Intern Project/
│
├── app.py
├── voicebot.ipynb
├── input.m4a
├── output.mp3
├── README.md

---

## 🎯 Conclusion
This project demonstrates a complete AI voice-based customer support system using ML and APIs.

---

## 👤 Author
Shyam Sundar
