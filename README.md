# 🛡️ Aegis AI — Multimodal Video Content Moderation

Aegis AI is a multimodal content moderation system that analyzes videos using **text, audio, and visual signals** to detect harmful content such as:

- Hate Speech
- Violence
- Sexual Content
- Neutral Content

It combines deep learning models across multiple modalities and produces:
- Final classification
- Confidence scores
- Segment-wise analysis with timestamps

---

## 🔥 Features

- 🎥 Video upload and preview
- 🧠 Multimodal AI analysis (Text + Audio + Vision)
- ⏱️ Timestamp-based segmentation
- 📊 Confidence scores across modalities
- 📈 Interactive frontend dashboard
- ⚖️ Multimodal fusion
- 🔍 Optional OCR pipeline for text-in-video
- 🧩 Segment-level explainability
- 📈 Dataset evaluation pipeline

---

## 🧠 How It Works

### Pipeline Overview

```
Video
↓
Audio Extraction
↓
Transcription (Whisper)
↓
Segment Generation (with gap filling)
↓
For each segment:
  → Text Model (RoBERTa)
  → Audio Model (CNN)
  → Vision Model (YOLO + CNN)
↓
Segment-wise results
↓
Modality aggregation (backend)
↓
Weighted multimodal fusion
↓
Final verdict + confidence
```
---

## 🧱 Project Structure
```
Cornerstone_Project/
│
├── backend/
│   ├── models/
│   │   ├── audio/
│   │   │   ├── audio_moderation_model.h5
│   │   │   └── inference.py
│   │   │
│   │   ├── vision/
│   │   │   ├── violence_best.pth
│   │   │   ├── nudity_best.pth
│   │   │   ├── inference.py
│   │   │   └── model.py
│   │   │
│   │   └── text/
│   │       ├── inference.py
│   │       ├── extract_frames.py
│   │       ├── ocr_processor.py
│   │       └── roberta/
│   │           ├── config.json
│   │           ├── tokenizer.json
│   │           ├── tokenizer_config.json
│   │           └── model.safetensors
│   │
│   ├── services/
│   │   └── pipeline.py
│   │
│   ├── routes/
│   │   └── moderation.py
│   │
│   ├── utils/
│   │   ├── media.py
│   │   └── transcription.py
│   │
│   ├── uploads/
│   ├── temp/
│   └── main.py
│   └── evaluation.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── assets/
│   │
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```


---

## ⚙️ Installation & Setup

### 🔹 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```
### 🔹 2. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 🔹 3. Install Required System Dependencies
Make sure you have:

Python 3.9+
FFmpeg (required for audio/video processing)
```bash
sudo apt install ffmpeg
```

### 🔹 4. Run Backend
```bash
uvicorn main:app --reload
```
Backend will run at:
```bash
http://127.0.0.1:8000
```

### 🔹 5. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend will run at:
```bash
http://localhost:5173
```
---

# 📦 Model Download (IMPORTANT)

Due to size constraints, model files are not included in the repository.

👉 Download all models from the link below:

🔗 **[Download Models Here](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**


## 📁 After Download, Place Files Like This:

```
backend/models/
├── audio/
│   └── audio_moderation_model.h5
│
├── vision/
│   └── best_model.pth
│
└── text/
    └── roberta/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── model.safetensors
```
---

## ⚖️ Fusion Strategy

Aegis AI uses weighted multimodal fusion:

- Hate Speech → Text dominant
- Violence → Vision-heavy
- Sexual Content → Vision + Audio
- Neutral → Balanced

Final scores are normalized to form a valid probability distribution.

---

## 📊 Output Format
```
The backend returns:
{
  "verdict": "violence",
  "confidence": 0.82,

  "final_scores": {
    "neutral": 0.1,
    "violence": 0.7,
    "sexual_content": 0.1,
    "hate_speech": 0.1
  },

  "modalities": {
    "text": {...},
    "audio": {...},
    "vision": {...}
  },

  "segments": [
    {
      "start": 2.0,
      "end": 6.5,
      "modalities": {
        "text": {...},
        "audio": {...},
        "vision": {...}
      }
    }
  ]
}
```
---

## 🚧 Current Limitations

- Models are not SOTA (research prototype)
- Processing is slow due to per-segment inference
- OCR is global (not timestamp-aligned)
- No real-time streaming support yet

---

## 🔮 Future Improvements

- ⚡ Optimize pipeline (FFmpeg, batching)
- 🎯 Smarter fusion (Ensemble Learning/MLP)
- ⏱️ Timestamp-level UI interaction
- 🌐 Scalable deployment (GPU inference)
- 📡 Live-stream moderation support 

---

## 👨‍💻 Authors

- Ankur  
- Siddharth  
- Yuvraj Verma  
- Jayesh  
- Hrishav  

---

## 📜 License

This project is for academic/research purposes.

---

## ⭐ Acknowledgements

- OpenAI Whisper  
- HuggingFace Transformers  
- YOLO-based vision models  
- TensorFlow / PyTorch ecosystem  

---