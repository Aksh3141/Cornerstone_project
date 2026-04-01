#!/bin/bash

echo "🚀 Setting up Aegis AI backend environment..."

python3 -m venv venv

echo "✅ Virtual environment created"

source venv/bin/activate

echo "✅ Virtual environment activated"

pip install --upgrade pip

pip install fastapi uvicorn

pip install numpy scipy

pip install torch torchvision torchaudio

pip install tensorflow

pip install transformers safetensors

pip install openai-whisper

pip install librosa moviepy opencv-python

pip install python-multipart

pip install tim

pip install langdetect

pip install sentencepiece

pip install easyocr

echo "🎉 Setup complete!"

echo "👉 To activate later: source venv/bin/activate"
echo "👉 To run backend: uvicorn main:app --reload"