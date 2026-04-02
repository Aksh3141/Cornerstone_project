import os
import re
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import MarianMTModel, MarianTokenizer

from models.text.extract_frames import extract_smart_frames
from models.text.ocr_processor import run_ocr_on_frames

from langdetect import detect

CLASSES = ["neutral", "sexual_content", "violence", "hate_speech"]

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "roberta")

TEMP_OCR_DIR = os.path.join("temp", "ocr_frames")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("🧠 Loading Text Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
text_model.eval()

print("🌍 Loading Translation Model...")

try:
    trans_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    trans_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    print("✅ Translation model loaded")
except:
    trans_tokenizer = None
    trans_model = None

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_garbage_text(text):
    if not text or len(text.split()) < 2:
        return True

    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > 0.4:
        return True

    return False


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def translate_to_english(text):
    if trans_model is None or trans_tokenizer is None:
        return text

    try:
        inputs = trans_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        translated = trans_model.generate(**inputs)
        return trans_tokenizer.decode(translated[0], skip_special_tokens=True)
    except:
        return text


def extract_ocr_text(video_path):
    try:
        os.makedirs(TEMP_OCR_DIR, exist_ok=True)

        frames = extract_smart_frames(video_path,TEMP_OCR_DIR,max_frames=3)

        if not frames:
            return ""

        ocr_results = run_ocr_on_frames(frames)
        texts = [t for t, _ in ocr_results if t.strip()]

        return " ".join(texts)

    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

KEYWORDS = {
    "violence": ["kill", "murder", "shoot", "attack", "die", "blood"],
    "sexual_content": ["sex", "nude", "porn", "xxx", "naked"],
    "hate_speech": ["hate", "racist", "slur"]
}

def keyword_override(text):
    text = text.lower()

    scores = {k: 0.0 for k in CLASSES}

    for label, words in KEYWORDS.items():
        for w in words:
            if w in text:
                scores[label] += 0.2

    return scores

def classify_text(text):

    if not text.strip():
        return {
            "neutral": 1.0,
            "sexual_content": 0.0,
            "violence": 0.0,
            "hate_speech": 0.0
        }

    raw = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 3]

    if not sentences:
        sentences = [text]

    all_probs = []

    for i in range(0, len(sentences), 16):
        batch = sentences[i:i + 16]

        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = text_model(**inputs).logits

        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.extend(probs)

    n = len(all_probs)

    return {
        "neutral": float(sum(p[0] for p in all_probs) / n),
        "sexual_content": float(sum(p[3] for p in all_probs) / n),
        "violence": float(sum(p[1] for p in all_probs) / n),
        "hate_speech": float(sum(p[2] for p in all_probs) / n)
    }

def predict_text(text, ocr_text=None):

    try:
        text = clean_text(text)

        if is_garbage_text(text):
            text = ""

        if text:
            lang = detect_language(text)

            if lang == "hi":
                text = translate_to_english(text)

        ocr_text = clean_text(ocr_text or "")
        combined = f"{text} {ocr_text}".strip()

        print("TRANSCRIPT:", text)
        print("OCR TEXT:", ocr_text)
        print("COMBINED:", combined)

        if not text and not ocr_text:
            return {
                "neutral": 1.0,
                "sexual_content": 0.0,
                "violence": 0.0,
                "hate_speech": 0.0
            }

        base_scores = classify_text(combined)

        keyword_scores = keyword_override(combined)

        for k in base_scores:
            base_scores[k] += keyword_scores[k]

        total = sum(base_scores.values())
        if total > 0:
            for k in base_scores:
                base_scores[k] /= total

        return base_scores

    except Exception as e:
        print(f"❌ predict_text error: {e}")

        return {
            "neutral": 1.0,
            "sexual_content": 0.0,
            "violence": 0.0,
            "hate_speech": 0.0
        }