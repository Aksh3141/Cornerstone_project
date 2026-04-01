#!/usr/bin/env python3
"""
test_pipeline.py  –  Video → Audio → Transcript → Hate Speech Classification
=============================================================================
Full end-to-end pipeline that:
  1. Finds all video files in a directory (organized by ground-truth label folders)
  2. Extracts audio using ffmpeg
  3. Transcribes audio using OpenAI Whisper
  4. Classifies each transcript using the trained 4-class model
  5. Compares predictions against ground-truth folder labels
  6. Generates an accuracy report

Expected folder structure:
  test_videos/
    hate/       → video1.mp4, video2.mp4 ...
    neutral/    → video3.mp4, video4.mp4 ...
    nudity/     → video5.mp4, video6.mp4 ...
    violence/   → video7.mp4, video8.mp4 ...

Usage:
  python test_pipeline.py --video_dir ./test_videos --model_dir ./results/best_model
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Our 4-class labels ──────────────────────────────────────
LABEL_NAMES = ["Neutral", "Offensive", "Hate", "Sexual"]
NUM_LABELS = 4
SEVERITY = {0: 0, 1: 1, 2: 3, 3: 2}

# ── Map folder names to our class IDs ───────────────────────
FOLDER_TO_LABEL = {
    "neutral": 0,
    "violence": 1,
    "offensive": 1,
    "hate": 2,
    "hate speech": 2,
    "nudity": 3,
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".3gp"}


# ══════════════════════════════════════════════════════════════
# Text Cleaning Pipeline
# ══════════════════════════════════════════════════════════════
import re as _re
import html as _html

def clean_transcript(text):
    """Clean Whisper transcript output.
    
    Handles:
    - HTML entities
    - URLs, @mentions
    - Excessive whitespace
    - Whisper hallucination artifacts (bracketed tags, repeated filler)
    - Non-ASCII garbage from forced-English transcription
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = _html.unescape(text)
    # Remove URLs
    text = _re.sub(r"https?://\S+", "", text)
    # Remove @mentions
    text = _re.sub(r"@\w+", "", text)
    # Remove Whisper bracket artifacts like [Music], (Music), *Music*
    text = _re.sub(r"[\[\(\*](music|laughter|applause|silence|inaudible|cheering)[\]\)\*]", "", text, flags=_re.IGNORECASE)
    # Collapse whitespace
    text = _re.sub(r"\s+", " ", text).strip()
    
    return text


def clean_ocr_text(text):
    """Clean OCR extracted text.
    
    Handles:
    - Watermark URLs (e.g., 'Clips5z.com', 'Fry99 com', 'MALUI.COM')
    - Very short garbage tokens
    - Common OCR noise patterns
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = _html.unescape(text)
    # Remove URL-like patterns (watermarks)
    text = _re.sub(r"https?://\S+", "", text)
    text = _re.sub(r"\S+\.com\S*", "", text, flags=_re.IGNORECASE)
    text = _re.sub(r"\S+\.org\S*", "", text, flags=_re.IGNORECASE)
    text = _re.sub(r"\S+\.net\S*", "", text, flags=_re.IGNORECASE)
    # Remove social media handles and watermarks
    text = _re.sub(r"@\w+", "", text)
    text = _re.sub(r"#\w+", "", text)
    # Collapse whitespace
    text = _re.sub(r"\s+", " ", text).strip()
    
    return text


def is_hallucinated(text, detected_language="en"):
    """Detect if a Whisper transcript is likely a hallucination.
    
    Returns True if the text appears to be garbage/hallucinated.
    """
    if not text or len(text.strip()) < 3:
        return True
    
    text = text.strip()
    words_raw = text.lower().split()
    
    if len(words_raw) == 0:
        return True
    
    # Clean words of punctuation for frequency analysis
    words_clean = [_re.sub(r'[^\w\s]', '', w) for w in words_raw]
    words_clean = [w for w in words_clean if w]
    
    # Pattern 1: Excessive repetition
    if len(words_clean) >= 4:
        word_counts = {}
        for w in words_clean:
            word_counts[w] = word_counts.get(w, 0) + 1
        most_common_count = max(word_counts.values())
        # If one word makes up >45% of all words, it's repetitive garbage (e.g. "I love you. I love you. I love you.")
        if most_common_count / len(words_clean) >= 0.45:
            return True
        
        # Check phrase repetition (e.g. "I love you" x4 = 12 words, 3 unique words -> ratio 0.25)
        unique_ratio = len(set(words_clean)) / len(words_clean)
        if len(words_clean) >= 10 and unique_ratio < 0.3:
            return True
            
    # Pattern 2: High non-ASCII ratio (forced English/translate on non-English audio)
    if detected_language != "en":
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        if total_chars > 0 and ascii_chars / total_chars < 0.95:
            return True
    
    # Pattern 3: Known Whisper filler hallucinations
    hallucination_patterns = [
        r"^(you|the|it|a|,|\.|book|oh)+$",  # Single/short word transcripts
        r"^(mm-hmm[\s,]*)+$", 
        r"^(uh+[\s,]*)+$", 
        r"^(oh+[\s,]*)+$", 
        r"^thank(s| you)?\.?$",  
        r"^.*supports.*stsq.*$", # Specific figure skating ghost
        r"^(supports|quin)$",
    ]
    low_text = text.lower()
    for pattern in hallucination_patterns:
        if _re.match(pattern, low_text):
            return True
            
    return False


def is_ocr_garbage(text):
    """Check if OCR text is just watermark/logo garbage with no useful content."""
    if not text or len(text.strip()) < 3:
        return True
    
    cleaned = text.strip()
    
    # Too short after cleaning
    if len(cleaned) < 5:
        return True
    
    # Mostly single characters or numbers (OCR noise)
    words = cleaned.split()
    if len(words) > 0:
        short_words = sum(1 for w in words if len(w) <= 2)
        if short_words / len(words) > 0.7:
            return True
    
    return False


# ══════════════════════════════════════════════════════════════
# Step 1: Extract audio from video using ffmpeg
# ══════════════════════════════════════════════════════════════
def extract_audio(video_path, output_dir):
    """Extract audio as WAV from a video file."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(output_dir, f"{basename}.wav")

    if os.path.exists(wav_path):
        return wav_path

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # WAV format
        "-ar", "16000",           # 16kHz (Whisper expects this)
        "-ac", "1",               # mono
        wav_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed for {video_path}: {result.stderr.decode()[:200]}")
        return None
    return wav_path


# ══════════════════════════════════════════════════════════════
# Step 2: Transcribe audio using Whisper
# ══════════════════════════════════════════════════════════════
def transcribe_audio(wav_path, whisper_model):
    """Transcribe a WAV file using the loaded Whisper model. Returns (text, language).
    
    Strategy: 
    1. Detect language first (cheap)
    2. If English → transcribe normally
    3. If non-English → transcribe in DETECTED language first, then translate
       This avoids Whisper hallucinating fake English words from non-English audio.
    """
    import whisper
    # Step 1: Detect language (fast — only reads first 30s of audio)
    audio = whisper.load_audio(wav_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    lang_confidence = probs.get(detected_language, 0)
    
    # Step 2: Transcribe based on detected language
    if detected_language == "en" or lang_confidence < 0.5:
        # English or uncertain → transcribe as English
        result = whisper_model.transcribe(wav_path, language="en", fp16=torch.cuda.is_available())
    else:
        # Non-English → use Whisper's translation mode to get English
        # This produces much better results than forcing language="en" on non-English audio
        result = whisper_model.transcribe(wav_path, task="translate", fp16=torch.cuda.is_available())
    
    raw_text = result.get("text", "").strip()
    
    # Step 3: Clean the transcript
    text = clean_transcript(raw_text)
    
    # Step 4: Check for hallucinations
    if is_hallucinated(text, detected_language):
        print(f"      ⚠️  Hallucination detected, discarding: \"{text[:60]}\"")
        text = ""
    
    return text, detected_language


# ══════════════════════════════════════════════════════════════
# Step 3: Classify transcript (reused from inference.py)
# ══════════════════════════════════════════════════════════════
import re

def split_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = [s.strip() for s in raw if s.strip() and len(s.strip()) > 3]
    if not sentences:
        sentences = [text.strip()]
    return sentences

# --- Toxic Trigger Keywords (The "Silent" Word Rule) ---
TOXIC_KEYWORDS = {
    "hate": [
        "nigger", "nigga", "faggot", "kike", "spik", "chink", "retard", "subhuman",
        "terrorist", "infidel", "parasite", "holocaust", "hitler"
    ],
    "offensive": [
        "fuck", "bitch", "shit", "dick", "pussy", "asshole", "bastard",
        # Violence-specific keywords (from CSV audit)
        "shooting", "gunshot", "shootout", "bullets", "blast", "stabbing",
        "guns", "gunfire", "open fire", "wounding"
    ]
}

def check_keywords(text):
    """Returns 'Hate', 'Offensive' or None if keywords found."""
    low_text = text.lower()
    for word in TOXIC_KEYWORDS["hate"]:
        if word in low_text: return "Hate"
    for word in TOXIC_KEYWORDS["offensive"]:
        if word in low_text: return "Offensive"
    return None

# --- Music Detection Helper ---
MUSIC_INDICATORS = ["music", "♪", "🎵", "we don't talk anymore", "bring back memories"]

def is_music_content(text):
    """Detect if transcript is primarily music/song lyrics."""
    low = text.lower()
    # Check for Whisper's MUSIC tag
    if low.startswith("music") or "music " in low[:30]:
        return True
    # Check for known music indicators
    for indicator in MUSIC_INDICATORS:
        if indicator in low:
            return True
    # Heuristic: short, repetitive phrases often = song lyrics  
    words = low.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:  # Less than 40% unique words = very repetitive
            return True
    return False

def classify_sentences(sentences, tokenizer, model, device, max_length=128, batch_size=16):
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu()
        for j in range(len(batch)):
            pred_id = probs[j].argmax().item()
            results.append({
                "sentence": batch[j],
                "label": LABEL_NAMES[pred_id],
                "label_id": pred_id,
                "confidence": round(probs[j][pred_id].item(), 4),
                "scores": {LABEL_NAMES[k]: round(probs[j][k].item(), 4) for k in range(NUM_LABELS)},
            })
    return results


def detect_safe_topic(text: str) -> bool:
    """Detect if the text is likely about a generally safe but 'aggressive' topic 
    like Gaming, Sports, Action Movies, or Medical Documentaries."""
    low_text = text.lower()
    gaming_sports = ["game", "play", "hp", "level", "player", "match", "cannon", "field", "wicket", "ball", "wheel", "race", "speed", "score", "damage"]
    doc_food = ["health", "recipe", "spicy", "taste", "food", "painful", "doctor", "history", "mystery", "death", "mysteriously", "medical"]
    
    sports_matches = sum(1 for w in gaming_sports if w in low_text)
    doc_matches = sum(1 for w in doc_food if w in low_text)
    
    return sports_matches >= 2 or doc_matches >= 2


def aggregate_video_score(audio_text, ocr_text, sentence_results, detected_language="en"):
    if not sentence_results:
        # Check raw text for keywords even if sentences failed
        kw = check_keywords(audio_text + " " + ocr_text)
        if kw: return {"label": kw, "label_id": FOLDER_TO_LABEL[kw.lower()] if kw.lower() in FOLDER_TO_LABEL else (2 if kw=="Hate" else 1), "confidence": 1.0}
        return {"label": "Neutral", "label_id": 0, "confidence": 1.0}

    n = len(sentence_results)
    avg_scores = {name: 0.0 for name in LABEL_NAMES}
    for r in sentence_results:
        for name, score in r["scores"].items():
            avg_scores[name] += score / n

    # Find raw highest probability label
    video_label_id = max(range(NUM_LABELS), key=lambda i: avg_scores[LABEL_NAMES[i]])
    video_label = LABEL_NAMES[video_label_id]

    # --- Use a "Normalizer" for confidence display ---
    confidence = round(avg_scores[video_label], 4)

    # --- Rule 1: Keyword "Silent" Trigger (Extreme Sensitivity for explicit slurs) ---
    keyword_trigger = check_keywords(audio_text + " " + ocr_text)
    if keyword_trigger:
        video_label = keyword_trigger
        video_label_id = 2 if video_label == "Hate" else 1
        return {"label": video_label, "label_id": video_label_id, "confidence": 1.0, "avg_scores": avg_scores}

    # --- Rule 2: Music & Language Shields ---
    # If transcript is music/song lyrics → force Neutral (songs about love ≠ Sexual)
    if is_music_content(audio_text) and not check_keywords(audio_text + " " + ocr_text):
        return {"label": "Neutral", "label_id": 0, "confidence": round(avg_scores["Neutral"], 4),
                "avg_scores": {k: round(v, 4) for k, v in avg_scores.items()}, "total_sentences": n}

    # Adaptive Neutral Shield based on detected language
    # Non-English audio = more Whisper hallucinations = need higher bar
    NEUTRAL_SHIELD = 0.80 if detected_language != "en" else 0.70
    if avg_scores["Neutral"] < NEUTRAL_SHIELD:
        if avg_scores["Hate"] > 0.15:
            video_label = "Hate"
            video_label_id = 2
        elif avg_scores["Offensive"] > 0.25:
            video_label = "Offensive"
            video_label_id = 1

    # --- Rule 3: Density Voting & Topic Shield ---
    CONFIDENCE_THRESHOLD = 0.50
    strong_flags = {name: 0 for name in LABEL_NAMES}
    for r in sentence_results:
        if r["confidence"] > CONFIDENCE_THRESHOLD:
            strong_flags[r["label"]] += 1
            
    hate_density = strong_flags["Hate"] / n
    offensive_density = (strong_flags["Offensive"] + strong_flags["Hate"]) / n
    
    is_safe_topic = detect_safe_topic(audio_text + " " + ocr_text)
    req_hate_density = 0.25 if is_safe_topic else 0.15 # 15% sentences must be Hate
    req_off_density = 0.50 if is_safe_topic else 0.30  # 30% sentences must be Offensive

    if hate_density >= req_hate_density:
        video_label_id = 2
        video_label = "Hate"
    elif offensive_density >= req_off_density:
        video_label_id = 1
        video_label = "Offensive"

    # Single-sentence override for EXTREME hate (Slurs / High-Confidence > 0.85)
    for r in sentence_results:
        if r["label_id"] == 2 and r["confidence"] > 0.85:
            video_label_id = 2
            video_label = "Hate"
            
    # Allow pure average to override Native Hate/Offensive predictions
    native_max = max(range(NUM_LABELS), key=lambda i: avg_scores[LABEL_NAMES[i]])
    if SEVERITY[native_max] > SEVERITY[video_label_id]:
        video_label_id = native_max
        video_label = LABEL_NAMES[native_max]

    return {
        "label": video_label,
        "label_id": video_label_id,
        "confidence": confidence if video_label == LABEL_NAMES[max(range(NUM_LABELS), key=lambda i: avg_scores[LABEL_NAMES[i]])] else round(avg_scores[video_label], 4),
        "avg_scores": {k: round(v, 4) for k, v in avg_scores.items()},
        "total_sentences": n
    }


def classify_transcript(text, tokenizer, model, device):
    sentences = split_sentences(text)
    sentence_results = classify_sentences(sentences, tokenizer, model, device)
    # Pass the raw text so we can check for "Silent Keywords"
    video_result = aggregate_video_score(text, "", sentence_results)
    return video_result, sentence_results


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True, help="Root dir with subfolders: hate/, neutral/, nudity/, violence/")
    ap.add_argument("--model_dir", default="./results/best_model", help="Path to trained classifier model")
    ap.add_argument("--whisper_model", default="base", help="Whisper model size: tiny, base, small, medium, large")
    ap.add_argument("--output", default="test_results.json", help="Output JSON file")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # ── Load Whisper ────────────────────────────────────────
    print("=" * 60)
    print("  Loading Whisper model ...")
    print("=" * 60)
    import whisper
    whisper_model = whisper.load_model(args.whisper_model, device=device)
    print(f"  Whisper '{args.whisper_model}' loaded on {device}")

    # ── Load Classifier ─────────────────────────────────────
    print(f"\n  Loading classifier from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    classifier = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
    print(f"  Classifier loaded on {device}")

    # ── Discover videos ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Scanning {args.video_dir} for videos ...")
    print(f"{'=' * 60}")

    all_results = []
    correct, total = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    audio_tmp = os.path.join(args.video_dir, "_audio_tmp")

    for folder_name in sorted(os.listdir(args.video_dir)):
        folder_path = os.path.join(args.video_dir, folder_name)
        if not os.path.isdir(folder_path) or folder_name.startswith("_"):
            continue

        ground_truth_id = FOLDER_TO_LABEL.get(folder_name.lower())
        if ground_truth_id is None:
            print(f"  [SKIP] Unknown folder: {folder_name}")
            continue

        ground_truth_label = LABEL_NAMES[ground_truth_id]
        videos = [f for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]

        print(f"\n  📁 {folder_name}/ ({len(videos)} videos) → Ground truth: {ground_truth_label}")

        for vf in sorted(videos):
            video_path = os.path.join(folder_path, vf)
            total += 1
            class_total[ground_truth_label] += 1

            # Step 1: Extract audio
            print(f"    🎬 {vf}")
            wav_path = extract_audio(video_path, audio_tmp)
            if wav_path is None:
                print(f"      ❌ Audio extraction failed, skipping")
                continue

            # Step 2: Transcribe (Audio)
            print(f"      🎤 Transcribing Audio ...")
            audio_transcript, detected_language = transcribe_audio(wav_path, whisper_model)
            if audio_transcript:
                print(f"      📝 Audio: \"{audio_transcript[:60]}...\"")
                if detected_language != "en":
                    print(f"      🌐 Language: {detected_language} (non-English shield active)")

            # Step 2.5: OCR (Visual)
            print(f"      🖼️  Processing Visual OCR ...")
            from extract_frames import extract_smart_frames
            from ocr_processor import run_ocr_on_frames
            
            frame_dir = os.path.join(audio_tmp, "frames")
            extracted_frames = extract_smart_frames(video_path, frame_dir)
            
            ocr_texts = []
            if extracted_frames:
                ocr_texts = run_ocr_on_frames(extracted_frames)
            
            ocr_plain_texts = [text for text, ts in ocr_texts]
            ocr_transcript_raw = " ".join(ocr_plain_texts).strip()
            # Clean OCR text
            ocr_transcript = clean_ocr_text(ocr_transcript_raw)
            if is_ocr_garbage(ocr_transcript) and not check_keywords(ocr_transcript):
                if ocr_transcript:
                    print(f"      🗑️  OCR garbage discarded: \"{ocr_transcript[:60]}\"")
                ocr_transcript = ""
            elif ocr_transcript:
                print(f"      👁️  OCR: \"{ocr_transcript[:60]}...\"")

            # Step 3: Classify Independently
            audio_result = {"label": "Neutral", "label_id": 0, "confidence": 0.0}
            if audio_transcript and len(audio_transcript) >= 3:
                _, audio_sentences = classify_transcript(audio_transcript, tokenizer, classifier, device)
                audio_result = aggregate_video_score(audio_transcript, "", audio_sentences, detected_language)
            
            ocr_result = {"label": "Neutral", "label_id": 0, "confidence": 0.0}
            if ocr_transcript and len(ocr_transcript) >= 3:
                # Short OCR Filter: Skip MODEL classification for very short text (logos, watermarks)
                # BUT still check keywords (e.g., if OCR reads "F*ck" from a meme, we must catch it)
                if len(ocr_transcript) < 10 and not check_keywords(ocr_transcript):
                    print(f"      [SKIP] OCR too short ({len(ocr_transcript)} chars), ignoring: \"{ocr_transcript}\"")
                else:
                    _, ocr_sentences = classify_transcript(ocr_transcript, tokenizer, classifier, device)
                    ocr_result = aggregate_video_score("", ocr_transcript, ocr_sentences)

            # Step 4: Aggregate Final Video Label (Most Severe of the two)
            # Use severity ranking: Hate(3) > Sexual(2) > Offensive(1) > Neutral(0)
            if SEVERITY[ocr_result["label_id"]] > SEVERITY[audio_result["label_id"]]:
                final_label = ocr_result["label"]
                final_id = ocr_result["label_id"]
                final_conf = ocr_result["confidence"]
                trigger = "Visual (OCR)"
            else:
                final_label = audio_result["label"]
                final_id = audio_result["label_id"]
                final_conf = audio_result["confidence"]
                trigger = "Audio"

            # Compare with Ground Truth
            is_correct = (final_id == ground_truth_id)
            if is_correct:
                correct += 1
                class_correct[ground_truth_label] += 1
                marker = "✅"
            else:
                marker = "❌"

            # Display
            print(f"      🎙️  Audio Label: {audio_result['label']} ({int(audio_result.get('confidence',0)*100)}%)")
            if ocr_transcript:
                print(f"      👁️  OCR Label:   {ocr_result['label']} ({int(ocr_result.get('confidence',0)*100)}%)")
            print(f"      {marker} FINAL Prediction: {final_label} | Ground Truth: {ground_truth_label} (via {trigger})")

            all_results.append({
                "video": vf,
                "folder": folder_name,
                "ground_truth": ground_truth_label,
                "ground_truth_id": ground_truth_id,
                "predicted": final_label,
                "predicted_id": final_id,
                "correct": is_correct,
                "trigger": trigger,
                "audio": {
                    "transcript": audio_transcript,
                    "label": audio_result["label"],
                    "confidence": audio_result["confidence"],
                    "scores": audio_result.get("avg_scores", {})
                },
                "ocr": {
                    "transcript": ocr_transcript,
                    "label": ocr_result["label"],
                    "confidence": ocr_result["confidence"],
                    "scores": ocr_result.get("avg_scores", {})
                }
            })

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    if total > 0:
        print(f"  Overall Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"\n  Per-class Accuracy:")
        for label in LABEL_NAMES:
            ct = class_total.get(label, 0)
            cc = class_correct.get(label, 0)
            if ct > 0:
                print(f"    {label:>10s}: {cc}/{ct} = {cc/ct*100:.1f}%")
            else:
                print(f"    {label:>10s}: No samples")
    else:
        print("  No videos found!")

    # ── Save results ──────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": round(correct / max(total, 1) * 100, 2),
                "per_class": {
                    label: {
                        "total": class_total.get(label, 0),
                        "correct": class_correct.get(label, 0),
                        "accuracy": round(class_correct.get(label, 0) / max(class_total.get(label, 0), 1) * 100, 2),
                    }
                    for label in LABEL_NAMES
                },
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  📊 Detailed results saved → {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
