#!/usr/bin/env python3
"""
inference.py  –  Classify video transcripts at sentence + paragraph level
=========================================================================
Your teammate extracts text from video.  This script takes that text and:
  1. Splits it into sentences
  2. Classifies each sentence (Neutral / Offensive / Hate / Sexual)
  3. Aggregates into an overall VIDEO-level score

Output modes:
  - JSON  (for API / pipeline integration)
  - Pretty-print  (for human reading)

Classes:  0=Neutral  1=Offensive  2=Hate  3=Sexual

Usage:
  # Single paragraph (video transcript)
  python inference.py --model_dir ./results/best_model \
      --text "First sentence. Second sentence. Third sentence."

  # From a file (one transcript per line, or a single multi-line transcript)
  python inference.py --model_dir ./results/best_model --file transcript.txt

  # Interactive mode
  python inference.py --model_dir ./results/best_model --interactive

  # JSON output (for integration with teammate's pipeline)
  python inference.py --model_dir ./results/best_model --text "..." --json
"""

import argparse
import html
import json
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = ["Neutral", "Offensive", "Hate", "Sexual"]
NUM_LABELS = 4

# Severity ranking for "worst class" aggregation
# Higher = more severe.  Used to pick the dominant harmful class.
SEVERITY = {0: 0, 1: 1, 2: 3, 3: 2}  # Hate > Sexual > Offensive > Neutral


# ══════════════════════════════════════════════════════════════
# Text cleaning
# ══════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Clean input text before classification.
    
    Removes URLs, @mentions, HTML entities, excessive whitespace,
    and Whisper bracket artifacts.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[\[\(\*](music|laughter|applause|silence|inaudible|cheering)[\]\)\*]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════
# Sentence splitter
# ══════════════════════════════════════════════════════════════
def split_sentences(text: str) -> list:
    """
    Split paragraph into sentences.
    Handles:  .  !  ?  ...  and newline boundaries.
    """
    # Split on sentence-ending punctuation followed by space/newline or end
    raw = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = [s.strip() for s in raw if s.strip() and len(s.strip()) > 3]
    # If no split happened (single sentence), return as-is
    if not sentences:
        sentences = [text.strip()]
    return sentences


# ══════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════
def load_model(model_dir: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()
    return tokenizer, model, device


# ══════════════════════════════════════════════════════════════
# Sentence-level classification
# ══════════════════════════════════════════════════════════════
def classify_sentences(sentences, tokenizer, model, device, max_length=128, batch_size=16):
    """Classify a list of sentences. Returns list of per-sentence results."""
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


# ══════════════════════════════════════════════════════════════
# Context & Topic Detection
# ══════════════════════════════════════════════════════════════
def detect_safe_topic(text: str) -> bool:
    """Detect if the text is likely about a generally safe but 'aggressive' topic 
    like Gaming, Sports, Action Movies, or Medical Documentaries."""
    low_text = text.lower()
    
    # Gaming / Sports / Racing keywords
    gaming_sports = ["game", "play", "hp", "level", "player", "match", "cannon", "field", "wicket", "ball", "wheel", "race", "speed", "score", "damage"]
    # Food / Health / Documentary keywords
    doc_food = ["health", "recipe", "spicy", "taste", "food", "painful", "doctor", "history", "mystery", "death", "mysteriously", "medical"]
    
    # Count matches
    sports_matches = sum(1 for w in gaming_sports if w in low_text)
    doc_matches = sum(1 for w in doc_food if w in low_text)
    
    # Require at least 2 distinct keywords to confirm topic, 
    # to avoid false triggering on random sentences.
    return sports_matches >= 2 or doc_matches >= 2


# ══════════════════════════════════════════════════════════════
# Video-level (paragraph) aggregation
# ══════════════════════════════════════════════════════════════
def aggregate_video_score(text: str, sentence_results: list) -> dict:
    """
    Aggregate sentence predictions into a single video-level score using
    Topic Shields and Sentence Density voting.
    """
    if not sentence_results:
        return {"label": "Neutral", "label_id": 0, "confidence": 1.0,
                "avg_scores": {n: 0.0 for n in LABEL_NAMES},
                "flag_counts": {n: 0 for n in LABEL_NAMES},
                "total_sentences": 0, "worst_sentence": None}

    n = len(sentence_results)

    # Average probabilities
    avg_scores = {name: 0.0 for name in LABEL_NAMES}
    for r in sentence_results:
        for name, score in r["scores"].items():
            avg_scores[name] += score / n

    # Flag counts (sentences where that class is the prediction)
    flag_counts = {name: 0 for name in LABEL_NAMES}
    for r in sentence_results:
        flag_counts[r["label"]] += 1

    # Find worst sentence (highest severity)
    worst = max(sentence_results, key=lambda r: (SEVERITY[r["label_id"]], r["confidence"]))

    # Base inference is Neutral unless proven otherwise
    video_label = "Neutral"
    video_label_id = 0

    # Minimum confidence to consider a sentence "flagged"
    CONFIDENCE_THRESHOLD = 0.50

    strong_flags = {name: 0 for name in LABEL_NAMES}
    for r in sentence_results:
        if r["confidence"] > CONFIDENCE_THRESHOLD:
            strong_flags[r["label"]] += 1

    # --- Density Voting ---
    hate_density = strong_flags["Hate"] / n
    offensive_density = (strong_flags["Offensive"] + strong_flags["Hate"]) / n # Hate also counts towards offensive density
    
    # --- Topic Shield ---
    is_safe_topic = detect_safe_topic(text)
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
            
    # Allow pure average to override if it natively predicted Hate/Offensive over everything else
    native_max = max(range(NUM_LABELS), key=lambda i: avg_scores[LABEL_NAMES[i]])
    if SEVERITY[native_max] > SEVERITY[video_label_id]:
        video_label_id = native_max
        video_label = LABEL_NAMES[native_max]

    return {
        "label": video_label,
        "label_id": video_label_id,
        "confidence": round(avg_scores[video_label], 4),
        "avg_scores": {k: round(v, 4) for k, v in avg_scores.items()},
        "flag_counts": flag_counts,
        "total_sentences": n,
        "worst_sentence": {
            "text": worst["sentence"],
            "label": worst["label"],
            "confidence": worst["confidence"],
        },
    }


# ══════════════════════════════════════════════════════════════
# Full pipeline: text → sentence split → classify → aggregate
# ══════════════════════════════════════════════════════════════
def classify_video_transcript(text, tokenizer, model, device):
    """
    Main entry point.  Takes a full transcript (paragraph),
    returns sentence-level + video-level results.
    """
    text = clean_text(text)
    sentences = split_sentences(text)
    sentence_results = classify_sentences(sentences, tokenizer, model, device)
    video_result = aggregate_video_score(text, sentence_results)
    return {
        "video_level": video_result,
        "sentence_level": sentence_results,
    }


# ══════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════
def pretty_print(result):
    v = result["video_level"]
    print(f"\n{'═'*60}")
    print(f"  VIDEO-LEVEL CLASSIFICATION")
    print(f"{'═'*60}")
    print(f"  Label:       {v['label']}")
    print(f"  Confidence:  {v['confidence']:.2%}")
    print(f"  Sentences:   {v['total_sentences']}")
    print(f"  Avg Scores:  {' | '.join(f'{k}: {v2:.2%}' for k,v2 in v['avg_scores'].items())}")
    print(f"  Flags:       {' | '.join(f'{k}: {v2}' for k,v2 in v['flag_counts'].items())}")
    if v["worst_sentence"]:
        ws = v["worst_sentence"]
        print(f"  Worst:       \"{ws['text'][:60]}...\" → {ws['label']} ({ws['confidence']:.2%})")

    print(f"\n{'─'*60}")
    print(f"  SENTENCE-LEVEL BREAKDOWN")
    print(f"{'─'*60}")
    for i, s in enumerate(result["sentence_level"], 1):
        marker = "⚠️" if s["label_id"] > 0 else "✅"
        print(f"  {marker} [{i}] {s['label']:>10s} ({s['confidence']:.0%})  \"{s['sentence'][:70]}\"")
    print()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Classify video transcripts (sentence + video level)")
    ap.add_argument("--model_dir", required=True, help="Path to saved model")
    ap.add_argument("--text", default=None, help="Transcript text (paragraph)")
    ap.add_argument("--file", default=None, help="File containing transcript")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model_dir, args.device)
    print(f"Model loaded on {device}")

    if args.text:
        result = classify_video_transcript(args.text, tokenizer, model, device)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            pretty_print(result)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        result = classify_video_transcript(text, tokenizer, model, device)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            pretty_print(result)

    elif args.interactive:
        print("\nInteractive mode. Paste transcript, press Enter twice to classify. Type 'quit' to exit.\n")
        while True:
            lines = []
            print("Paste transcript (Enter twice to submit):")
            while True:
                line = input()
                if line == "":
                    if lines:
                        break
                    continue
                if line.lower() in ("quit", "exit", "q"):
                    sys.exit(0)
                lines.append(line)
            text = " ".join(lines)
            result = classify_video_transcript(text, tokenizer, model, device)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                pretty_print(result)

    else:
        # Demo
        demo = (
            "The weather is beautiful today and everyone is having a great time. "
            "You are such a disgusting worthless piece of trash. "
            "I hate all people from that country they should be deported. "
            "She was wearing something really provocative and explicit. "
            "The kids were playing in the park and laughing together."
        )
        print(f"\nDemo transcript:\n  \"{demo[:80]}...\"\n")
        result = classify_video_transcript(demo, tokenizer, model, device)
        pretty_print(result)


if __name__ == "__main__":
    main()
