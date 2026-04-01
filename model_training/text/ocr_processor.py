import os
import argparse
import json
from difflib import SequenceMatcher

def run_ocr_on_frames(frame_data, lang_list=['en'], min_confidence=0.40):
    """
    Args:
        frame_data: List of (path, timestamp) tuples
    Returns:
        List of (text, timestamp) tuples
    """
    try:
        import easyocr
    except ImportError:
        print("  [ERROR] EasyOCR not installed. Run: pip install easyocr")
        return []

    print(f"  [OCR] Initializing EasyOCR on CPU for: {lang_list}...")
    reader = easyocr.Reader(lang_list, gpu=False) 
    
    all_text_results = []
    last_seen_text = ""

    print(f"  [OCR] Processing {len(frame_data)} timestamped frames...")
    
    for i, (frame_path, timestamp) in enumerate(frame_data):
        results = reader.readtext(frame_path)
        
        frame_text_parts = []
        for (bbox, text, prob) in results:
            if prob >= min_confidence:
                frame_text_parts.append(text.strip())
        
        current_frame_text = " ".join(frame_text_parts).strip()
        
        if not current_frame_text:
            continue
            
        # Deduplication: Check similarity with last seen unique text
        if last_seen_text:
            similarity = SequenceMatcher(None, current_frame_text.lower(), last_seen_text.lower()).ratio()
            if similarity > 0.80:
                continue
        
        # Store text and the time it appeared
        all_text_results.append((current_frame_text, timestamp))
        last_seen_text = current_frame_text
        print(f"    [{timestamp:5.1f}s] OCR Found: \"{current_frame_text[:60]}...\"")

    return all_text_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on a directory of images.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", default="ocr_results.json")
    
    args = parser.parse_args()
    
    frames = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.endswith(".jpg")])
    if frames:
        results = run_ocr_on_frames(frames)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  [OK] OCR analysis complete. Unique text segments: {len(results)}")
