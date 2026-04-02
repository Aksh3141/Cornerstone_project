import os
import argparse
import json
from difflib import SequenceMatcher

print("🔍 Loading EasyOCR model...")

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    print("✅ EasyOCR loaded")
except Exception as e:
    print(f"❌ EasyOCR load error: {e}")
    reader = None

def run_ocr_on_frames(frame_data, min_confidence=0.40):
    """
    Args:
        frame_data: List of (path, timestamp)
    Returns:
        List of (text, timestamp)
    """

    if reader is None:
        return []

    all_text_results = []
    last_seen_text = ""

    print(f"  [OCR] Processing {len(frame_data)} frames...")

    for frame_path, timestamp in frame_data:
        try:
            results = reader.readtext(frame_path)
        except Exception as e:
            print(f"  [OCR ERROR] {e}")
            continue

        frame_text_parts = []

        for (_, text, prob) in results:
            if prob >= min_confidence:
                frame_text_parts.append(text.strip())

        current_frame_text = " ".join(frame_text_parts).strip()

        if not current_frame_text:
            continue

        # Deduplication
        if last_seen_text:
            similarity = SequenceMatcher(
                None,
                current_frame_text.lower(),
                last_seen_text.lower()
            ).ratio()

            if similarity > 0.80:
                continue

        all_text_results.append((current_frame_text, timestamp))
        last_seen_text = current_frame_text

        print(f"    [{timestamp:5.1f}s] OCR: \"{current_frame_text[:60]}...\"")

    return all_text_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on images.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", default="ocr_results.json")

    args = parser.parse_args()

    frames = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.endswith(".jpg")
    ])

    if frames:
        frame_data = [(f, i) for i, f in enumerate(frames)]

        results = run_ocr_on_frames(frame_data)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"  [OK] OCR complete. Segments: {len(results)}")