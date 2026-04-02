import os
import requests
import pandas as pd
from tqdm import tqdm
import time

BACKEND_URL = "http://127.0.0.1:8000/api/moderate-video"

DATASET_PATH = os.path.expanduser("~/Downloads/dataset")

CLASS_FOLDERS = [
    "violence",
    "hate_speech",
    "sexual_content",
    "neutral"
]

OUTPUT_FILE = "evaluation_results.xlsx"
FAILED_LOG = "failed_videos.txt"

REQUEST_TIMEOUT = 180
DELAY_BETWEEN_REQUESTS = 2

def r(x):
    return round(float(x), 2) if x is not None else 0.0

def get_all_videos():
    data = []

    for cls in CLASS_FOLDERS:
        folder = os.path.join(DATASET_PATH, cls)

        if not os.path.exists(folder):
            print(f"⚠️ Missing folder: {folder}")
            continue

        for file in os.listdir(folder):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                data.append({
                    "path": os.path.join(folder, file),
                    "true_class": cls,
                    "name": file
                })
    return data

def evaluate():
    videos = get_all_videos()
    print(f"🎥 Found {len(videos)} videos\n")

    rows = []

    for i, vid in enumerate(tqdm(videos)):
        print(f"\n➡️ [{i+1}/{len(videos)}] Processing: {vid['name']} ({vid['true_class']})")

        try:
            with open(vid["path"], "rb") as f:

                try:
                    response = requests.post(
                        BACKEND_URL,
                        files={"file": f},
                        timeout=(10, REQUEST_TIMEOUT)
                    )
                except requests.exceptions.Timeout:
                    raise Exception("Request timed out")

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")

            data = response.json()

            final_scores = data.get("final_scores", {})
            modalities = data.get("modalities", {})

            text = modalities.get("text", {})
            audio = modalities.get("audio", {})
            vision = modalities.get("vision", {})

            predicted = max(final_scores, key=final_scores.get) if final_scores else "unknown"

            row = {
                "video_name": vid["name"],
                "true_class": vid["true_class"],
                "predicted_class": predicted,

                # FINAL
                "final_neutral": r(final_scores.get("neutral", 0)),
                "final_sexual": r(final_scores.get("sexual_content", 0)),
                "final_violence": r(final_scores.get("violence", 0)),
                "final_hate": r(final_scores.get("hate_speech", 0)),

                # TEXT
                "text_neutral": r(text.get("neutral", 0)),
                "text_sexual": r(text.get("sexual_content", 0)),
                "text_violence": r(text.get("violence", 0)),
                "text_hate": r(text.get("hate_speech", 0)),

                # AUDIO
                "audio_neutral": r(audio.get("neutral", 0)),
                "audio_sexual": r(audio.get("sexual_content", 0)),
                "audio_violence": r(audio.get("violence", 0)),
                "audio_hate": r(audio.get("hate_speech", 0)),

                # VISION
                "vision_neutral": r(vision.get("neutral", 0)),
                "vision_sexual": r(vision.get("sexual_content", 0)),
                "vision_violence": r(vision.get("violence", 0)),
                "vision_hate": r(vision.get("hate_speech", 0)),
            }

            rows.append(row)

            # SAVE PROGRESS
            df = pd.DataFrame(rows)
            df.to_excel(OUTPUT_FILE, index=False)

            time.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"❌ Failed: {vid['name']} → {e}")

            with open(FAILED_LOG, "a") as f:
                f.write(f"{vid['path']}\n")

            time.sleep(3)
            continue

    print(f"\n✅ Completed. Results saved to {OUTPUT_FILE}")
    print(f"⚠️ Failed videos logged in {FAILED_LOG}")

if __name__ == "__main__":
    evaluate()