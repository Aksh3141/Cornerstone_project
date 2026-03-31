import os
import requests
import pandas as pd
from tqdm import tqdm
import time

# ==============================
# CONFIG
# ==============================

BACKEND_URL = "http://127.0.0.1:8000/api/moderate-video"

REQUEST_TIMEOUT = 180  # seconds
DELAY_BETWEEN_REQUESTS = 2


# ==============================
# GET VIDEOS FROM ONE FOLDER
# ==============================

def get_videos_from_folder(folder_path, true_class):
    data = []

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return data

    for file in os.listdir(folder_path):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            data.append({
                "path": os.path.join(folder_path, file),
                "true_class": true_class,
                "name": file
            })

    return data


# ==============================
# ROUND FUNCTION
# ==============================

def r(x):
    return round(float(x), 2) if x is not None else 0.0


# ==============================
# MAIN EVALUATION
# ==============================

def evaluate_single_class(folder_path, true_class):
    videos = get_videos_from_folder(folder_path, true_class)

    print(f"\n🎥 Found {len(videos)} videos in '{true_class}'\n")

    rows = []

    output_file = f"evaluation_{true_class}.xlsx"
    failed_log = f"failed_{true_class}.txt"

    for i, vid in enumerate(tqdm(videos)):
        print(f"\n➡️ [{i+1}/{len(videos)}] Processing: {vid['name']}")

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

                # FINAL (rounded)
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

            # SAVE AFTER EACH VIDEO
            df = pd.DataFrame(rows)
            df.to_excel(output_file, index=False)

            time.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"❌ Failed: {vid['name']} → {e}")

            with open(failed_log, "a") as f:
                f.write(f"{vid['path']}\n")

            time.sleep(3)
            continue

    print(f"\n✅ Done: {true_class}")
    print(f"📄 Saved to: {output_file}")
    print(f"⚠️ Failed log: {failed_log}")


# ==============================
# RUN (MANUAL CONTROL)
# ==============================

if __name__ == "__main__":

    # 🔥 CHANGE THESE EACH TIME
    FOLDER_PATH = os.path.expanduser("~/Downloads/dataset/hate_speech")
    TRUE_CLASS = "hate"

    evaluate_single_class(FOLDER_PATH, TRUE_CLASS)