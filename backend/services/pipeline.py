import os
import gc
import uuid

from models.audio.inference import predict_audio
from models.vision.inference import predict_vision
from models.text.inference import predict_text, extract_ocr_text

from utils.media import extract_audio_from_video, get_video_duration
from utils.transcription import transcribe_audio

from moviepy.video.io.VideoFileClip import VideoFileClip

TEMP_DIR = "temp"


# ==============================
# CLAMP SEGMENTS
# ==============================

def clamp_segments(segments, video_duration):
    fixed = []

    for seg in segments:
        start = max(0, float(seg["start"]))
        end = min(float(seg["end"]), video_duration)

        if end <= start:
            continue

        fixed.append({
            "start": start,
            "end": end,
            "text": seg.get("text", "")
        })

    return fixed


# ==============================
# FALLBACK SEGMENTS
# ==============================

def create_fallback_segments(video_duration, chunk_size=7.0):
    segments = []
    start = 0.0

    while start < video_duration:
        end = min(start + chunk_size, video_duration)

        segments.append({
            "start": start,
            "end": end,
            "text": ""
        })

        start = end

    return segments


# ==============================
# TEXT AGGREGATION
# ==============================

def aggregate_text_segments(segment_results, full_transcript=""):

    if not segment_results:
        return {
            "neutral": 1.0,
            "sexual_content": 0.0,
            "violence": 0.0,
            "hate_speech": 0.0
        }

    n = len(segment_results)

    avg = {k: 0.0 for k in ["neutral", "sexual_content", "violence", "hate_speech"]}

    for seg in segment_results:
        t = seg["modalities"]["text"]
        for k in avg:
            avg[k] += t.get(k, 0.0) / n

    # Density
    strong_counts = {k: 0 for k in avg}

    for seg in segment_results:
        t = seg["modalities"]["text"]
        label = max(t, key=t.get)

        if t[label] > 0.6:
            strong_counts[label] += 1

    for k in avg:
        avg[k] += 0.2 * (strong_counts[k] / n)

    text = full_transcript.lower()

    if any(w in text for w in ["kill", "murder", "shoot"]):
        avg["violence"] += 0.3

    if any(w in text for w in ["sex", "nude", "porn"]):
        avg["sexual_content"] += 0.3

    if any(w in text for w in ["hate", "racist"]):
        avg["hate_speech"] += 0.3

    if any(w in text for w in ["game", "player", "level", "mission", "weapon"]):
        avg["violence"] *= 0.6

    if any(w in text for w in ["lyrics", "song", "music"]):
        avg["sexual_content"] *= 0.7
        avg["violence"] *= 0.7

    total = sum(avg.values())
    if total > 0:
        for k in avg:
            avg[k] /= total

    return avg


# ==============================
# MAIN PIPELINE
# ==============================

# ADD THIS IMPORT AT TOP
from models.text.inference import predict_text, extract_ocr_text


def process_video(video_path: str):

    base_name = str(uuid.uuid4())
    audio_path = os.path.join(TEMP_DIR, f"{base_name}.wav")

    try:
        video_duration = get_video_duration(video_path)

        # ======================
        # 🔥 GLOBAL OCR (ONCE)
        # ======================
        print("🔍 Running OCR on full video...")
        full_ocr_text = extract_ocr_text(video_path)
        print("✅ OCR Done")

        # ======================
        # AUDIO HANDLING
        # ======================
        audio_file = extract_audio_from_video(video_path, audio_path)

        if audio_file is None:
            print("⚠️ No audio → fallback segmentation")
            segments = create_fallback_segments(video_duration)
            has_audio = False
        else:
            segments = transcribe_audio(audio_file)
            has_audio = True

        if not segments:
            segments = create_fallback_segments(video_duration)

        segments = clamp_segments(segments, video_duration)

        segment_results = []
        full_transcript = []

        video = VideoFileClip(video_path)

        for seg in segments:
            start, end, text = seg["start"], seg["end"], seg["text"]

            full_transcript.append(text)

            seg_id = str(uuid.uuid4())
            seg_audio_path = os.path.join(TEMP_DIR, f"{seg_id}.wav")
            seg_video_path = os.path.join(TEMP_DIR, f"{seg_id}.mp4")

            try:
                if end <= start:
                    continue

                subclip = video.subclip(start, end)

                subclip.write_videofile(
                    seg_video_path,
                    codec="libx264",
                    audio_codec="aac",
                    verbose=False,
                    logger=None
                )

                # AUDIO
                if has_audio and subclip.audio is not None:
                    subclip.audio.write_audiofile(
                        seg_audio_path,
                        codec="pcm_s16le",
                        logger=None
                    )
                    audio_scores = predict_audio(seg_audio_path)
                else:
                    audio_scores = {
                        "neutral": 1.0,
                        "sexual_content": 0.0,
                        "violence": 0.0,
                        "hate_speech": 0.0
                    }

                # 🔥 TEXT (NOW USING GLOBAL OCR)
                text_scores = predict_text(text, ocr_text=full_ocr_text)

                vision_scores = predict_vision(seg_video_path)

                segment_results.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "modalities": {
                        "text": text_scores,
                        "audio": audio_scores,
                        "vision": vision_scores
                    }
                })

            except Exception as e:
                print(f"❌ Segment error ({start}-{end}): {e}")

            finally:
                if os.path.exists(seg_audio_path):
                    os.remove(seg_audio_path)
                if os.path.exists(seg_video_path):
                    os.remove(seg_video_path)

        video.close()
        gc.collect()

        text_modality = aggregate_text_segments(
            segment_results,
            full_transcript=" ".join(full_transcript)
        )

        return {
            "verdict": "frontend_computed",
            "confidence": 0.0,
            "segments": segment_results,
            "transcript": " ".join(full_transcript).strip(),
            "modalities": {
                "text": text_modality
            }
        }

    except Exception as e:
        print(f"❌ Pipeline error: {e}")

        return {
            "verdict": "error",
            "confidence": 0.0,
            "segments": [],
            "transcript": "",
            "modalities": {},
            "error": str(e)
        }