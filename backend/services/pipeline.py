import os
import gc
import uuid
import torch
import time
from models.audio.inference import predict_audio
from models.vision.inference import predict_vision
from models.text.inference import predict_text, extract_ocr_text
from utils.media import extract_audio_from_video, get_video_duration
from utils.transcription import transcribe_audio
from moviepy.video.io.VideoFileClip import VideoFileClip

TEMP_DIR = "temp"
USE_OCR = False

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


def average_modalities(segment_results, modality):
    if not segment_results:
        return {
            "neutral": 1.0,
            "sexual_content": 0.0,
            "violence": 0.0,
            "hate_speech": 0.0
        }

    avg = {k: 0.0 for k in ["neutral", "sexual_content", "violence", "hate_speech"]}
    count = 0

    for seg in segment_results:
        m = seg["modalities"].get(modality)
        if not m:
            continue

        count += 1
        for k in avg:
            avg[k] += m.get(k, 0.0)

    if count == 0:
        return avg

    for k in avg:
        avg[k] /= count

    return avg


def combine_modalities(modalities):

    text = modalities.get("text", {})
    audio = modalities.get("audio", {})
    vision = modalities.get("vision", {})

    classes = ["neutral", "sexual_content", "violence", "hate_speech"]

    def avg(cls):
        return (
            text.get(cls, 0.0) +
            audio.get(cls, 0.0) +
            vision.get(cls, 0.0)
        ) / 3.0

    def hate_dominating():
        final_scores = {}

        text_hate = text.get("hate_speech", 0.0)
        final_hate = text_hate * 1.5
        final_hate = min(final_hate, 1.0)  
        final_scores["hate_speech"] = final_hate
        remaining = 1.0 - final_hate

        temp = {
            "neutral": avg("neutral"),
            "sexual_content": avg("sexual_content"),
            "violence": avg("violence")
        }

        total = sum(temp.values())

        if total > 0:
            for k in temp:
                final_scores[k] = remaining * (temp[k] / total)
        else:
            for k in temp:
                final_scores[k] = 0.0

        return final_scores

    def sexual_dominating():
        final_scores = {}

        vision_sex = vision.get("sexual_content", 0.0)
        final_sex = vision_sex * 1.2
        final_sex = min(final_sex, 1.0)

        final_scores["sexual_content"] = final_sex
        remaining = 1.0 - final_sex

        temp = {
            "neutral": avg("neutral"),
            "violence": avg("violence"),
            "hate_speech": avg("hate_speech")
        }

        total = sum(temp.values())
        if total > 0:
            for k in temp:
                final_scores[k] = remaining * (temp[k] / total)
        else:
            for k in temp:
                final_scores[k] = 0.0

        return final_scores

    if text.get("hate_speech", 0.0) > 0.3:
        return hate_dominating()

    if vision.get("sexual_content", 0.0) > 0.5:
        return sexual_dominating()

    final_scores = {}

    final_scores["hate_speech"] = avg("hate_speech") * 0.02
    final_scores["sexual_content"] = avg("sexual_content") * 0.02

    remaining = 1.0 - (final_scores["hate_speech"] + final_scores["sexual_content"])

    temp = {
        "neutral": avg("neutral"),
        "violence": avg("violence")
    }

    total = sum(temp.values())

    if total > 0:
        final_scores["neutral"] = remaining * (temp["neutral"] / total)
        final_scores["violence"] = remaining * (temp["violence"] / total)
    else:
        final_scores["neutral"] = 0.0
        final_scores["violence"] = 0.0

    return final_scores

def adjust_text_probs(text_probs):
    """
    Fix confusion between hate_speech and violence in text modality
    """

    hate = text_probs.get("hate_speech", 0.0)
    violence = text_probs.get("violence", 0.0)

    if hate > 0.1:
        if violence > 0.01:
            excess = violence - 0.01
            text_probs["violence"] = 0.01
            text_probs["hate_speech"] += excess

    else:
        if hate > 0.01:
            excess = hate - 0.01
            text_probs["hate_speech"] = 0.01
            text_probs["violence"] += excess

    total = sum(text_probs.values())
    if total > 0:
        for k in text_probs:
            text_probs[k] /= total

    return text_probs

def process_video(video_path: str):

    base_name = str(uuid.uuid4())
    audio_path = os.path.join(TEMP_DIR, f"{base_name}.wav")

    try:
        video_duration = get_video_duration(video_path)

        # OCR
        if USE_OCR:
            print("🔍 Running OCR on full video...")
            full_ocr_text = extract_ocr_text(video_path)
        else:
            full_ocr_text = ""

        # AUDIO
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

            if end <= start:
                continue

            full_transcript.append(text)

            seg_id = str(uuid.uuid4())
            seg_audio_path = os.path.join(TEMP_DIR, f"{seg_id}.wav")
            seg_video_path = os.path.join(TEMP_DIR, f"{seg_id}.mp4")

            try:
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

                # TEXT
                text_scores = predict_text(text, ocr_text=full_ocr_text)

                # VISION
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

        if os.path.exists(audio_path):
            os.remove(audio_path)

        gc.collect()

        text_modality = aggregate_text_segments(
            segment_results,
            full_transcript=" ".join(full_transcript)
        )
        text_modality = adjust_text_probs(text_modality)

        def avg_modality(name):
            avg = {k: 0.0 for k in ["neutral","sexual_content","violence","hate_speech"]}
            count = 0

            for seg in segment_results:
                m = seg["modalities"].get(name)
                if not m:
                    continue
                count += 1
                for k in avg:
                    avg[k] += m.get(k, 0.0)

            if count > 0:
                for k in avg:
                    avg[k] /= count

            return avg

        audio_modality = avg_modality("audio")
        vision_modality = avg_modality("vision")

        modalities = {
            "text": text_modality,
            "audio": audio_modality,
            "vision": vision_modality
        }

        final_scores = combine_modalities(modalities)

        final_label = max(final_scores, key=final_scores.get)
        confidence = final_scores[final_label]

        return {
            "verdict": final_label,
            "confidence": confidence,
            "final_scores": final_scores,
            "segments": segment_results,
            "transcript": " ".join(full_transcript).strip(),
            "modalities": modalities
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