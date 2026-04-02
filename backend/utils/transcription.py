import whisper
import subprocess
import json

print("🧠 Loading Whisper model...")

try:
    model = whisper.load_model("base")
    print("✅ Whisper loaded")
except Exception as e:
    print(f"❌ Whisper load error: {e}")
    model = None

def transcribe_audio(audio_path: str):

    if model is None:
        return []

    try:
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language=None
        )

        raw_segments = result.get("segments", [])

        total_duration = get_audio_duration(audio_path)

        if not raw_segments:
            print("⚠️ Whisper returned no segments → fallback chunking")
            return create_fixed_segments(total_duration)

        segments = [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip()
            }
            for seg in raw_segments
        ]

        segments = merge_short_segments(segments)
        segments = fill_gaps(segments, total_duration)
        segments = clamp_segments(segments, total_duration)

        return segments

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return []


def merge_short_segments(segments, min_duration=5.0):

    if not segments:
        return []

    merged = []
    i = 0

    while i < len(segments):
        current = segments[i]

        start = current["start"]
        end = current["end"]
        text = current["text"]

        duration = end - start

        while duration < min_duration and i + 1 < len(segments):
            i += 1
            next_seg = segments[i]

            end = next_seg["end"]
            text = (text + " " + next_seg["text"]).strip()
            duration = end - start

        merged.append({
            "start": start,
            "end": end,
            "text": text
        })

        i += 1

    return merged


def fill_gaps(segments, total_duration):

    if not segments:
        return []

    filled = []
    prev_end = 0.0

    for seg in segments:
        start = seg["start"]
        end = seg["end"]

        if start > prev_end:
            filled.append({
                "start": prev_end,
                "end": start,
                "text": ""
            })

        filled.append(seg)
        prev_end = end

    if prev_end < total_duration:
        filled.append({
            "start": prev_end,
            "end": total_duration,
            "text": ""
        })

    return filled


def create_fixed_segments(total_duration, chunk_size=7.0):

    if total_duration <= 0:
        total_duration = 30.0  # assume 30s if unknown

    segments = []
    start = 0.0

    while start < total_duration:
        end = min(start + chunk_size, total_duration)

        segments.append({
            "start": start,
            "end": end,
            "text": ""
        })

        start = end

    return segments

def clamp_segments(segments, total_duration):

    fixed = []

    for seg in segments:
        start = max(0.0, seg["start"])
        end = min(seg["end"], total_duration)

        if end <= start:
            continue

        fixed.append({
            "start": start,
            "end": end,
            "text": seg["text"]
        })

    return fixed

def get_audio_duration(audio_path):

    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            audio_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        data = json.loads(result.stdout)

        duration = float(data["format"]["duration"])

        if duration <= 0:
            raise Exception("Invalid duration")

        return duration

    except Exception as e:
        print(f"⚠️ ffprobe failed: {e}")

        try:
            result = model.transcribe(audio_path)
            return result.get("duration", 30.0)
        except:
            return 30.0