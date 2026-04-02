import os
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_audio_from_video(video_path: str, audio_path: str):
    try:
        with VideoFileClip(video_path) as video:
            if video.audio is None:
                return None

            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            # Write audio
            video.audio.write_audiofile(
                audio_path,
                codec="pcm_s16le",  # standard WAV codec
                logger=None        # suppress verbose logs
            )

            return audio_path

    except Exception as e:
        print(f"❌ Audio extraction error: {e}")
        return None

def get_video_duration(video_path):
    with VideoFileClip(video_path) as clip:
        return clip.duration