import os
import torch
import math
from models.vision.model import Model, Config, preprocess

CLASSES = ["neutral", "sexual_content", "violence", "hate_speech"]

BASE_DIR = os.path.dirname(__file__)

VIOLENCE_MODEL_PATH = os.path.join(BASE_DIR, "violence_best.pth")
NUDITY_MODEL_PATH = os.path.join(BASE_DIR, "nudity_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🎥 Loading Vision Models...")

CFG = Config()

try:
    # Violence model
    violence_model = Model(CFG).to(DEVICE)
    violence_model.load_state_dict(
        torch.load(VIOLENCE_MODEL_PATH, map_location=DEVICE)
    )
    violence_model.eval()

    # Nudity model
    nudity_model = Model(CFG).to(DEVICE)
    nudity_model.load_state_dict(
        torch.load(NUDITY_MODEL_PATH, map_location=DEVICE)
    )
    nudity_model.eval()

    print("✅ Vision models ready")

except Exception as e:
    print(f"❌ Vision model load error: {e}")
    violence_model = None
    nudity_model = None

def softmax(values):
    exps = [math.exp(v) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def combine_probs(p_n, p_v):
    if p_n > 0.9999:
        sexual = 0.97
        high_sexual = True
    else:
        sexual = 0.02
        high_sexual = False

    if p_v > 0.10:
        violence = 0.85
    elif p_v >= 0.05:
        violence = 0.4
    else:
        violence = 0.02

    if high_sexual:
        neutral = max(0.0, 1.0 - (sexual + violence))

        vals = softmax([sexual, violence, neutral])
        sexual, violence, neutral = vals

    else:
        # remaining space
        remaining = 1.0 - sexual

        # temporary neutral
        neutral = max(0.0, 1.0 - (sexual + violence))

        # normalize only v & n using softmax
        v_n = softmax([violence, neutral])

        violence = remaining * v_n[0]
        neutral = remaining * v_n[1]

    return {
        "neutral": float(neutral),
        "sexual_content": float(sexual),
        "violence": float(violence),
        "hate_speech": 0.0
    }

def predict_vision(video_path: str):
    """
    Input: video clip path
    Output: probability distribution
    """

    if violence_model is None or nudity_model is None:
        return {c: 0.0 for c in CLASSES}

    try:
        # Preprocess video -> tensor
        video = preprocess(video_path).to(DEVICE)

        with torch.no_grad():
            v_logit = violence_model(video)
            n_logit = nudity_model(video)

            p_v = torch.sigmoid(v_logit).item()
            p_n = torch.sigmoid(n_logit).item()

        return combine_probs(p_n, p_v)

    except Exception as e:
        print(f"❌ Vision inference error: {e}")
        return {c: 0.0 for c in CLASSES}