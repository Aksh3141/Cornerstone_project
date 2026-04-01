import os
import torch

from models.vision.model import Model, Config, preprocess

# ==============================
# CONFIG
# ==============================

CLASSES = ["neutral", "sexual_content", "violence", "hate_speech"]

BASE_DIR = os.path.dirname(__file__)

VIOLENCE_MODEL_PATH = os.path.join(BASE_DIR, "violence_best.pth")
NUDITY_MODEL_PATH = os.path.join(BASE_DIR, "nudity_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODELS (ONCE)
# ==============================

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


# ==============================
# PROBABILITY COMBINATION
# ==============================
def combine_probs(p_n, p_v):
    """
    STRICT threshold-based logic (no interpolation)
    """

    # =========================
    # SEXUAL (HARD GATE)
    # =========================
    if p_n > 0.999:
        sexual = 0.97
    else:
        sexual = 0.02

    # =========================
    # VIOLENCE (BUCKETED)
    # =========================
    if p_v > 0.10:
        violence = 0.85
    elif p_v >= 0.05:
        violence = 0.4
    else:
        violence = 0.02

    # =========================
    # NEUTRAL
    # =========================
    neutral = 1.0 - (sexual + violence)

    if neutral < 0:
        neutral = 0.0

    # =========================
    # NORMALIZE
    # =========================
    total = sexual + violence + neutral

    if total > 0:
        sexual /= total
        violence /= total
        neutral /= total

    return {
        "neutral": float(neutral),
        "sexual_content": float(sexual),
        "violence": float(violence),
        "hate_speech": 0.0
    }

# ==============================
# MAIN INFERENCE FUNCTION
# ==============================

def predict_vision(video_path: str):
    """
    Input: video clip path
    Output: probability distribution
    """

    if violence_model is None or nudity_model is None:
        return {c: 0.0 for c in CLASSES}

    try:
        # Preprocess video → tensor
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