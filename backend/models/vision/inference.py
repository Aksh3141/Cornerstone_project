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
    Convert binary outputs → stable 3-class distribution
    """

    # Neutral suppressed by strongest signal
    neutral = max(0.0, 1.0 - max(p_n, p_v))

    # Normalize
    total = neutral + p_n + p_v
    if total > 0:
        neutral /= total
        p_n /= total
        p_v /= total

    return {
        "neutral": float(neutral),
        "sexual_content": float(p_n),
        "violence": float(p_v),
        "hate_speech": 0.0  # vision can't detect this
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