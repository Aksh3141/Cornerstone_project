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
    # Step 1: cap overlap
    overlap = p_n * p_v

    # Step 2: adjusted signals
    p_n_adj = p_n * (1 - p_v)
    p_v_adj = p_v * (1 - p_n)

    # Step 3: neutral = remaining probability mass
    neutral = max(0.0, 1 - (p_n_adj + p_v_adj + overlap))

    # Step 4: normalize
    total = neutral + p_n_adj + p_v_adj + overlap

    if total > 0:
        neutral /= total
        p_n_adj /= total
        p_v_adj /= total

    return {
        "neutral": float(neutral),
        "sexual_content": float(p_n_adj),
        "violence": float(p_v_adj),
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