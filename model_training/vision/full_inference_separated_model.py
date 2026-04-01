import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

import torch.nn as nn
import torchvision.transforms as T
import timm

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


# ================= CONFIG =================
@dataclass
class Config:
    violence_model_path: str = "separate_models/violence_best.pth"
    nudity_model_path: str = "separate_models/nudity_best.pth"

    pretrained_weights: str = "pretrained/convnext_tiny.fb_in1k.pth"

    image_size: int = 224
    num_frames: int = 16

    backbone_name: str = "convnext_tiny.fb_in1k"
    embed_dim: int = 768
    proj_dim: int = 512


CFG = Config()

DATASET_ROOT = "/home/aksh/Desktop/dataset_final/dataset"
CLASSES = ["nudity", "violence", "neutral"]

#CLASSES = ["nudity", "violence"]

# ================= UTILS =================
def load_image_rgb(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def read_video(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    frames, idx = [], 0
    wanted = set(indices.tolist())

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            frames.append(frame)
        idx += 1

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames[:num_frames]


# ================= MODEL =================
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=False,
            num_classes=0
        )

        if os.path.exists(cfg.pretrained_weights):
            state = torch.load(cfg.pretrained_weights, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        self.proj = nn.Linear(cfg.embed_dim, cfg.proj_dim)

        self.cls = nn.Parameter(torch.randn(1, 1, cfg.proj_dim))
        self.pos = nn.Parameter(torch.randn(1, cfg.num_frames + 1, cfg.proj_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.proj_dim, 8, 1024, batch_first=True),
            num_layers=2
        )

        self.head = nn.Linear(cfg.proj_dim, 1)

    def forward(self, video):
        B, T, C, H, W = video.shape

        x = video.view(B * T, C, H, W)
        x = self.backbone(x)
        x = self.proj(x)
        x = x.view(B, T, -1)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :T + 1]

        x = self.transformer(x)
        z = x[:, 0]

        return self.head(z).squeeze(-1)


# ================= LOAD MODELS =================
def load_model(path, cfg, device):
    model = Model(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ================= PREPROCESS =================
transform = T.Compose([
    T.Resize((CFG.image_size, CFG.image_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def preprocess(video_path):
    frames = read_video(video_path, CFG.num_frames)
    frames = [transform(load_image_rgb(f)) for f in frames]
    video = torch.stack(frames)
    return video.unsqueeze(0)


# ================= SINGLE INFERENCE =================
def predict_with_models(video_path, violence_model, nudity_model, device):
    video = preprocess(video_path).to(device)

    with torch.no_grad():
        v_logit = violence_model(video)
        n_logit = nudity_model(video)

        v_prob = torch.sigmoid(v_logit).item()
        n_prob = torch.sigmoid(n_logit).item()

    v_pred = 1 if v_prob > 0.1 else 0
    n_pred = 1 if n_prob > 0.9999 else 0

    final_pred = 1 if (v_pred or n_pred) else 0

    return v_prob, n_prob, v_pred, n_pred, final_pred


# ================= DATASET =================
def load_dataset(root):
    data = []
    for label in CLASSES:
        folder = os.path.join(root, label)

        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if fname.endswith(".mp4"):
                data.append((os.path.join(folder, fname), label))

    return data


def gt_to_binary(label):
    return 0 if label == "neutral" else 1


def gt_violence(label):
    return 1 if label == "violence" else 0


def gt_nudity(label):
    return 1 if label == "nudity" else 0


# ================= EVALUATION =================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models once...")
    violence_model = load_model(CFG.violence_model_path, CFG, device)
    nudity_model = load_model(CFG.nudity_model_path, CFG, device)

    dataset = load_dataset(DATASET_ROOT)

    y_true, y_pred = [], []
    y_pred_v, y_pred_n = [], []
    y_true_v, y_true_n = [], []

    print(f"Total videos: {len(dataset)}\n")

    for i, (video_path, label) in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] {video_path}")

        try:
            v_prob, n_prob, v_pred, n_pred, final_pred = predict_with_models(
                video_path, violence_model, nudity_model, device
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        gt_bin = gt_to_binary(label)
        gt_v = gt_violence(label)
        gt_n = gt_nudity(label)

        # store
        y_true.append(gt_bin)
        y_pred.append(final_pred)

        y_true_v.append(gt_v)
        y_pred_v.append(v_pred)

        y_true_n.append(gt_n)
        y_pred_n.append(n_pred)

        # ================= PRINT PER SAMPLE =================
        print(f"GT Label        : {label}")
        print(f"Violence -> prob: {v_prob:.4f}, pred: {v_pred}")
        print(f"Nudity   -> prob: {n_prob:.4f}, pred: {n_pred}")
        print(f"Final Prediction: {final_pred}")

    # ================= FINAL METRICS =================
    print("\n==============================")
    print("FINAL RESULTS (COMBINED)")
    print("==============================")
    print(classification_report(y_true, y_pred, target_names=["SAFE", "UNSAFE"], digits=4))

    print("\n==============================")
    print("VIOLENCE MODEL RESULTS")
    print("==============================")
    print(classification_report(y_true_v, y_pred_v, target_names=["NO-V", "V"], digits=4))

    print("\n==============================")
    print("NUDITY MODEL RESULTS")
    print("==============================")
    print(classification_report(y_true_n, y_pred_n, target_names=["NO-N", "N"], digits=4))


# ================= MAIN =================
if __name__ == "__main__":
    evaluate()
