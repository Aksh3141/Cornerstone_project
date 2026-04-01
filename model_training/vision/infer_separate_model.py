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


# ================= UTILS =================
def load_image_rgb(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def read_video(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= num_frames:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
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
    video = torch.stack(frames)  # [T,C,H,W]
    return video.unsqueeze(0)    # [1,T,C,H,W]


# ================= INFERENCE =================
def predict(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    violence_model = load_model(CFG.violence_model_path, CFG, device)
    nudity_model = load_model(CFG.nudity_model_path, CFG, device)

    video = preprocess(video_path).to(device)

    with torch.no_grad():
        v_logit = violence_model(video)
        n_logit = nudity_model(video)

        v_prob = torch.sigmoid(v_logit).item()
        n_prob = torch.sigmoid(n_logit).item()

    # thresholds
    v_label = "VIOLENCE" if v_prob > 0.5 else "SAFE"
    n_label = "NUDITY" if n_prob > 0.5 else "SAFE"

    print("\n===== RESULTS =====")
    print(f"Video: {video_path}")
    print(f"Violence: {v_label} ({v_prob:.3f})")
    print(f"Nudity:   {n_label} ({n_prob:.3f})")

    return {
        "violence_prob": v_prob,
        "nudity_prob": n_prob,
        "violence_label": v_label,
        "nudity_label": n_label
    }


# ================= MAIN =================
if __name__ == "__main__":
    video_path = "/home/aksh/Desktop/dataset_final/dataset/violence/violence_1.mp4"  
    predict(video_path)