import os
import cv2
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass

import torch.nn as nn
import torchvision.transforms as T
import timm

@dataclass
class Config:
    violence_model_path: str = "models/vision/violence_best.pth"
    nudity_model_path: str = "models/vision/nudity_best.pth"

    pretrained_weights: str = None  # optional

    image_size: int = 224
    num_frames: int = 16

    backbone_name: str = "convnext_tiny.fb_in1k"
    embed_dim: int = 768
    proj_dim: int = 512

def load_image_rgb(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def read_video(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    idx = 0
    wanted = set(indices.tolist())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx in wanted:
            frames.append(frame)

        idx += 1

    cap.release()

    # Pad if needed
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    return frames[:num_frames]

def build_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


def preprocess(video_path, cfg=None):
    if cfg is None:
        cfg = Config()

    frames = read_video(video_path, cfg.num_frames)

    if not frames:
        raise ValueError("No frames extracted from video")

    transform = build_transform(cfg.image_size)

    frames = [transform(load_image_rgb(f)) for f in frames]
    video = torch.stack(frames)  # [T, C, H, W]

    return video.unsqueeze(0)  # [1, T, C, H, W]


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        # Backbone
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=False,
            num_classes=0
        )

        # Optional pretrained weights
        if cfg.pretrained_weights and os.path.exists(cfg.pretrained_weights):
            state = torch.load(cfg.pretrained_weights, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        # Projection
        self.proj = nn.Linear(cfg.embed_dim, cfg.proj_dim)

        # CLS token + positional embedding
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.proj_dim))
        self.pos = nn.Parameter(torch.randn(1, cfg.num_frames + 1, cfg.proj_dim))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.proj_dim,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=2
        )

        # Final head (binary)
        self.head = nn.Linear(cfg.proj_dim, 1)

    def forward(self, video):
        """
        Input: [B, T, C, H, W]
        Output: [B] logits
        """

        B, T, C, H, W = video.shape

        # Flatten frames for backbone
        x = video.view(B * T, C, H, W)

        x = self.backbone(x)
        x = self.proj(x)

        # Restore temporal dimension
        x = x.view(B, T, -1)

        # Add CLS token
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional encoding
        x = x + self.pos[:, :T + 1]

        # Transformer
        x = self.transformer(x)

        # CLS output
        z = x[:, 0]

        # Binary head
        return self.head(z).squeeze(-1)