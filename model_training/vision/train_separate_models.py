import os, random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import timm
from tqdm import tqdm


# ================= CONFIG =================
@dataclass
class Config:
    dataset_root: str = "dataset_binary"

    violence_csv: str = "dataset_binary/violence_metadata.csv"
    nudity_csv: str = "dataset_binary/nudity_metadata.csv"

    output_dir: str = "runs/separate_models"

    # 🔥 NEW
    pretrained_weights: str = "pretrained/convnext_tiny.fb_in1k.pth"

    image_size: int = 224
    num_frames: int = 16

    batch_size: int = 16
    num_workers: int = 4

    backbone_name: str = "convnext_tiny.fb_in1k"
    embed_dim: int = 768
    proj_dim: int = 512

    lr: float = 1e-4
    epochs: int = 20


CFG = Config()


# ================= UTILS =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_image_rgb(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def read_video(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= num_frames:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        start = random.randint(0, total - num_frames)
        indices = np.arange(start, start + num_frames)

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


# ================= DATASET =================
class VideoDataset(Dataset):
    def __init__(self, df, cfg, label_col, train=True):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.label_col = label_col
        self.train = train

        self.transform = T.Compose([
            T.Resize((cfg.image_size, cfg.image_size)),
            T.RandomHorizontalFlip() if train else T.Lambda(lambda x: x),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1) if train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = row["abs_filepath"]
        frames = read_video(video_path, self.cfg.num_frames)
        frames = [self.transform(load_image_rgb(f)) for f in frames]

        video = torch.stack(frames)
        label = torch.tensor(row[self.label_col], dtype=torch.float32)

        return video, label

    def __len__(self):
        return len(self.df)


# ================= MODEL =================
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # IMPORTANT CHANGE
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=False,
            num_classes=0
        )

        # LOAD PRETRAINED WEIGHTS (same as unified code)
        if os.path.exists(cfg.pretrained_weights):
            state = torch.load(cfg.pretrained_weights, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)
            print(" Loaded pretrained backbone")

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


# ================= TRAIN =================
def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train() if train else model.eval()

    total_loss = 0

    for video, label in tqdm(loader):
        video, label = video.to(device), label.to(device)

        if train:
            opt.zero_grad()

        logits = model(video)
        loss = F.binary_cross_entropy_with_logits(logits, label)

        if train:
            loss.backward()
            opt.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ================= TRAIN TASK =================
def train_task(task, csv_path, label_col, cfg, device):
    print(f"\n🔥 Training {task.upper()} model")

    df = pd.read_csv(csv_path)

    root = Path(cfg.dataset_root)
    df["abs_filepath"] = df["filepath"].apply(lambda x: str(root / x))

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_loader = DataLoader(
        VideoDataset(train_df, cfg, label_col, True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    val_loader = DataLoader(
        VideoDataset(val_df, cfg, label_col, False),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )

    model = Model(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = float("inf")

    for epoch in range(cfg.epochs):
        train_loss = run_epoch(model, train_loader, device, opt)
        val_loss = run_epoch(model, val_loader, device)

        print(f"{task} | Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{cfg.output_dir}/{task}_best.pth")


# ================= MAIN =================
def main():
    cfg = CFG
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_task("violence", cfg.violence_csv, "violence", cfg, device)
    train_task("nudity", cfg.nudity_csv, "nudity", cfg, device)


if __name__ == "__main__":
    main()