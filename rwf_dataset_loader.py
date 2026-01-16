"""
PyTorch Dataset and DataLoader for RWF-2000
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import torchvision.transforms as transforms

class RWF2000Dataset(Dataset):
    def __init__(self, data_info, transform=None):
        """
        Args:
            data_info: List of dicts with video info
            transform: Transformations to apply
        """
        self.data_info = data_info
        self.transform = transform
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        item = self.data_info[idx]
        
        # Load frames
        frames = np.load(item['frames_path'])  # Shape: (T, H, W, C)
        label = item['label']
        
        # Convert to tensor and normalize
        # Shape: (T, C, H, W)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        if self.transform:
            # Apply transform to each frame
            transformed_frames = []
            for frame in frames_tensor:
                transformed_frames.append(self.transform(frame))
            frames_tensor = torch.stack(transformed_frames)
        
        return frames_tensor, torch.tensor(label, dtype=torch.long)


def get_transforms(mode='train'):
    """Get data augmentation transforms"""
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(dataset_info_path, batch_size=8, num_workers=4):
    """Create train and validation dataloaders"""
    
    # Load dataset info
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Create datasets
    train_dataset = RWF2000Dataset(
        dataset_info['train'],
        transform=get_transforms('train')
    )
    
    val_dataset = RWF2000Dataset(
        dataset_info['val'],
        transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Usage example
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(
        dataset_info_path='./processed_rwf2000/dataset_info.json',
        batch_size=8,
        num_workers=4
    )
    
    # Test loading
    for frames, labels in train_loader:
        print(f"Frames shape: {frames.shape}")  # (B, T, C, H, W)
        print(f"Labels shape: {labels.shape}")  # (B,)
        break
