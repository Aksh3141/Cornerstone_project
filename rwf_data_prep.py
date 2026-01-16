"""
RWF-2000 Dataset Preparation Script
Extracts frames from videos and prepares dataset for training
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class RWF2000DataPreparation:
    def __init__(self, dataset_root, output_dir, frames_per_video=16):
        """
        Args:
            dataset_root: Path to RWF-2000 dataset
                Expected structure:
                RWF-2000/
                ├── train/
                │   ├── Fight/
                │   └── NonFight/
                └── val/
                    ├── Fight/
                    └── NonFight/
            output_dir: Where to save extracted frames
            frames_per_video: Number of frames to extract per video
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.frames_per_video = frames_per_video
        
    def extract_frames_uniform(self, video_path, num_frames=16):
        """Extract frames uniformly from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            # If video has fewer frames, extract all
            frame_indices = list(range(total_frames))
        else:
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize to standard size
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def prepare_dataset(self):
        """Prepare entire dataset"""
        splits = ['train', 'val']
        classes = ['Fight', 'NonFight']
        
        dataset_info = {
            'train': [],
            'val': []
        }
        
        for split in splits:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for class_name in classes:
                class_dir = self.dataset_root / split / class_name
                output_class_dir = split_dir / class_name
                output_class_dir.mkdir(exist_ok=True)
                
                video_files = list(class_dir.glob('*.avi')) + list(class_dir.glob('*.mp4'))
                
                print(f"\nProcessing {split}/{class_name}: {len(video_files)} videos")
                
                for video_path in tqdm(video_files):
                    try:
                        # Extract frames
                        frames = self.extract_frames_uniform(video_path, self.frames_per_video)
                        
                        if len(frames) > 0:
                            # Save as numpy array
                            output_path = output_class_dir / f"{video_path.stem}.npy"
                            np.save(output_path, frames)
                            
                            # Record metadata
                            dataset_info[split].append({
                                'video_name': video_path.name,
                                'frames_path': str(output_path),
                                'label': 1 if class_name == 'Fight' else 0,
                                'num_frames': len(frames)
                            })
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
        
        # Save dataset info
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✓ Dataset preparation complete!")
        print(f"Train samples: {len(dataset_info['train'])}")
        print(f"Val samples: {len(dataset_info['val'])}")
        
        return dataset_info


# Usage example
if __name__ == "__main__":
    # Configure paths
    DATASET_ROOT = "/home/aksh/Desktop/RWF-2000"  
    OUTPUT_DIR = "./processed_rwf2000"
    
    # Prepare dataset
    preparator = RWF2000DataPreparation(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        frames_per_video=16  # Extract 16 frames per video
    )
    
    dataset_info = preparator.prepare_dataset()
