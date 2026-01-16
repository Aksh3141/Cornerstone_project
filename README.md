# Violence Detection System

## Overview
This repository contains a **video-based violence detection system** that classifies videos into:
- **Violence**
- **Non-Violence**

It also includes an optional **text-description module** for detected violent scenes using a vision–language model.

The system learns:
- **Spatial features** → using a CNN  
- **Temporal motion patterns** → using an LSTM  

---

## System Pipeline

1. **Video Input** – raw video is provided to the system  
2. **Frame Sampling** – fixed number of frames extracted per video  
3. **Feature Extraction** – CNN extracts features from each frame  
4. **Temporal Modeling** – LSTM processes frame sequences  
5. **Classification** – system predicts violence or non-violence  
6. **Optional Description** – generates a text explanation for violent scenes  

---

## Dataset

### RWF-2000
A benchmark dataset for violence detection containing:
- Two balanced classes: **Fight** and **NonFight**
- Real-world CCTV and surveillance footage
- Both indoor and outdoor environments  
- Strong motion cues and temporal patterns

This dataset is used for training, validation, and evaluation.

---

## Model Architecture

### CNN + LSTM Hybrid

| Component | Description |
|----------|-------------|
| **CNN Backbone (ResNet-50)** | Extracts spatial features from video frames |
| **Bidirectional LSTM** | Learns temporal motions and patterns |
| **Classification Head** | Outputs probability for Violence / Non-Violence |
| **Dropout Layers** | Prevent overfitting |

### Why This Architecture?
- CNN understands **what** is in the frame  
- LSTM understands **how** movement evolves  
- Ideal for detecting violent behaviors in videos

### Optional Variant
- **EfficientNet + LSTM** for faster inference  

---

## Evaluation Results

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 82.25% |
| **Precision** | 0.80 |
| **Recall** | 0.86 |
| **F1-Score** | 0.829 |
| **Specificity** | 0.785 |
| **AUC-ROC** | 0.885 |

---

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| **Non-Violence** | 0.849 | 0.785 | 0.816 |
| **Violence** | 0.800 | 0.860 | 0.829 |

---

### Confusion Matrix

|                      | Predicted Non-Violence | Predicted Violence |
|----------------------|------------------------|--------------------|
| **Actual Non-Violence** | 157 | 43 |
| **Actual Violence**     | 28  | 172 |

---

## Output Format

### Classification Output Includes:
- Violence flag (True/False)  
- Confidence score  
- Probability of each class  

### Optional Description Output Includes:
- Type of violence  
- Confidence  
- A short natural-language description  
- Scene context (location, severity, participants)  

---







