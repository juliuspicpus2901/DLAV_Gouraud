
# 🚗 DLAV Phase 1 – End-to-End Trajectory Planner

This repository contains an end-to-end deep learning pipeline for predicting future trajectories of a self-driving vehicle from RGB camera images and past motion data, as part of the DLAV project.

---

## 📁 Project Structure

```
.
├── DLAV_Phase1.ipynb               # Main Colab notebook for experimentation
├── model_yolo_transformer.py       # YOLOv11 + Transformer-based trajectory planner
├── dataset_semantic.py             # Augmented dataset using camera, semantic labels, and history
├── train_yolo_transformer.py       # Training script using momentum and smoothness regularization
├── logger.py                       # Custom logger for ADE, FDE, heading error, and miss rate
├── README.md                       # You're here!
```

---

## 📊 Data Format

Each input sample is a `.pkl` file with:

- `'camera'`: (200, 300, 3) RGB image
- `'semantic_label'`: (200, 300) class index per pixel (0–14)
- `'sdc_history_feature'`: (21, 3) → past [x, y, heading]
- `'sdc_future_feature'`: (60, 3) → ground truth future trajectory

---

## 🚀 How to Train

```python
from dataset_semantic import SemanticDrivingDataset
from model_yolo_transformer import YoloSemanticTransformerPlanner
from train_yolo_transformer import train
from logger import Logger
from torch.utils.data import DataLoader

# Load data
train_dataset = SemanticDrivingDataset(train_files)
val_dataset = SemanticDrivingDataset(val_files)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize model and train
model = YoloSemanticTransformerPlanner(yolo_weights_path="yolov11.pt")
logger = Logger()
train(model, train_loader, val_loader, logger, num_epochs=30)
```

---

## 🧪 Inference Example

```python
model.eval()
with torch.no_grad():
    batch = next(iter(val_loader))
    pred = model(batch['camera'].to(device),
                 batch['semantic'].to(device),
                 batch['history'].to(device))  # shape: (B, 60, 3)
```

---

## 📈 Evaluation Metrics

| Metric       | Description                        | Good Value   |
|--------------|------------------------------------|--------------|
| ADE          | Avg. L2 error over all timesteps   | < 1.5 m      |
| FDE          | L2 error at final step             | < 2.5 m      |
| MissRate@2m  | % of samples with FDE > 2m         | < 20%        |
| HeadingErr   | Final step heading angle error     | ~0.2 rad     |

---

## ✅ Features

- Pretrained YOLOv11 backbone
- Semantic map integration
- GRU or MLP-based trajectory encoders
- Transformer-based trajectory decoder
- Smoothness regularization loss
- Momentum SGD optimizer
- Advanced logger with ADE/FDE/heading metrics

---

## 📌 To-Do

- [ ] Test-time inference on held-out test set
- [ ] Export CSV for leaderboard submission
- [ ] Visualize plans over semantic maps

---

## 🔗 Requirements

- PyTorch ≥ 1.12
- torchvision
- YOLOv11 repo (for DetectMultiBackend)
- Python ≥ 3.8

---

## 📝 License

For educational use in DLAV Phase 1.
