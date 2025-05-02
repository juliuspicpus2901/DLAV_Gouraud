
# DLAV Project - Phase 1

This notebook trains a basic planner for autonomous driving. The planner takes as input dash cam images and the vehicle's past positions, and outputs a predicted future trajectory.

## Features

- Dataset handling and preprocessing
- Simple neural network model for trajectory prediction
- Training loop with logging and loss computation
- Evaluation with basic metrics (e.g., smoothness)

---

## Setup

Ensure you have Python 3 and the following packages installed:

```bash
pip install torch torchvision numpy matplotlib
```

You may also need Jupyter to run the notebook:

```bash
pip install notebook
```

---

## Running the Notebook

1. Open the notebook:

```bash
jupyter notebook "DLAV_Phase1 (1).ipynb"
```

2. Execute all cells sequentially. This includes:
   - Loading the dataset
   - Initializing the model
   - Training the planner
   - Evaluating performance (basic metrics and loss)

---

## Dataset

The dataset is handled via a custom `DrivingDataset` class that:
- Loads image frames and past vehicle positions
- Applies preprocessing
- Data augmentation using pytorch.transforms
- Returns input-output pairs for training


---

## Model

The model used is a simple neural network defined in the `DrivingModel` (or similarly named) class. It contains:
    - A GRU-based trajectory encoder
    - Layer normalization and dropout
    - Transformer-based decoder

---

## Training

Training is done through the `train` function. It includes:
- Epoch-based training
- Batch processing
- Loss computation and logging
- Metric evaluation (e.g., smoothness loss)

---

## Inference

The notebook does not provide a standalone inference function. To run inference:
```python
model.eval()
with torch.no_grad():
    prediction = model(input_image, past_positions)
```

Consider creating an `infer()` helper for reuse.

---

## Metrics

The loggers computes the following metrics:
- ADE - Average displacement error
- FDE - Forward displacement error
- Heading Error - Final difference between predicted and actual heading
- Miss rate - percentage of final predicted position more than 20% off