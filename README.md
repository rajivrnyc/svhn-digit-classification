# SVHN Digit Classification with CNN & AlexNet

Digit classification on the [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/) dataset using two PyTorch-based CNN architectures: a custom Simple CNN and an adapted AlexNet.

---

## Dataset

The SVHN dataset consists of real-world digit images cropped from Google Street View house numbers.

| Split | Samples |
|-------|---------|
| Train + Extra (combined) | 604,388 |
| Validation (10% of above) | 60,438 |
| Test | 26,032 |

- **Image dimensions:** 32×32 RGB (3 channels)
- **Classes:** 10 (digits 0–9)
- **Format:** `torch.float32` tensors, shape `[3, 32, 32]`

---

## Setup & Installation

### Requirements

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

### Download Data

The SVHN dataset is downloaded automatically via `torchvision.datasets.SVHN` on first run. Set the `root` path to your preferred local directory.

---

## Preprocessing & Augmentation

### Normalization
All images are normalized to `[-1, 1]` using mean and std of `[0.5, 0.5, 0.5]` per channel.

### Data Augmentation (training only)
To improve generalization and simulate real-world variation, the following augmentations are applied to the training set:

- **Random Crop** (32×32 with padding=4) — simulates slight positional shifts
- **Random Horizontal Flip** — creates mirror variations of digits
- **Color Jitter** (brightness, contrast, saturation, hue) — simulates varying lighting conditions

Validation and test sets use only normalization — no augmentation.

### Batching
All splits are loaded using `DataLoader` with a batch size of **64**. Training data is shuffled each epoch; validation and test data are not.

---

## Model Architectures

### Simple CNN

A lightweight two-block CNN designed for efficient training.

| Layer | Details |
|-------|---------|
| Conv Block 1 | Conv2d(3→32), BatchNorm, ReLU, Dropout(0.1), MaxPool(2) |
| Conv Block 2 | Conv2d(32→64), BatchNorm, ReLU, Dropout(0.1), MaxPool(2) |
| FC Layer | Linear(4096→256), BatchNorm, ReLU, Dropout(0.3) |
| Output | Linear(256→10) |

### AlexNet (adapted for 32×32)

A deeper architecture with 5 convolutional layers followed by 3 fully connected layers.

| Layer | Details |
|-------|---------|
| Conv 1 | Conv2d(3→64), BatchNorm, ReLU, MaxPool(2) → 16×16 |
| Conv 2 | Conv2d(64→192), BatchNorm, ReLU, MaxPool(2) → 8×8 |
| Conv 3 | Conv2d(192→384), ReLU |
| Conv 4 | Conv2d(384→256), ReLU |
| Conv 5 | Conv2d(256→256), ReLU, MaxPool(2) → 4×4 |
| FC 1 | Linear(4096→1024), BatchNorm, ReLU, Dropout(0.5) |
| FC 2 | Linear(1024→512), BatchNorm, ReLU, Dropout(0.5) |
| Output | Linear(512→10) |

Both models use **Adam optimizer** (lr=0.001) and **CrossEntropyLoss**.

---

## Training

Models are trained for **10 epochs**. The training loop tracks loss per epoch and evaluates F1, precision, and recall on the validation set after each epoch.

To train both models:

```python
train_losses_simple, val_losses_simple = train_model(
    simple_model, train_loader, val_loader, criterion, optimizer_simple, device
)

train_losses_alex, val_losses_alex = train_model(
    alexnet_model, train_loader, val_loader, criterion, optimizer_alexnet, device
)
```

---

## Results

### Simple CNN

| Epoch | Train Loss | Val Loss | F1 | Precision | Recall |
|-------|-----------|----------|----|-----------|--------|
| 1 | 0.5760 | 0.3107 | 0.9106 | 0.9115 | 0.9108 |
| 5 | 0.2602 | 0.1914 | 0.9450 | 0.9453 | 0.9451 |
| 10 | 0.2264 | 0.1747 | 0.9521 | 0.9524 | 0.9521 |

**Test Accuracy: 93.30% | Test Loss: 0.2417**

### AlexNet

| Epoch | Train Loss | Val Loss | F1 | Precision | Recall |
|-------|-----------|----------|----|-----------|--------|
| 1 | 0.4554 | 0.1688 | 0.9504 | 0.9507 | 0.9505 |
| 5 | 0.1221 | 0.1046 | 0.9707 | 0.9707 | 0.9706 |
| 10 | 0.0953 | 0.0873 | 0.9763 | 0.9763 | 0.9763 |

**Test Accuracy: 96.07% | Test Loss: 0.1487**

### Comparison

| Model | Accuracy | F1 | Precision | Recall | Test Loss | Time/Sample |
|-------|----------|----|-----------|--------|-----------|-------------|
| Simple CNN | 93.30% | 0.9329 | 0.9332 | 0.9330 | 0.2417 | ~0.000394s |
| AlexNet | 96.07% | 0.9607 | 0.9610 | 0.9607 | 0.1487 | ~0.000401s |

AlexNet outperforms the Simple CNN across all metrics, owing to its deeper feature extraction and stronger regularization, at a minimal additional cost in training time per sample.

---

## Overfitting Prevention

Both models use the following regularization strategies:

- **Batch Normalization** — stabilizes activations and acts as a mild regularizer
- **Dropout** — 10% in Simple CNN conv layers, 30% in FC layer; 50% in AlexNet FC layers
- **Data Augmentation** — reduces reliance on specific image characteristics
- **Validation monitoring** — loss and metrics tracked each epoch to detect divergence

---

## Technologies

- Python 3.x
- PyTorch & torchvision
- scikit-learn (metrics)
- NumPy / Matplotlib
