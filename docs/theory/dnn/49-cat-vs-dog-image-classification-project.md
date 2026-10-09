---
id: cat-vs-dog-project
title: "Cat vs dog image classification project"
sidebar_label: "49 · Cat vs Dog Project"
sidebar_position: 49
slug: /theory/dnn/cat-vs-dog-image-classification-project
description: "End-to-end binary image classification with a custom CNN: dataset loading, augmentation, training loop, validation, and common debugging patterns."
tags: [cnn, image-classification, project, pytorch, training-loop, deep-learning]
---

# Cat vs dog image classification project

Binary image classification — distinguishing cats from dogs — is the standard "hello world" for CNNs. It combines all the concepts from the CNN module into one complete pipeline: data loading, preprocessing, augmentation, model design, training loop, validation, and debugging. This project uses the Kaggle Dogs vs. Cats dataset but the pipeline applies to any binary classification task.

![A full ConvNet architecture — activations flowing through conv, pooling, and FC layers as in this cat vs dog pipeline](https://cs231n.stanford.edu/assets/cnn/convnet.jpeg)
*Source: [CS231n — Convolutional Neural Networks](https://cs231n.stanford.edu/convolutional-networks/) (Stanford)*

## Problem setup

| | |
|---|---|
| Task | Binary classification: cat (0) vs dog (1) |
| Input | RGB images, resized to 224×224 |
| Output | Single logit (sigmoid → probability) |
| Loss | Binary cross-entropy |
| Metric | Accuracy |

## Dataset structure

The Kaggle dataset organizes images by class name in the filename:
```
data/train/
    cat.0.jpg  cat.1.jpg  ...  cat.12499.jpg
    dog.0.jpg  dog.1.jpg  ...  dog.12499.jpg
data/test/
    1.jpg  2.jpg  ...
```

We create a PyTorch Dataset class to load and label images from this structure.

## Full end-to-end implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# Dataset
# ============================================================
class CatDogDataset(Dataset):
    """
    Expects images named 'cat.*.jpg' and 'dog.*.jpg' in a flat directory.
    Label: 0 = cat, 1 = dog.
    """
    def __init__(self, folder: str, transform=None):
        self.paths = list(Path(folder).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Label from filename: 'cat.0.jpg' → 0, 'dog.0.jpg' → 1
        label = 0 if path.stem.startswith("cat") else 1
        return image, label


# ============================================================
# Transforms: separate augmentation for train vs validation
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# Model: custom CNN from scratch
# ============================================================
class CatDogCNN(nn.Module):
    """
    5-block CNN for binary cat/dog classification.
    Uses same padding throughout, stride=2 for downsampling.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),

            # Block 2: 112 → 56
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),

            # Block 3: 56 → 28
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),

            # Block 4: 28 → 14
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # → (B, 256, 1, 1)
            nn.Flatten(),                # → (B, 256)
            nn.Dropout(dropout),
            nn.Linear(256, 1),           # binary: single logit
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


# ============================================================
# Training utilities
# ============================================================
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits > 0).long()    # sigmoid > 0.5 ↔ logit > 0
    return (preds == labels).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), labels.long())
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += accuracy(logits, labels.long())
    n = len(loader)
    return total_loss / n, total_acc / n


# ============================================================
# Main training loop
# ============================================================
def train(data_dir: str, num_epochs: int = 20, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset and split
    full_dataset = CatDogDataset(data_dir, transform=train_transform)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform   # override augmentation for val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = CatDogCNN(dropout=0.5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        epochs=num_epochs, steps_per_epoch=len(train_loader),
    )

    # History
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{num_epochs}  "
              f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_catdog.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model, history


# ============================================================
# Inference on a single image
# ============================================================
@torch.no_grad()
def predict(model, image_path: str, device: str = "cpu") -> dict:
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(device)
    logit = model(tensor).item()
    prob_dog = torch.sigmoid(torch.tensor(logit)).item()
    return {
        "prediction": "dog" if prob_dog > 0.5 else "cat",
        "confidence": prob_dog if prob_dog > 0.5 else 1 - prob_dog,
        "prob_dog": prob_dog,
        "prob_cat": 1 - prob_dog,
    }
```

## Using a pretrained backbone instead

For small datasets (< 5000 images per class), training from scratch is risky — the model may overfit. Use a pretrained ResNet-50 backbone:

```python
def build_pretrained_model(freeze_backbone: bool = True) -> nn.Module:
    """ResNet-50 pretrained on ImageNet, fine-tuned for cat/dog."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
```

With a frozen backbone, only the final linear layer is trained — fast and effective even with 1000 images per class.

## Debugging checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Train acc ~50%, no improvement | Learning rate too low or too high | Try `lr=1e-3`, check loss is decreasing in first 10 batches |
| Train acc high, val acc ~50% | Severe overfitting | Add dropout, augmentation, reduce model size, or use pretrained backbone |
| Loss = NaN after first batch | Learning rate too large or bad init | Reduce `lr`, add gradient clipping, check input normalization |
| Val acc fluctuating wildly | Batch size too small for val | Use larger batch for validation, increase val set size |
| Training very slow | Images loading from disk each step | Use `num_workers > 0`, `pin_memory=True`, or cache to RAM |

## Training curve interpretation

```
Epoch  1: train=0.693 val=0.693  → Both near log(2); model outputting ~0.5 for all inputs
Epoch  5: train=0.420 val=0.450  → Learning, slight gap
Epoch 10: train=0.250 val=0.320  → Overfitting starts: val loss growing while train falls
Epoch 15: train=0.150 val=0.380  → Clear overfit: train ≪ val
```

If val loss increases while train loss decreases → overfit. Remedies: more augmentation, dropout, weight decay, early stopping, pretrained backbone.

## Interview questions

<details>
<summary>Why use BCEWithLogitsLoss instead of BCE + sigmoid?</summary>

`BCEWithLogitsLoss` computes sigmoid and binary cross-entropy together using the numerically stable log-sum-exp trick: $\log(1 + e^x)$ is unstable for large $x$, but the equivalent $x - x\cdot y + \log(1 + e^{-x})$ is stable. Using `BCE(sigmoid(logit), y)` can produce NaN or inf gradients for large logits. Always use `BCEWithLogitsLoss` — pass raw logits, not probabilities.
</details>

<details>
<summary>What is the OneCycleLR scheduler and why is it effective?</summary>

OneCycleLR starts from a low learning rate, warms up to a maximum over the first 30% of training, then cosine-decays to near zero. This combines: (1) warmup — prevents early divergence when parameters are far from good values, and (2) annealing — finds a flat minimum at the end. The "1-cycle" policy was shown by Smith (2018) to often converge faster than constant LR with less hyperparameter tuning. It is a good default for image classification tasks.
</details>

## Common mistakes

- Not separating train and val transforms — augmentation should only be applied to training data, not validation
- Using the same dataset instance for both train and val after `random_split` — they share the transform; override `val_ds.dataset.transform` explicitly, or create separate dataset instances
- Forgetting `model.eval()` during validation — batch norm and dropout behave differently in eval mode
- Reporting training accuracy as the model's performance — always report validation accuracy

## Final takeaway

The cat vs dog pipeline demonstrates the complete CNN workflow: `Dataset` → `DataLoader` → model → loss → optimizer → scheduler → train loop → eval loop. Key practices: separate augmentation for train and val, `BCEWithLogitsLoss` for binary tasks, gradient clipping for stability, and `AdaptiveAvgPool2d` for input-size-independent classifiers. For small datasets, always start with a pretrained backbone rather than training from scratch.
