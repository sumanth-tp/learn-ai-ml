---
id: 11-building-cnn
title: "Video 11: Building a CNN using PyTorch"
sidebar_label: "11 · Building a CNN"
sidebar_position: 11
description: Convolutional Neural Networks from scratch — convolution, pooling, feature maps, and image classification.
tags: [pytorch, cnn, convolution, image-classification, campusx]
---

# Building a CNN using PyTorch
**📺 CampusX — Practical Deep Learning using PyTorch | Video 11**

> **What you'll learn:** How CNNs work from the ground up — convolution operation, pooling, feature maps, and how to build and train a CNN for image classification.

---

## 1. Why CNNs Instead of ANNs for Images?

Consider a 224×224 RGB image:
- Total pixels = 224 × 224 × 3 = **150,528 inputs**
- A single hidden layer with 1024 neurons = **150M+ parameters** (just one layer!)
- The ANN treats each pixel independently — ignores **spatial relationships**
- An ANN can't handle different image sizes

**CNNs exploit three properties of images:**

| Property | What it means | How CNN uses it |
|---|---|---|
| Local structure | Nearby pixels are related | Small kernel slides over image |
| Translation invariance | A cat is a cat anywhere in image | Weight sharing across positions |
| Hierarchy | Edges → shapes → objects | Stacked conv layers |

## Visual Reference

![Full CNN architecture — Conv, Pool, FC layers stacked](https://cs231n.stanford.edu/assets/cnn/convnet.jpeg)

This is the full CNN pipeline: stacked Conv+ReLU blocks detect local patterns, pooling layers downsample the spatial dimensions, and fully-connected layers at the end map the learned feature maps to class scores. The same filter weights slide across every spatial position (weight sharing), which is why CNNs need far fewer parameters than fully-connected networks for images.

### CampusX project continuity

This lesson is deliberately positioned as the payoff after several ANN videos.
The transcript makes that comparison explicit:

- the ANN needed a lot of tuning effort to approach the high-80s / low-90s range
- a fairly basic CNN already works much better on image data

So this chapter is not just "how CNN works", but also "why architecture choice
matters more than endlessly squeezing a weaker model family."

---

## 2. The Convolution Operation

Before looking at the formula, let's understand what a convolution actually does to an image.

Imagine a 3×3 **filter** (also called a kernel) — a small grid of weights. You slide this filter across the entire image, position by position. At each position, you place the filter over that 3×3 region of the image and compute the **dot product** between the filter weights and the pixel values underneath. This produces one number. After sliding the filter across all positions, you get a 2D grid of numbers — called a **feature map**.

Each filter is specialized: one filter might detect vertical edges (high values where pixels transition from dark to light vertically), another detects horizontal edges, another detects corners. These patterns emerge through training — you don't hand-design them. That's the power of CNNs: they learn which patterns are useful for the task.

Here's why this is so much better than ANNs for images:
- A 3×3 filter has only 9 weights — but it applies those same 9 weights at every position in the image (weight sharing). A fully-connected layer would need a separate weight for every pixel connection.
- The filter "sees" a 3×3 neighborhood — it inherently captures local spatial relationships. An ANN treats pixel 1 and pixel 784 as equally related.

```
Input:  (H, W)   → e.g., (28, 28) grayscale
Filter: (k, k)   → e.g., (3, 3)
Output: (H_out, W_out)

H_out = (H + 2*padding - kernel_size) / stride + 1
W_out = (W + 2*padding - kernel_size) / stride + 1

Intuition:
- padding=0: output is smaller than input (filter can't go past the edge)
- padding=1 with kernel=3: output = same size as input ("same padding")
- stride=2: filter jumps 2 pixels at a time → output is half the size
```

```python
import torch
import torch.nn as nn

# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv = nn.Conv2d(
    in_channels=1,    # Grayscale image has 1 channel; RGB has 3
    out_channels=32,  # Learn 32 different filters = produce 32 feature maps
                      # Each filter detects a different pattern (edges, curves, etc.)
    kernel_size=3,    # Each filter is 3×3 pixels
    stride=1,         # Move filter 1 pixel at a time (dense scanning)
    padding=1         # Add 1 pixel of zeros around the border
                      # With kernel=3 and padding=1: output H,W = input H,W ("same")
)

x = torch.rand(8, 1, 28, 28)   # Batch of 8 grayscale 28×28 MNIST-like images
out = conv(x)
print(out.shape)   # (8, 32, 28, 28)
# 8 = batch size (unchanged)
# 32 = number of output channels (one per filter)
# 28, 28 = spatial dimensions (unchanged because padding=1)

# Parameter count for this layer:
print(conv.weight.shape)  # (32, 1, 3, 3) — 32 filters, each is (in_channels × 3 × 3)
print(conv.bias.shape)    # (32,) — one bias per filter
# Total: 32 × (1×3×3) + 32 = 288 + 32 = 320 parameters
# Compare to an ANN: fully-connected from 784 → 32 neurons = 784×32 = 25,088 params!
```

### Output size formula — worked examples

```python
def conv_output(H, kernel, stride=1, padding=0):
    return (H + 2*padding - kernel) // stride + 1

# MNIST: 28×28, Conv(3,3) no padding
print(conv_output(28, 3))          # 26  → (26, 26)
# With same padding
print(conv_output(28, 3, padding=1))  # 28  → (28, 28)
# With stride=2 (downsampling)
print(conv_output(28, 3, stride=2))   # 13  → (13, 13)
```

---

## 3. Pooling

Pooling **reduces spatial dimensions** (downsampling), increasing the receptive field and providing translation invariance.

```python
# MaxPool: takes the maximum in each window
pool = nn.MaxPool2d(kernel_size=2, stride=2)   # Halves H and W
x = torch.rand(8, 32, 28, 28)
print(pool(x).shape)   # (8, 32, 14, 14)

# AvgPool: takes average
avg_pool = nn.AvgPool2d(2, 2)

# Adaptive Pooling: output always a fixed size, regardless of input
global_pool = nn.AdaptiveAvgPool2d((1, 1))   # Global Average Pooling
x = torch.rand(8, 64, 7, 7)
print(global_pool(x).shape)   # (8, 64, 1, 1)   — works for ANY input size!
print(global_pool(x).squeeze().shape)   # (8, 64)  — flatten spatial dims
```

---

## 4. Building a CNN for MNIST

Before reading the code, it helps to name the big pieces of a CNN pipeline.

### What are we building, conceptually?

A CNN for image classification usually has two stages:

- **feature extractor**: convolution layers learn visual patterns such as edges, corners, textures, and simple shapes
- **classifier**: final layers use those learned patterns to decide the class label

So the model is not trying to recognize "7" or "cat" in one jump. It first builds useful visual features, then classifies using those features.

### Why does CNN code look more complicated than ANN code?

Because images have spatial structure.

In an ANN, we flatten the image into one long vector and lose the "nearby pixels belong together" idea.

In a CNN, we keep the image as:

```python
(batch, channels, height, width)
```

That allows the network to learn local patterns and reuse them across the image.

### What are the main layer types below?

- `Conv2d`: learns local visual filters
- `BatchNorm2d`: stabilizes activations during training
- `ReLU`: adds non-linearity
- `MaxPool2d`: reduces spatial size while keeping strong features
- `AdaptiveAvgPool2d`: collapses the spatial map to one value per channel
- `Linear`: turns final features into class scores

Once you read the code with that map in mind, the architecture becomes much easier to follow.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── 1. Data ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean and std
])

train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False, num_workers=2)

# ── 2. Model ─────────────────────────────────────────────────
class MNISTNet(nn.Module):
    """
    Architecture:
    Input: (N, 1, 28, 28)
    Conv1(32) → BN → ReLU → Pool → (N, 32, 14, 14)
    Conv2(64) → BN → ReLU → Pool → (N, 64, 7, 7)
    Global Avg Pool         → (N, 64, 1, 1)
    Flatten                 → (N, 64)
    FC(64→10)               → (N, 10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # ── Block 1: Input (N,1,28,28) → Output (N,32,14,14) ──
            # bias=False because BatchNorm has its own bias (β parameter).
            # Adding both would be redundant — saves a small number of params.
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            # BatchNorm2d normalizes across the spatial (H×W) dimensions per channel.
            # Stabilizes training, acts as mild regularization, enables higher LR.
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),    # inplace=True saves memory by modifying tensor in place
            nn.MaxPool2d(2, 2),       # Halves H and W: 28×28 → 14×14
                                      # Increases receptive field, provides translation invariance

            # ── Block 2: (N,32,14,14) → (N,64,14,14) → (N,64,7,7) ──
            # Double the channels: richer feature representations as we go deeper.
            # Spatial size stays the same (padding=1 with kernel=3) until MaxPool.
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 14×14 → 7×7

            # ── Block 3: (N,64,7,7) → (N,128,7,7) ──
            # No pooling here: at 7×7 we're already small; pooling would lose too much info.
            # More channels instead: push to 128 feature maps.
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling: instead of flatten→large FC layer,
        # average all 7×7 spatial positions for each of the 128 channels.
        # Result: (N, 128, 7, 7) → (N, 128, 1, 1) regardless of input resolution.
        # Benefits: works with any input size, fewer parameters, better regularization.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),           # Regularize the final representation before classification
            nn.Linear(128, num_classes)  # 128 features → 10 class logits
        )

    def forward(self, x):
        x = self.features(x)   # Extract spatial features: (N,1,28,28) → (N,128,7,7)
        x = self.gap(x)        # Collapse spatial dims: (N,128,7,7) → (N,128,1,1)
        x = x.flatten(1)       # Remove empty spatial dims: (N,128,1,1) → (N,128)
        return self.classifier(x)  # Score each class: (N,128) → (N,10)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = MNISTNet(num_classes=10).to(device)

# Parameter count
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")   # ~120K — much less than an ANN!

# ── 3. Training ──────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2,
    steps_per_epoch=len(train_loader), epochs=10
)

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def test_epoch(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

for epoch in range(1, 11):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    test_acc = test_epoch(model, test_loader, device)
print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

### Code walkthrough

- `Normalize((0.1307,), (0.3081,))` centers MNIST pixels so optimization is easier.
- each conv block increases channels while pooling reduces height and width
- `AdaptiveAvgPool2d((1, 1))` removes the need to hard-code a flatten size for the classifier
- the final layer outputs raw logits for `10` classes, which matches `CrossEntropyLoss`
- `OneCycleLR` is stepped every batch because it is designed around total training steps, not just epochs

### Transcript-style architecture walkthrough

CampusX explains the CNN as two big stages:

```text
image -> feature extractor -> classifier
```

The feature extractor is built from two conv+pool pairs:

- Conv block 1: `32` filters, `3x3`, zero padding
- MaxPool: `2x2`, stride `2`
- Conv block 2: `64` filters, `3x3`, zero padding
- MaxPool: `2x2`, stride `2`

Then the classifier uses dense layers:

- flatten
- hidden layer `128`
- hidden layer `64`
- output layer `10`

That is a very useful beginner template because it separates:

- spatial feature learning
- final class scoring

### Why reshaping matters before CNNs

In the transcript, the dataset originally exists in flattened form, so one key
step is reshaping the input back into image form before feeding it to the CNN.

```python
# Flat vector -> image tensor
x = x.reshape(-1, 1, 28, 28)
```

Read this as:

- `-1` = batch size placeholder
- `1` = grayscale channel count
- `28, 28` = image height and width

That reshape is often the first bug when moving from an ANN-based image project
to a CNN-based one.

---

## 5. Building a CNN for CIFAR-10

MNIST is grayscale and simple. CIFAR-10 is harder:

- color images instead of grayscale
- 10 object classes instead of handwritten digits
- much more visual variation

So a CIFAR model usually needs a slightly deeper feature extractor than an MNIST model.

### What changes from MNIST to CIFAR-10?

- input channels change from `1` to `3` because RGB images have red, green, and blue channels
- the feature extractor usually gets deeper
- regularization matters more because the task is harder

The code below uses repeated **conv blocks**. That pattern is common in CNN design: instead of writing every layer manually, define one standard block and reuse it.

```python
class CIFAR10Net(nn.Module):
    """
    Input: (N, 3, 32, 32)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        def conv_block(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            conv_block(3, 64),          # (N, 64, 32, 32)
            conv_block(64, 64),
            nn.MaxPool2d(2, 2),         # (N, 64, 16, 16)
            nn.Dropout2d(0.1),          # Channel dropout

            conv_block(64, 128),        # (N, 128, 16, 16)
            conv_block(128, 128),
            nn.MaxPool2d(2, 2),         # (N, 128, 8, 8)
            nn.Dropout2d(0.1),

            conv_block(128, 256),       # (N, 256, 8, 8)
            conv_block(256, 256),
            nn.AdaptiveAvgPool2d((2, 2))  # (N, 256, 2, 2) — any input size works
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # (N, 256×2×2=1024)
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

---

## 6. Interview Questions

<details>
<summary><strong>Q1: What is the receptive field in a CNN and why does it matter?</strong></summary>

The receptive field is the region of the input image that influences a particular neuron's output. A single 3×3 conv has a 3×3 RF. After two 3×3 convs, the RF is 5×5. After max pooling, the RF doubles. For the model to classify an object, the final feature map neurons must have a receptive field large enough to "see" the whole object. This is why deep networks (more layers) are better at recognizing large, complex objects — they have larger effective receptive fields.
</details>

<details>
<summary><strong>Q2: Why do CNNs use weight sharing?</strong></summary>

The same filter (set of weights) is applied at every spatial position of the image. This is weight sharing. Reasons: (1) **Parameter reduction** — instead of unique weights per position, one filter covers the whole image; (2) **Translation equivariance** — a filter that detects a vertical edge works everywhere in the image, not just at one location; (3) **Regularization** — fewer parameters → less overfitting.
</details>

<details>
<summary><strong>Q3: What is the difference between MaxPool and GlobalAveragePool?</strong></summary>

`MaxPool2d(2,2)` takes the maximum in each 2×2 window, halving the spatial dimensions. Output size depends on input size. `AdaptiveAvgPool2d((1,1))` averages ALL spatial positions per channel, producing `(N, C, 1, 1)` regardless of input size. GAP advantages: (1) accepts any image resolution; (2) far fewer parameters (no large FC layer needed); (3) better regularization; (4) enables Class Activation Maps (CAM).
</details>

<details>
<summary><strong>Q4: Why use bias=False in Conv2d when followed by BatchNorm?</strong></summary>

BatchNorm has its own bias term (β parameter) that learns the same thing as the Conv bias — shifting the activation distribution. If both Conv bias AND BatchNorm β are present, one of them is redundant and wastes a parameter. Setting `bias=False` in Conv2d eliminates this redundancy, slightly reducing parameter count with no performance impact.
</details>

---

## 🔗 References
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [CampusX Video 11](https://www.youtube.com/watch?v=hkiBZLRFvO4)
