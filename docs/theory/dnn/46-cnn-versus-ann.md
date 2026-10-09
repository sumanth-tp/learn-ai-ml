---
id: cnn-versus-ann
title: "CNN versus ANN"
sidebar_label: "46 · CNN vs ANN"
sidebar_position: 46
slug: /theory/dnn/cnn-versus-ann
description: "Why ANNs fail on images and CNNs succeed: local connectivity, weight sharing, parameter counts, and when to choose each architecture."
tags: [cnn, ann, mlp, comparison, inductive-bias, deep-learning]
---

# CNN versus ANN

Applying a fully connected network (ANN) to images technically works — flatten the image into a vector, pass through dense layers, classify. But it fails in practice for three reasons: the parameter count explodes, the network ignores spatial structure, and it cannot generalize across positions. CNNs solve all three by exploiting the spatial structure of images with local connectivity and weight sharing.

## One-line definition

ANNs connect every input to every neuron (global, no structure); CNNs connect each neuron to a small local patch and share weights across positions (local, structured).

![Local connectivity in a CNN — each neuron connects only to a small 3D local region of the input, not the entire image](https://cs231n.stanford.edu/assets/cnn/depthcol.jpeg)
*Source: [CS231n — Convolutional Neural Networks](https://cs231n.stanford.edu/convolutional-networks/) (Stanford)*

## The parameter explosion problem

Consider classifying $224 \times 224$ RGB images (224 × 224 × 3 = 150,528 input values):

**ANN (first hidden layer):**
- 512 neurons × 150,528 inputs = **77 million parameters** — just the first layer
- Has no concept that pixel $(0, 0)$ and pixel $(0, 1)$ are spatially adjacent
- A cat in the top-left corner activates completely different neurons than a cat in the bottom-right corner

**CNN (first conv layer):**
- 64 filters × ($3 \times 3 \times 3$ kernel + 1 bias) = **1,792 parameters** — entire first layer
- The $3 \times 3$ filter slides across all positions, detecting the same features everywhere
- A cat anywhere in the image activates the same cat-detector filters

Ratio: **43,000:1** for the first layer alone.

## What makes CNNs special: three inductive biases

### 1. Local connectivity

Each CNN neuron connects only to a small local region ($3 \times 3$, $5 \times 5$) — not all 150K pixels. This is justified because:
- Adjacent pixels are correlated (spatial locality)
- Edges and textures are locally detectable
- No information from the opposite corner of the image is needed to detect a local edge

### 2. Weight sharing (translation equivariance)

The same $3 \times 3$ filter is applied to every spatial position. If the filter detects "horizontal edge," it detects horizontal edges everywhere — top-left, bottom-right, anywhere. Mathematically this is **equivariance**:

$$
f(\text{shift}(x)) = \text{shift}(f(x))
$$

Shifting the input shifts the feature map by the same amount — the detector follows the feature.

### 3. Hierarchical composition

Conv layers compose local detectors into increasingly complex detectors:
- Layer 1: edges and color blobs
- Layer 2: corners, curves, textures
- Layer 3: object parts (eyes, wheels, leaves)
- Layer 4+: whole objects and scenes

An ANN cannot compose local features because it operates globally from the first layer.

## Head-to-head comparison

| Property | ANN (MLP) | CNN |
|---|---|---|
| Connectivity | Fully connected (global) | Locally connected |
| Weight sharing | None | Yes (same filter across positions) |
| Spatial structure | Ignored | Exploited |
| Parameter count (first layer, 224×224 RGB, 512 units) | 77M | 1,792 (64 filters, 3×3×3) |
| Translation invariance | None (different weights per position) | Built-in via pooling |
| Equivariance | None | Yes (conv is equivariant to translation) |
| Good for | Tabular data, embeddings, NLP (with position) | Images, video, audio spectrograms |
| Input format | Flat vector | Spatial tensor (H×W×C) |
| Interpretability | Hard | Feature maps are inspectable |

## When ANNs are better than CNNs

CNNs assume spatial structure. When that assumption fails, ANNs win:

- **Tabular data**: columns have no spatial relationship — CNN convolutions make no sense
- **Small inputs**: for $10 \times 10$ images with tiny datasets, an MLP can outperform CNNs
- **Non-spatial embeddings**: word embeddings, user/item features — no grid structure
- **After global pooling**: once a CNN has pooled the spatial dimensions, the remaining MLP classifier is just an ANN

## PyTorch comparison

```python
import torch
import torch.nn as nn


# ============================================================
# ANN on a flattened image — works but ignores spatial structure
# ============================================================
class ImageANN(nn.Module):
    """Fully connected network for images — baseline."""
    def __init__(self, in_size=32*32*3, hidden=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# CNN on the same images — exploits spatial structure
# ============================================================
class ImageCNN(nn.Module):
    """CNN for images — local connectivity + weight sharing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                        # 32×32 → 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                        # 16×16 → 8×8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                # 8×8 → 1×1 (GAP)
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# Parameter count comparison
# ============================================================
ann = ImageANN()
cnn = ImageCNN()

ann_params = sum(p.numel() for p in ann.parameters())
cnn_params = sum(p.numel() for p in cnn.parameters())

print(f"ANN parameters: {ann_params:>10,}")   # ~4.2M for 32×32 input
print(f"CNN parameters: {cnn_params:>10,}")   # ~  186K

x = torch.randn(4, 3, 32, 32)
print(f"\nANN output: {ann(x).shape}")   # (4, 10)
print(f"CNN output: {cnn(x).shape}")   # (4, 10)


# ============================================================
# Demonstrating translation equivariance
# ============================================================
def shift_image(x, dx=5, dy=0):
    """Shift image by (dx, dy) pixels using roll."""
    return torch.roll(x, shifts=(dy, dx), dims=(-2, -1))


x_single = torch.randn(1, 3, 32, 32)
x_shifted = shift_image(x_single, dx=5)

# CNN features respond equivariantly to shifts
conv = nn.Conv2d(3, 8, 3, padding=1)
with torch.no_grad():
    f_original = conv(x_single)
    f_shifted = conv(x_shifted)
    # f_shifted ≈ shift(f_original) — feature map shifts with the input
    max_diff = (f_shifted - shift_image(f_original, dx=5)).abs().max()
    print(f"\nMax equivariance error: {max_diff.item():.6f}")  # ~0.0 (boundary effects aside)
```

## The spatial locality assumption — when it breaks

CNNs assume that nearby pixels carry correlated information. This fails when:

1. **Long-range dependencies**: classifying "is this a bedroom?" may require combining a bed in one corner and a window in another. CNNs handle this via depth (larger receptive fields) or attention (transformers).
2. **Non-uniform resolution**: satellite imagery with variable resolution objects
3. **Irregular grids**: point clouds, 3D meshes, molecular graphs — no grid structure

For these cases, graph neural networks (GNNs) or transformers are better suited.

## Modern hybrid: ViT

Vision Transformers (ViT, Dosovitskiy et al., 2020) chop the image into $16 \times 16$ patches, embed each patch as a token, and apply a transformer — effectively an attention-based ANN on patch embeddings. At large scale, ViT outperforms CNNs by learning global dependencies that CNNs can only capture at great depth. At small data scale, CNNs still win because their inductive biases (locality, equivariance) act as useful regularizers.

## Interview questions

<details>
<summary>Why can an ANN not achieve translation invariance on images, but a CNN can?</summary>

In an ANN, each pixel has its own dedicated weight in the first layer. A cat at position (10, 20) activates completely different neurons than a cat at position (50, 80) — the network sees them as different patterns. To detect a cat anywhere, the ANN must see cats at all possible positions during training. A CNN uses weight sharing: the same filter slides across all positions. Once a filter learns to detect "cat ears," it detects them anywhere. Pooling adds invariance: the max pooled output is the same regardless of small shifts within each pooling window.
</details>

<details>
<summary>What is the inductive bias of a CNN, and why is it useful for images but not for tabular data?</summary>

The CNN's inductive biases are: (1) local connectivity — relevant features are spatially local, and (2) translation equivariance — the same features appear at different positions. These are well-justified for images: a horizontal edge is horizontal regardless of where it is. For tabular data (e.g., age, income, city), the columns have no spatial relationship — "age" and "income" are not spatially adjacent in any meaningful sense. Applying a convolutional kernel to tabular columns is arbitrary and cannot exploit any genuine structure. The MLP's fully connected layers have no spatial bias and work equally well for all input arrangements.
</details>

## Common mistakes

- Flattening an image before a CNN — the CNN needs the spatial tensor `(B, C, H, W)`, not `(B, C*H*W)`
- Applying a CNN to tabular data just because "it's the more powerful architecture" — the spatial inductive bias is not just neutral, it actively misrepresents the data
- Forgetting that CNNs become ANN-like at the end: after global pooling and flattening, the classification head is a fully connected ANN

## Final takeaway

ANNs treat inputs as flat vectors with no structure. CNNs exploit the spatial structure of images via local connectivity and weight sharing, reducing parameters by orders of magnitude while gaining translation equivariance. The design hierarchy (edges → parts → objects) emerges from stacking local filters, not from architectural tricks. Use CNNs for spatially structured data (images, spectrograms); use ANNs for tabular data and embeddings where no spatial structure exists.
