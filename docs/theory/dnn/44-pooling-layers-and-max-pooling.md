---
id: pooling-max-pooling
title: "Pooling layers and max pooling"
sidebar_label: "44 · Pooling"
sidebar_position: 44
slug: /theory/dnn/pooling-layers-and-max-pooling
description: "How pooling reduces spatial resolution, why max pooling achieves translation invariance, average pooling vs max pooling, and global average pooling as a classifier replacement."
tags: [cnn, pooling, max-pooling, average-pooling, spatial-invariance, deep-learning]
---

# Pooling layers and max pooling

After a convolution layer produces a feature map, pooling reduces its spatial dimensions. This serves two purposes: it reduces the number of parameters in subsequent layers, and it introduces spatial invariance — small shifts in the input produce the same activation after pooling. Max pooling, which takes the maximum value in each pooling window, is the most common variant because it preserves the strongest detected feature in each region.

## One-line definition

Pooling applies a fixed aggregation function (max or average) over non-overlapping windows to reduce the spatial resolution of a feature map.

![Max pooling with a 2×2 filter and stride 2 — the maximum value in each window is kept, halving spatial dimensions](https://cs231n.stanford.edu/assets/cnn/maxpool.jpeg)
*Source: [CS231n — Convolutional Neural Networks](https://cs231n.stanford.edu/convolutional-networks/) (Stanford)*

## Max pooling

For a $P \times P$ pooling window with stride $S$ (typically $S = P$ so windows do not overlap):

$$
y[i, j] = \max_{0 \le r < P,\ 0 \le c < P} x[i \cdot S + r,\ j \cdot S + c]
$$

A $2 \times 2$ max pool with stride 2 halves both height and width. For each non-overlapping $2 \times 2$ block:
- Top-left: $\max(1,3,5,2) = 5$
- Top-right: $\max(2,4,1,0) = 4$
- Bottom-left: $\max(6,1,0,4) = 6$
- Bottom-right: $\max(5,3,2,1) = 5$

The pooled output is $[[5,4],[6,5]]$ from a $4 \times 4$ input — spatial size halved.

## Why max pooling produces translation invariance

If an edge detector fires with value 8 somewhere in a $2 \times 2$ window, the max pool output is 8 regardless of whether the edge is at position (0,0), (0,1), (1,0), or (1,1). The detector fired — its exact location within the window is discarded. This invariance to small translations is a key inductive bias for vision: a cat is still a cat whether shifted 2 pixels left or right.

## Average pooling

$$
y[i,j] = \frac{1}{P^2} \sum_{r=0}^{P-1} \sum_{c=0}^{P-1} x[i \cdot S + r,\ j \cdot S + c]
$$

Average pooling preserves all activations equally. A weak activation still contributes. Used primarily as **global average pooling (GAP)**: pool the entire feature map to a single value per channel, replacing fully connected layers.

## Global average pooling

Modern architectures (ResNet, EfficientNet, MobileNet) replace the flattened FC classifier with global average pooling:

```
Feature map: (batch, C, H, W)
After AdaptiveAvgPool2d(1): (batch, C, 1, 1) → flatten → (batch, C)
```

This eliminates dependence on fixed input size and dramatically reduces parameter count. A ResNet-50 with a $7 \times 7$ feature map and 2048 channels would need $7 \times 7 \times 2048 = 100{,}352$ inputs to the FC layer; GAP replaces this with 2048 scalars.

## Pooling vs strided convolution

| | Max pooling | Strided convolution |
|---|---|---|
| Parameters | 0 (fixed rule) | $K^2 \times C_{\text{in}} \times C_{\text{out}}$ |
| Downsampling | Fixed maximum | Learned |
| Translation invariance | Explicit | Implicit (learned) |
| Information loss | Discards non-max values | Learns what to preserve |
| Common use | Classic CNNs (VGG, AlexNet) | Modern CNNs (ResNet, EfficientNet) |

Modern best practice: replace `MaxPool2d(2)` with `Conv2d(..., stride=2)` for learnable downsampling in hidden layers.

## Output size formula for pooling

$$
H_{\text{out}} = \left\lfloor \frac{H - P}{S} \right\rfloor + 1
$$

For MaxPool2d(2, stride=2): $H_{\text{out}} = H / 2$ (when $H$ is even).

For `AdaptiveAvgPool2d(n)`: always produces output of size $n \times n$ regardless of input size — PyTorch computes the correct kernel and stride automatically.

## PyTorch code

```python
import torch
import torch.nn as nn


# ============================================================
# Max pooling and average pooling on a known input
# ============================================================
x = torch.tensor([[
    [[1., 3., 2., 4.],
     [5., 2., 1., 0.],
     [6., 1., 5., 3.],
     [0., 4., 2., 1.]]
]])   # shape: (1, 1, 4, 4)

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

print("Max pool:", max_pool(x))   # tensor([[[[5., 4.], [6., 5.]]]])
print("Avg pool:", avg_pool(x))   # mean of each 2×2 block


# ============================================================
# Global average pooling: (B, C, H, W) → (B, C)
# ============================================================
features = torch.randn(8, 512, 7, 7)      # ResNet-like feature map
gap = nn.AdaptiveAvgPool2d(1)
pooled = gap(features).flatten(1)          # → (8, 512)
print(f"After GAP: {pooled.shape}")


# ============================================================
# Classic VGG-style block: conv → conv → max pool
# ============================================================
class VGGBlock(nn.Module):
    """Two conv layers followed by max pool — standard VGG unit."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),    # halve H and W
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Modern block: conv with stride=2 (learned downsampling)
# ============================================================
class ResNetBlock(nn.Module):
    """Strided conv for downsampling — learnable, no fixed pooling."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Small CNN with GAP classifier (input-size independent)
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                    # H/2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                    # H/4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)     # (B, 128, 1, 1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)             # (B, 128)
        return self.classifier(x)


# Works with any input size:
model = SmallCNN()
for h in [32, 64, 128, 224]:
    out = model(torch.randn(2, 3, h, h))
    print(f"Input {h}×{h} → output {out.shape}")   # always (2, 10)
```

## Backpropagation through max pooling

During the forward pass, max pooling records which position had the maximum value (the **argmax**). During backpropagation, the gradient flows only to that position — all other positions receive zero gradient:

$$
\frac{\partial \mathcal{L}}{\partial x[r, c]} = \begin{cases}
\frac{\partial \mathcal{L}}{\partial y[i, j]} & \text{if } (r, c) = \arg\max \\
0 & \text{otherwise}
\end{cases}
$$

This is called **switch** or **max mask**. It means positions that weren't selected during the forward pass never receive gradient — they never learn to become more active. This can be a limitation for dense prediction tasks.

## Interview questions

<details>
<summary>Why does max pooling provide translation invariance but not equivariance?</summary>

**Equivariance** means "if the input shifts, the output shifts by the same amount." Convolution is equivariant: shifting a cat in the image shifts the feature map activations by the same amount. **Invariance** means "the output is unchanged regardless of the shift." Max pooling over a $2 \times 2$ window destroys exact position information — a feature detected at any position within the window produces the same pooled output. After pooling, the exact position within the window is lost. This is useful for classification (the cat label does not change with position) but harmful for detection (you need to know where the cat is).
</details>

<details>
<summary>What is global average pooling and why does it replace FC layers in modern CNNs?</summary>

Global average pooling takes the spatial mean of each entire channel, reducing a $(B, C, H, W)$ feature map to $(B, C)$. It eliminates FC layers whose input size depends on spatial dimensions, making the network accept any input size. It also drastically reduces parameters: a network with a 7×7 feature map and 2048 channels needs $7 \times 7 \times 2048 = 100{,}352$ inputs to FC; GAP replaces this with 2048 scalars. GAP also acts as regularization by averaging rather than memorizing spatial positions.
</details>

<details>
<summary>When would you choose average pooling over max pooling?</summary>

Max pooling retains the strongest detected feature in each region — good for detecting whether a feature is present (classification). Average pooling retains information about the average activation across the region — better when the density of a feature matters (e.g., texture classification or the final GAP before classification). Global average pooling specifically aggregates the entire feature map into one representative value per channel, corresponding to "how strongly does this channel activate across the whole image."
</details>

## Common mistakes

- Using `MaxPool2d` without specifying `stride` — PyTorch defaults `stride=kernel_size`, which is usually correct but easy to miss
- Applying pooling too aggressively early — a 4×4 max pool on a 32×32 input destroys spatial detail the first conv layers just learned
- Forgetting that max pooling has no learnable parameters but still affects gradient flow (only argmax receives gradient)
- Not squeezing the trailing dimensions after `AdaptiveAvgPool2d(1)` before the `Linear` layer — shape is `(B, C, 1, 1)`, not `(B, C)`

## Final takeaway

Max pooling halves spatial resolution while retaining the strongest local activation, providing translation invariance within each pooling window. Global average pooling collapses the entire feature map to one value per channel and is the modern replacement for flattened FC classifiers. For downsampling in hidden layers, strided convolution is increasingly preferred over max pooling because it learns how to downsample rather than applying a fixed rule.
