---
id: pytorch-functional-style
title: "Building non-linear models with the PyTorch functional style"
sidebar_label: "54 · PyTorch Functional Style"
sidebar_position: 54
slug: /theory/dnn/building-non-linear-models-with-the-pytorch-functional-style
description: "The difference between torch.nn modules and torch.nn.functional, when to use each, and how to build flexible architectures with explicit forward methods."
tags: [pytorch, functional, nn-module, architecture, deep-learning]
---

# Building non-linear models with the PyTorch functional style

PyTorch provides two parallel APIs for neural network operations: `torch.nn` modules (stateful, object-oriented) and `torch.nn.functional` (stateless functions). Understanding both and when to use each is essential for building flexible, production-quality models. Most architectures use both — modules for layers with learnable parameters, functional calls for operations like activations, dropout, and loss computation.

## One-line definition

`torch.nn.functional` (imported as `F`) provides stateless functions for neural network operations. `torch.nn` provides stateful module wrappers that own their parameters. In a well-structured model, parameters live in modules and operations use functional calls.

![Neuron arrangement in a ConvNet — `nn.Module` layers stack to form the network graph; `F.*` operations execute within each layer's `forward()`](https://cs231n.stanford.edu/assets/cnn/cnn.jpeg)
*Source: [CS231n — Convolutional Neural Networks](https://cs231n.stanford.edu/convolutional-networks/) (Stanford)*

## `nn.Module` vs `torch.nn.functional`

| | `nn.Module` (e.g., `nn.ReLU`) | `F.relu` |
|---|---|---|
| State | Stateful (optional parameters) | Stateless (no parameters) |
| When to use | When the operation has learnable parameters | When it doesn't |
| Example | `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d` | `F.relu`, `F.dropout`, `F.softmax` |
| train/eval aware | Yes (`nn.Dropout`, `nn.BatchNorm2d`) | Must pass `training` explicitly |
| Common in `forward()` | Via `self.layer(x)` | Via `F.function(x)` |

**Rule**: if a layer has learnable weights, use `nn.Module`. If it is a pure function (activation, normalization without learned params, loss), use `F`.

## The explicit forward method pattern

The `nn.Sequential` container is convenient but inflexible — it cannot express skip connections, multi-head outputs, or conditional computation. The explicit `forward` method is the correct pattern for any non-trivial architecture:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Example: ResNet-style block with skip connection
# (Cannot be expressed with nn.Sequential)
# ============================================================
class ResBlock(nn.Module):
    """
    A residual block: output = ReLU(F(x) + x)
    where F(x) is conv → BN → ReLU → conv → BN
    """
    def __init__(self, channels: int):
        super().__init__()
        # Learnable parameters → nn.Module
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                   # save input for the skip connection

        # Non-linearities → F. (stateless, no parameters)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return F.relu(out + residual)  # add skip connection before final ReLU


# ============================================================
# Example: multi-output model
# (Two prediction heads from the same backbone)
# ============================================================
class MultiTaskCNN(nn.Module):
    """
    Shared backbone with two task-specific heads.
    Cannot be expressed with nn.Sequential.
    """
    def __init__(self, num_classes_a: int, num_classes_b: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_a = nn.Linear(128, num_classes_a)   # e.g., object type
        self.head_b = nn.Linear(128, num_classes_b)   # e.g., scene type

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(self.backbone(x)).flatten(1)  # (B, 128)
        return self.head_a(features), self.head_b(features)


# ============================================================
# Example: dropout in functional style
# ============================================================
class MLPWithFunctionalDropout(nn.Module):
    """
    Uses F.dropout instead of nn.Dropout to demonstrate
    how training flag is managed.
    """
    def __init__(self, in_features: int, hidden: int, out_features: int,
                 dropout_p: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        # self.training is True during model.train(), False during model.eval()
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.fc2(x)
```

## The critical detail: `training` flag

`F.dropout` and `F.batch_norm` require you to pass the `training` flag explicitly. The `self.training` attribute of any `nn.Module` is automatically set to `True` by `model.train()` and `False` by `model.eval()`.

```python
class DropoutDemo(nn.Module):
    def forward(self, x):
        # Correct: pass self.training
        return F.dropout(x, p=0.5, training=self.training)

class DropoutBug(nn.Module):
    def forward(self, x):
        # WRONG: always drops at inference too
        return F.dropout(x, p=0.5, training=True)
```

Using `nn.Dropout` handles this automatically. Using `F.dropout` requires explicit `training=self.training`.

## When functional calls are mandatory

Some operations have no module equivalent and must use `F`:

```python
class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Scaled dot-product attention — functional call, no parameters
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        return out


class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))    # relu: functional
        x = F.adaptive_avg_pool2d(x, (1, 1)) # pooling: functional
        x = x.flatten(1)
        x = F.log_softmax(self.fc(x), dim=1) # log_softmax: functional
        return x
```

## Complete custom architecture example

```python
class CustomCNN(nn.Module):
    """
    Flexible CNN demonstrating the mix of nn.Module and F style.
    Shows:
    - Learnable layers as module attributes
    - Activations and dropout via F
    - Skip connections via explicit forward
    - Conditional computation (use_attention flag)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 use_attention: bool = False, dropout: float = 0.3):
        super().__init__()
        self.use_attention = use_attention

        # Block 1
        self.conv1a = nn.Conv2d(in_channels, 32, 3, padding=1, bias=False)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2 (with skip connection — need 1×1 conv for channel adjustment)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv2_skip = nn.Conv2d(32, 64, 1, bias=False)  # projection
        self.bn2 = nn.BatchNorm2d(64)

        # Optional spatial attention
        if use_attention:
            self.attn_conv = nn.Conv2d(64, 1, 1)  # channel → 1 for spatial mask

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = dropout
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        out = F.relu(self.conv1a(x))
        out = self.bn1(self.conv1b(out))
        out = F.relu(out)
        out = F.max_pool2d(out, 2)             # downsample

        # Block 2 with skip connection
        skip = self.conv2_skip(out)            # project channels: 32 → 64
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + skip                       # residual add (channels now match)
        out = F.relu(out)

        # Optional spatial attention
        if self.use_attention:
            attn = torch.sigmoid(self.attn_conv(out))  # (B, 1, H, W) mask
            out = out * attn                            # weighted feature map

        # Classifier
        out = self.gap(out).flatten(1)          # (B, 64)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.fc(out)                    # (B, num_classes)


# Test
model = CustomCNN(use_attention=True)
x = torch.randn(4, 3, 32, 32)
output = model(x)
print(f"Output: {output.shape}")              # (4, 10)

# Verify train/eval behavior
model.train()
out_train = model(x)

model.eval()
with torch.no_grad():
    out_eval = model(x)

print(f"Outputs differ (dropout): {not torch.allclose(out_train, out_eval)}")
```

## Common `F` functions reference

| Function | Module equivalent | Notes |
|---|---|---|
| `F.relu(x)` | `nn.ReLU()` | Prefer functional for activations |
| `F.dropout(x, p, training)` | `nn.Dropout(p)` | Must pass `training=self.training` |
| `F.batch_norm(...)` | `nn.BatchNorm2d` | Module version is easier; handles running stats |
| `F.cross_entropy(logits, y)` | `nn.CrossEntropyLoss()` | Functional in training loops |
| `F.softmax(x, dim)` | `nn.Softmax(dim)` | Functional is the norm |
| `F.scaled_dot_product_attention` | `nn.MultiheadAttention` | No module equiv for raw SDPA |
| `F.max_pool2d(x, k)` | `nn.MaxPool2d(k)` | Either works; module is conventional |
| `F.adaptive_avg_pool2d(x, n)` | `nn.AdaptiveAvgPool2d(n)` | Either works |
| `F.interpolate(x, scale_factor)` | None | Upsampling; functional only |
| `F.grid_sample(x, grid)` | None | Spatial transformer; functional only |

## Interview questions

<details>
<summary>What is the difference between nn.Dropout and F.dropout?</summary>

`nn.Dropout` is a module that stores `p` and automatically uses `self.training` from the module's state (set by `model.train()` / `model.eval()`). `F.dropout` is a stateless function that requires `training` to be passed explicitly. Using `F.dropout(x, p=0.5, training=True)` hardcodes dropout to always be active — a subtle bug that breaks inference. The correct usage is `F.dropout(x, p=0.5, training=self.training)`. When in doubt, use `nn.Dropout` as it handles this automatically.
</details>

<details>
<summary>When must you use an explicit forward method instead of nn.Sequential?</summary>

Use an explicit forward method whenever the computation graph is not a simple linear chain: (1) skip/residual connections — `out = F(x) + x` requires saving the input before the block; (2) multi-head outputs — two branches from the same features; (3) conditional computation — different paths depending on flags or inputs; (4) operations between layers that require intermediate results — attention weights, intermediate losses, auxiliary supervision. `nn.Sequential` only passes the output of each layer as the input to the next — no branching, no skip connections.
</details>

## Final takeaway

Use `nn.Module` for any layer with learnable parameters — they manage parameter registration, device placement, and train/eval switching. Use `F.function` for stateless operations in the forward method — activations, dropout (with `training=self.training`), pooling, loss functions. The explicit `forward` method enables any computation graph shape that `nn.Sequential` cannot express: skip connections, multi-head outputs, conditional blocks, and custom attention patterns.
