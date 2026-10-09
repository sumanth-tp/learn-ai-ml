---
id: cnn-filters-feature-maps
title: "What a CNN sees: filters and feature maps"
sidebar_label: "52 · CNN Filters and Feature Maps"
sidebar_position: 52
slug: /theory/dnn/what-a-cnn-sees-filters-and-feature-maps
description: "How CNN filters are interpreted as learned detectors, what feature maps represent, and how to visualize them — from raw filter weights to activation maximization."
tags: [cnn, filters, feature-maps, visualization, interpretability, deep-learning]
---

# What a CNN sees: filters and feature maps

Training a CNN is not a black box from the perspective of the feature maps. Each filter in each layer is a learned pattern detector. The feature map it produces shows where in the input that pattern was detected and how strongly. By visualizing filters and feature maps, we can understand what a CNN "pays attention to" at each layer — edges in layer 1, textures in layer 2, object parts in layer 3, whole objects in deeper layers.

## One-line definition

A CNN filter is a learned pattern detector. The feature map it produces is a heatmap showing where and how strongly that pattern appears in the input.

![First-layer filters learned by AlexNet on ImageNet — oriented edges, color blobs, and Gabor-like patterns emerge automatically](https://cs231n.stanford.edu/assets/cnn/weights.jpeg)
*Source: [CS231n — Convolutional Neural Networks](https://cs231n.stanford.edu/convolutional-networks/) (Stanford)*

## Filters as pattern detectors

A single $3 \times 3$ filter with weights:

```
[[-1, -1, -1],
 [ 0,  0,  0],
 [ 1,  1,  1]]
```

produces high activation wherever there is a horizontal edge (bright below, dark above). This is a Sobel-like edge detector. When a CNN is trained, it learns exactly these kinds of filters — but optimized for the task, not hand-crafted.

**Layer 1 filters** in a trained CNN typically look like:
- Oriented edge detectors (0°, 45°, 90°, 135°)
- Color blob detectors
- Gabor-like frequency detectors

This is not by design — it is what gradient descent discovers because edges and blobs are the most informative low-level features in natural images.

## Feature map hierarchy

As depth increases, each feature map represents a more abstract concept:

| Layer depth | Typical feature maps |
|---|---|
| Layer 1 | Edges, corners, color boundaries |
| Layer 2 | Textures, simple curves |
| Layer 3 | Object parts (wheels, eyes, leaves) |
| Layer 4+ | Whole objects, semantic concepts |

This was demonstrated by Zeiler & Fergus (2013) who deconvolved ResNet features back to pixel space to visualize what each filter "cares about."

## PyTorch: visualizing filters and feature maps

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Visualize raw filter weights (layer 1 only)
# ============================================================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Layer 1 filters: (64, 3, 7, 7) — 64 RGB filters of size 7×7
filters = model.conv1.weight.data.clone()

# Normalize each filter to [0, 1] for visualization
def normalize_filter(f):
    f_min, f_max = f.min(), f.max()
    return (f - f_min) / (f_max - f_min + 1e-8)

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[0]:
        # filters[i]: (3, 7, 7) → (7, 7, 3) for display
        f = normalize_filter(filters[i]).permute(1, 2, 0).numpy()
        ax.imshow(f)
    ax.axis("off")
plt.suptitle("ResNet-50 Layer 1 Filters (64 × 7×7 RGB)", y=1.01)
plt.tight_layout()
# plt.savefig("resnet_filters.png")


# ============================================================
# 2. Extract feature maps at multiple layers
# ============================================================
class FeatureExtractor(nn.Module):
    """
    Hooks into specified layers and captures their output during forward pass.
    """
    def __init__(self, model: nn.Module, layer_names: list[str]):
        super().__init__()
        self.model = model
        self.hooks = {}
        self.activations = {}

        for name, module in model.named_modules():
            if name in layer_names:
                self.hooks[name] = module.register_forward_hook(
                    lambda mod, inp, out, n=name: self.activations.update({n: out.detach()})
                )

    def forward(self, x):
        return self.model(x)

    def remove_hooks(self):
        for hook in self.hooks.values():
            hook.remove()


# Layers to visualize
layer_names = ["layer1.0.relu", "layer2.0.relu", "layer3.0.relu", "layer4.0.relu"]
extractor = FeatureExtractor(model, layer_names)

# Preprocess an image
preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def visualize_feature_maps(image_path: str, n_channels: int = 16):
    """Visualize feature maps at each layer for a given image."""
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        extractor(tensor)

    for layer_name, activation in extractor.activations.items():
        # activation: (1, C, H, W)
        maps = activation[0, :n_channels]   # first n_channels
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < maps.shape[0]:
                fmap = maps[i].numpy()
                ax.imshow(fmap, cmap="viridis")
            ax.axis("off")
        plt.suptitle(f"Feature maps: {layer_name} "
                     f"(shape: {tuple(activation.shape)})")
        plt.tight_layout()
        # plt.savefig(f"feature_map_{layer_name.replace('.', '_')}.png")


# ============================================================
# 3. Grad-CAM: visualize which regions influenced classification
# ============================================================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights the regions of the input most relevant to the predicted class.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, image_tensor: torch.Tensor,
                class_idx: int = None) -> np.ndarray:
        """Returns a (H, W) Grad-CAM heatmap, normalized to [0, 1]."""
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        output = self.model(image_tensor)   # (1, 1000)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool the gradients: (C,)
        weights = self.gradients[0].mean(dim=(-2, -1))  # (C,)
        # Weighted sum of activations: (H, W)
        cam = (weights[:, None, None] * self.activations[0]).sum(0)
        cam = torch.relu(cam).numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# Usage:
grad_cam = GradCAM(model, target_layer=model.layer4[-1])

# ============================================================
# 4. Activation maximization: what input maximally activates a filter?
# ============================================================
def activation_maximization(model: nn.Module, layer: nn.Module,
                             filter_idx: int, steps: int = 200,
                             lr: float = 0.1) -> np.ndarray:
    """
    Find the input image that maximally activates filter_idx in layer.
    This reveals what visual pattern the filter detects.
    """
    # Start from random noise
    image = torch.randn(1, 3, 224, 224, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=lr)

    activation_holder = {}

    def hook(mod, inp, out):
        activation_holder["out"] = out

    handle = layer.register_forward_hook(hook)

    for step in range(steps):
        optimizer.zero_grad()
        model(image)
        # Maximize mean activation of filter filter_idx
        loss = -activation_holder["out"][0, filter_idx].mean()
        loss.backward()
        optimizer.step()

    handle.remove()

    # Convert to displayable image
    img = image.detach()[0].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img
```

## What visualizations reveal about CNN learning

**Layer 1 (AlexNet/ResNet)**: Gabor-like edge detectors at multiple orientations. These emerge consistently across architectures trained on natural images — they are a universal starting point.

**Middle layers**: texture detectors, curve composites. The feature maps become increasingly coarse spatially (after pooling) but richer in semantic content.

**Late layers**: class-selective neurons. A neuron in ResNet layer 4 might fire strongly for "dog snout" patterns and barely activate for anything else.

**Grad-CAM**: shows that for a "dog" classification, the model focuses on the dog's face and body, not the background — evidence that the model is reasoning about the right features.

## Occlusion sensitivity

Systematically slide a gray square across the image and record the model's confidence drop:

```python
@torch.no_grad()
def occlusion_map(model, image_tensor, target_class, patch_size=32, stride=8):
    """
    Produce a heatmap where high values = important regions.
    Slides an occluding patch and measures confidence drop.
    """
    _, C, H, W = image_tensor.shape
    base_score = model(image_tensor).softmax(1)[0, target_class].item()

    scores = torch.zeros(H, W)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = image_tensor.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.5
            score = model(occluded).softmax(1)[0, target_class].item()
            # High drop = this region was important
            scores[y:y+patch_size, x:x+patch_size] += (base_score - score)

    return scores.numpy()
```

## Interview questions

<details>
<summary>What do layer 1 filters of a CNN typically detect, and why?</summary>

Layer 1 filters in CNNs trained on natural images consistently learn edge detectors at multiple orientations (0°, 45°, 90°, 135°), color blob detectors, and Gabor-like frequency patterns. This is not programmed — it emerges from gradient descent on natural image datasets. Edges and color blobs are the most informative low-level statistics of natural images: the gradient of pixel intensity carries far more information than the raw intensity value. Every competitive CNN architecture, regardless of architecture choices, learns very similar layer 1 filters — suggesting these are the universally optimal features for natural image statistics.
</details>

<details>
<summary>What is Grad-CAM and how does it work?</summary>

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which spatial regions of the input contributed most to a specific class prediction. It works by: (1) running a forward pass and recording the feature maps of a target convolutional layer; (2) computing the gradient of the class score with respect to those feature maps; (3) globally averaging the gradients to get a weight per feature map channel; (4) computing a weighted sum of the feature maps, ReLU'd to keep only positive contributions. The result is a coarse spatial heatmap showing which regions "mattered" for the prediction. Grad-CAM is useful for debugging: a model that highlights background rather than the object is using spurious correlations.
</details>

## Common mistakes

- Visualizing raw filter weights beyond layer 1 — deep layer filters are not interpretable as 2D images because they live in high-dimensional channel space
- Interpreting a feature map as "what the model sees" — it shows where a specific learned pattern was detected, not what the model holistically perceives
- Using Grad-CAM on ReLU-postprocessed activations — apply Grad-CAM to the pre-activation conv output for more accurate gradients

## Final takeaway

CNN filters are learned pattern detectors that form a hierarchy: edges → textures → parts → objects. Feature maps show where each detector fires across the input. Visualization tools — filter plotting, Grad-CAM, activation maximization, occlusion sensitivity — make this hierarchy inspectable, which is both intellectually satisfying and practically useful for debugging models that use the wrong features.
