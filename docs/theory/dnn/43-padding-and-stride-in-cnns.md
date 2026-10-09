---
id: padding-stride-cnns
title: "Padding and stride in CNNs"
sidebar_label: "43 · Padding and Stride"
sidebar_position: 43
slug: /theory/dnn/padding-and-stride-in-cnns
description: "How padding controls whether spatial dimensions shrink after convolution, and how stride controls the rate of downsampling — with the exact output-size formula and PyTorch examples."
tags: [cnn, padding, stride, spatial-resolution, convolution, deep-learning]
---

# Padding and stride in CNNs

Applying a $K \times K$ kernel to an $H \times W$ image without padding shrinks the output to $(H - K + 1) \times (W - K + 1)$. After just a few layers a large image collapses to a tiny feature map. Padding adds extra rows and columns around the input so that spatial resolution is preserved or controlled deliberately. Stride determines how many pixels the kernel steps each time it moves — large strides produce small feature maps quickly and cheaply.

## One-line definition

Padding adds zeros (or other values) around the input to control output size; stride sets the step size of the kernel, trading spatial resolution for computation.

![Stride-1 vs stride-2 convolution — with stride 2 the filter jumps 2 pixels at each step, halving the output spatial dimensions](https://cs231n.stanford.edu/assets/cnn/stride.jpeg)
*Source: [CS231n — Convolutional Neural Networks](https://cs231n.stanford.edu/convolutional-networks/) (Stanford)*

## Output size formula

For a 2D convolution with:
- Input height $H$, width $W$
- Kernel size $K$
- Padding $P$ (on each side)
- Stride $S$

$$
H_{\text{out}} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1
$$

$$
W_{\text{out}} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1
$$

## Padding modes

### Valid padding (no padding, $P = 0$)

The kernel only slides over positions where it fits completely inside the input:

$$
H_{\text{out}} = H - K + 1
$$

For a $32 \times 32$ input with a $3 \times 3$ kernel: output is $30 \times 30$. Each layer loses 2 pixels per side. After 15 layers the feature map is $2 \times 2$ — spatial information is destroyed.

**When to use**: when you intentionally want to reduce spatial size, or in the final classification layers.

### Same padding ($P = \lfloor K/2 \rfloor$)

Padding is chosen so the output has the same spatial dimensions as the input (when $S = 1$):

For $K = 3$: $P = 1$. For $K = 5$: $P = 2$.

$$
H_{\text{out}} = H \quad \text{(when } S = 1\text{)}
$$

**When to use**: the standard choice for hidden convolutional layers. Preserves spatial size, allowing you to stack many layers without explicit size calculations.

### Full padding ($P = K - 1$)

Output is larger than the input: $H_{\text{out}} = H + K - 1$. Used in transposed convolution / deconvolution layers (e.g., in autoencoders and GANs to upsample).

## Stride

Stride $S > 1$ makes the kernel skip $S - 1$ positions between each application:

| Input $H$ | Kernel $K$ | Padding $P$ | Stride $S$ | Output $H_{\text{out}}$ |
|---|---|---|---|---|
| 32 | 3 | 1 | 1 | 32 |
| 32 | 3 | 0 | 1 | 30 |
| 32 | 3 | 1 | 2 | 16 |
| 32 | 3 | 0 | 2 | 15 |
| 224 | 7 | 3 | 2 | 112 (ResNet stem) |

**Stride vs pooling**: both reduce spatial resolution. Strided convolution learns how to downsample; max pooling uses a fixed rule. Modern architectures (ResNet, EfficientNet) often replace pooling with strided convolutions so downsampling is learnable.

## Receptive field

The **receptive field** of a neuron is the region of the original input it can "see." Each layer of convolution expands the receptive field:

- 1 conv layer, $K = 3$: receptive field = $3 \times 3$
- 2 conv layers, $K = 3$: receptive field = $5 \times 5$
- 3 conv layers, $K = 3$: receptive field = $7 \times 7$

Stacking three $3 \times 3$ layers has the same receptive field as one $7 \times 7$ layer but uses fewer parameters ($3 \times 3^2 = 27$ vs $7^2 = 49$ per channel) and introduces more nonlinearities. This is why modern CNNs stack small kernels instead of using large ones.

## Dilated (atrous) convolution

Dilation $d$ inserts $d - 1$ zeros between kernel elements, expanding the receptive field without increasing parameters or losing resolution:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2P - d(K - 1) - 1}{S} \right\rfloor + 1
$$

A $3 \times 3$ kernel with dilation $d = 2$ has the same receptive field as a $5 \times 5$ kernel. Used in semantic segmentation (DeepLab) and audio modeling (WaveNet).

## PyTorch code

```python
import torch
import torch.nn as nn

x = torch.randn(4, 3, 32, 32)   # batch=4, channels=3, H=W=32


# ============================================================
# Valid padding: output shrinks
# ============================================================
conv_valid = nn.Conv2d(3, 16, kernel_size=3, padding=0)  # P=0, S=1
out_valid = conv_valid(x)
print(f"valid: {x.shape} → {out_valid.shape}")   # (4,16,30,30)


# ============================================================
# Same padding: output keeps spatial size
# ============================================================
conv_same = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # P=1, S=1
out_same = conv_same(x)
print(f"same:  {x.shape} → {out_same.shape}")    # (4,16,32,32)


# ============================================================
# Strided convolution: halves spatial size (like MaxPool2d(2))
# ============================================================
conv_stride = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
out_stride = conv_stride(x)
print(f"stride=2: {x.shape} → {out_stride.shape}")  # (4,16,16,16)


# ============================================================
# Dilated convolution: expands receptive field
# ============================================================
conv_dilated = nn.Conv2d(3, 16, kernel_size=3, padding=2, dilation=2)
out_dilated = conv_dilated(x)
print(f"dilation=2: {x.shape} → {out_dilated.shape}")  # (4,16,32,32)


# ============================================================
# Output size formula verification
# ============================================================
def output_size(H, K, P, S):
    return (H + 2 * P - K) // S + 1

print("\nFormula verification:")
print(f"H=32, K=3, P=0, S=1: {output_size(32, 3, 0, 1)}")   # 30
print(f"H=32, K=3, P=1, S=1: {output_size(32, 3, 1, 1)}")   # 32
print(f"H=32, K=3, P=1, S=2: {output_size(32, 3, 1, 2)}")   # 16
print(f"H=224, K=7, P=3, S=2: {output_size(224, 7, 3, 2)}")  # 112 (ResNet stem)


# ============================================================
# Typical CNN block: same padding throughout, downsample with stride
# ============================================================
class ConvBlock(nn.Module):
    """
    Standard conv block: conv → BN → ReLU with same padding.
    Use stride=2 to halve spatial resolution.
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


model = nn.Sequential(
    ConvBlock(3, 32),           # 32×32 → 32×32
    ConvBlock(32, 64, stride=2),  # 32×32 → 16×16  (strided, no pooling)
    ConvBlock(64, 128, stride=2), # 16×16 →  8×8
    nn.AdaptiveAvgPool2d(1),      #  8×8  →  1×1
    nn.Flatten(),
    nn.Linear(128, 10),
)

out = model(x)
print(f"\nfull model output: {out.shape}")   # (4, 10)
```

## Padding for odd vs even kernels

Odd kernel sizes ($K = 1, 3, 5, 7$) allow symmetric same padding: $P = K // 2$. Even kernel sizes ($K = 2, 4$) cannot be perfectly centered, producing asymmetric padding (one extra pixel on one side). This is why nearly all practical CNNs use odd kernels.

## Interview questions

<details>
<summary>What is the output size of a convolution with input 64×64, kernel 5×5, padding 2, stride 1?</summary>

Using the formula: $(64 + 2 \times 2 - 5) / 1 + 1 = 63/1 + 1 = 64$. The output is $64 \times 64$ — this is same padding for a $5 \times 5$ kernel.
</details>

<details>
<summary>What is the difference between stride=2 and MaxPool2d(2) for downsampling?</summary>

Both halve spatial resolution. MaxPool2d takes the maximum value in each 2×2 window using a fixed rule — no parameters, no learning. Strided convolution learns a $K \times K$ filter and the learned weights decide how to downsample. Strided convolution is therefore more expressive but adds parameters; max pooling is parameter-free and translation-invariant. Modern architectures increasingly use strided convolution because learned downsampling can preserve more task-relevant features.
</details>

<details>
<summary>Why does same padding use P = K // 2 specifically?</summary>

For a kernel of size $K$ sliding with stride 1, the output shrinks by $K - 1$ total ($(K-1)/2$ on each side). Adding $P = \lfloor K/2 \rfloor$ zeros on each side adds back exactly that many rows and columns, leaving $H_{\text{out}} = H$. For odd $K$ (the standard case), $\lfloor K/2 \rfloor = (K-1)/2$, so the padding is perfectly symmetric and the center of the kernel aligns exactly with each input position.
</details>

## Common mistakes

- Forgetting that same padding only preserves size when $S = 1$ — with $S = 2$ the output is halved even with same padding
- Using even kernel sizes and then hitting asymmetric padding issues; prefer $K = 3, 5, 7$
- Computing output size by hand for deep networks — parameterize with the formula or use a forward-pass probe
- Confusing padding=1 in PyTorch (adds 1 pixel on each side) with total padding of 2

## Final takeaway

Same padding ($P = K // 2$, $S = 1$) is the default for hidden layers — it preserves spatial size and lets you reason about the network independently of input dimensions. Valid padding ($P = 0$) and strided convolution ($S = 2$) are the two standard ways to intentionally reduce spatial resolution. The output size formula $\lfloor (H + 2P - K) / S \rfloor + 1$ governs every convolutional layer.
