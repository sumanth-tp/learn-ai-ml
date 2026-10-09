---
id: backprop-cnns-1
title: "Backpropagation in CNNs part 1"
sidebar_label: "47 · Backprop in CNNs I"
sidebar_position: 47
slug: /theory/dnn/backpropagation-in-cnns-part-1
description: "How gradients flow through a convolutional layer: the gradient with respect to the input, the gradient with respect to the filter weights, and why weight sharing creates gradient accumulation."
tags: [cnn, backpropagation, gradients, convolution, weight-sharing, deep-learning]
---

# Backpropagation in CNNs part 1

Backpropagation in a CNN is the same chain rule as in any neural network — the difference is the computational structure. In a dense layer, each output connects to every input, so every input gets one gradient contribution. In a convolution, each input pixel contributes to multiple output positions (because the filter overlaps as it slides), and each filter weight is shared across all positions. Both effects change how gradients accumulate.

## One-line definition

Backpropagation through a conv layer requires computing: (1) the gradient of the loss with respect to the input (to continue backprop to earlier layers) and (2) the gradient with respect to the filter weights (to update the filters).

![Learned first-layer filters in AlexNet — the gradients during training tune these from random noise into edge and color detectors](https://cs231n.stanford.edu/assets/cnn/weights.jpeg)
*Source: [CS231n — Convolutional Neural Networks](https://cs231n.stanford.edu/convolutional-networks/) (Stanford) — Krizhevsky et al., 2012*

## The forward pass (recap)

For a 1D convolution (for clarity), a filter $w$ of size $K$ produces output $y$ from input $x$:

$$
y[i] = \sum_{k=0}^{K-1} w[k] \cdot x[i + k], \quad i = 0, 1, \ldots, N - K
$$

In 2D, each output position is:

$$
y[i, j] = \sum_{r=0}^{K-1} \sum_{c=0}^{K-1} w[r, c] \cdot x[i+r,\ j+c]
$$

## Gradient with respect to the filter weights

We need $\frac{\partial \mathcal{L}}{\partial w[r, c]}$ to update the filter.

By the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial w[r, c]} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial y[i, j]} \cdot \frac{\partial y[i, j]}{\partial w[r, c]}
$$

Since $y[i, j] = \sum_{r', c'} w[r', c'] \cdot x[i+r', j+c']$:

$$
\frac{\partial y[i, j]}{\partial w[r, c]} = x[i+r,\ j+c]
$$

Therefore:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial w[r, c]} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial y[i, j]} \cdot x[i+r,\ j+c]}
$$

**This is a convolution.** The gradient of the loss with respect to a filter weight is the convolution of the upstream gradient (the loss gradient flowing back) with the input. This is why backprop through a conv layer is itself a convolution.

**Weight sharing effect**: every output position $y[i, j]$ depends on the same filter weight $w[r, c]$. All their gradient contributions add up. The gradient is a sum over all output positions — this is gradient accumulation due to weight sharing.

## Gradient with respect to the input

We need $\frac{\partial \mathcal{L}}{\partial x[p, q]}$ to continue backpropagating to earlier layers.

Each input position $x[p, q]$ contributes to all output positions where the filter covers $(p, q)$. For a $K \times K$ filter, that is all $(i, j)$ such that $0 \le p - i \le K-1$ and $0 \le q - j \le K-1$:

$$
\frac{\partial \mathcal{L}}{\partial x[p, q]} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial y[i, j]} \cdot w[p-i,\ q-j]
$$

**This is a full (padded) convolution with the flipped filter.** Computing the gradient with respect to the input is a convolution of the upstream gradient with the filter rotated 180°. In PyTorch this is computed as a transposed convolution.

## Summary: two gradient convolutions

| Gradient | Operation | Used for |
|---|---|---|
| $\partial \mathcal{L} / \partial w$ | Convolve upstream gradient with input | Updating filter weights |
| $\partial \mathcal{L} / \partial x$ | Convolve upstream gradient with flipped filter | Propagating gradient to previous layer |

Both operations are convolutions — GPUs can run backprop through a conv layer using the same CUDA kernels as the forward pass.

## Numerical example (1D)

Forward pass:
```
input x = [1, 2, 3, 4, 5]
filter w = [1, 0, -1]     (edge detector)
output y = [1*1 + 2*0 + 3*(-1),   2*1 + 3*0 + 4*(-1),   3*1 + 4*0 + 5*(-1)]
         = [1 - 3, 2 - 4, 3 - 5]
         = [-2, -2, -2]
```

Suppose upstream gradient $\frac{\partial \mathcal{L}}{\partial y} = [1, 1, 1]$ (all ones for simplicity).

**Gradient w.r.t. filter weights:**
```
dL/dw[0] = sum over i of: (dL/dy[i]) * x[i+0] = 1*1 + 1*2 + 1*3 = 6
dL/dw[1] = sum over i of: (dL/dy[i]) * x[i+1] = 1*2 + 1*3 + 1*4 = 9
dL/dw[2] = sum over i of: (dL/dy[i]) * x[i+2] = 1*3 + 1*4 + 1*5 = 12
```

So $\partial \mathcal{L} / \partial w = [6, 9, 12]$ — this is the cross-correlation of `dL/dy` with `x`.

**Gradient w.r.t. input:**
```
dL/dx[0] = dL/dy[0] * w[0]                   = 1*1           =  1
dL/dx[1] = dL/dy[0] * w[1] + dL/dy[1] * w[0] = 1*0 + 1*1     =  1
dL/dx[2] = dL/dy[0] * w[2] + dL/dy[1] * w[1] + dL/dy[2] * w[0]
         = 1*(-1) + 1*0 + 1*1                               =  0
dL/dx[3] = dL/dy[1] * w[2] + dL/dy[2] * w[1]
         = 1*(-1) + 1*0                                     = -1
dL/dx[4] = dL/dy[2] * w[2]                   = 1*(-1)       = -1
```

So $\partial \mathcal{L} / \partial x = [1, 1, 0, -1, -1]$.

## PyTorch verification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Verify backprop through a 1D conv using autograd
# ============================================================
x = torch.tensor([[1., 2., 3., 4., 5.]]).unsqueeze(0)  # (1, 1, 5)
w = torch.tensor([[[1., 0., -1.]]])                     # (1, 1, 3) filter

x.requires_grad_(True)
w_param = nn.Parameter(w.clone())

# Forward
y = F.conv1d(x, w_param)   # output: (1, 1, 3) = [[-2, -2, -2]]
print(f"Output y: {y}")

# Backward with upstream gradient = all ones
loss = y.sum()
loss.backward()

print(f"dL/dx:   {x.grad}")        # should be [1, 1, 0, -1, -1]
print(f"dL/dw:   {w_param.grad}")  # should be [6, 9, 12]


# ============================================================
# 2D conv backprop — same mechanics, two spatial dimensions
# ============================================================
x2d = torch.randn(1, 1, 6, 6, requires_grad=True)
conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

output = conv(x2d)
output.sum().backward()

print(f"\n2D conv input shape:    {x2d.shape}")
print(f"2D conv output shape:   {output.shape}")
print(f"2D gradient input shape:{x2d.grad.shape}")    # same as input
print(f"2D gradient weight shape:{conv.weight.grad.shape}")  # same as filter


# ============================================================
# Multi-filter gradient accumulation
# ============================================================
# With C_out filters, each filter gets its own gradient
conv_multi = nn.Conv2d(3, 8, kernel_size=3, padding=1)
x_rgb = torch.randn(4, 3, 32, 32, requires_grad=True)

out = conv_multi(x_rgb)
out.sum().backward()

print(f"\nMulti-filter weight grad: {conv_multi.weight.grad.shape}")
# (8, 3, 3, 3) — 8 filters, each 3×3×3
# The gradient for each filter is summed over the batch dimension (4 images)
# and over all output positions where that filter contributed
```

## The weight sharing gradient is a sum, not a per-position gradient

In a dense layer: $\partial \mathcal{L} / \partial w_{ij}$ is the gradient from a single connection between input $j$ and output $i$.

In a conv layer: $\partial \mathcal{L} / \partial w[r, c]$ is the **sum of gradients from all positions** where this weight participated — typically hundreds or thousands of positions per image, times the batch size. This larger gradient sum is why batch size matters more for conv layers than for dense layers.

## Interview questions

<details>
<summary>Why is the gradient with respect to the filter weights itself a convolution?</summary>

In the forward pass, $y[i, j] = \sum_{r,c} w[r,c] \cdot x[i+r, j+c]$. To find $\partial \mathcal{L} / \partial w[r, c]$, we sum over all output positions: $\sum_{i,j} (\partial \mathcal{L} / \partial y[i, j]) \cdot x[i+r, j+c]$. This sum over $(i, j)$ with a spatial offset $(r, c)$ is exactly the definition of a cross-correlation between the upstream gradient and the input. So gradient computation through a conv layer uses the same convolution primitive as the forward pass — just with different operands.
</details>

<details>
<summary>Why does weight sharing cause the filter gradient to be a sum rather than a single value?</summary>

The filter weight $w[r, c]$ is used at every output position $(i, j)$ in the forward pass — it participates in $H_{\text{out}} \times W_{\text{out}}$ multiplications. The chain rule requires summing all gradient contributions from all positions where this weight appeared: $\partial \mathcal{L} / \partial w[r, c] = \sum_{i,j} (\partial \mathcal{L} / \partial y[i,j]) \cdot x[i+r, j+c]$. For a $224 \times 224$ image with a $3 \times 3$ filter, each filter weight collects gradients from roughly $224 \times 224 \approx 50{,}000$ output positions per image.
</details>

<details>
<summary>What is the gradient of the loss with respect to the CNN input, and why is it needed?</summary>

The gradient $\partial \mathcal{L} / \partial x[p, q]$ is the signal that tells the previous layer how much each input pixel contributed to the loss. This allows the chain rule to continue backward through the network — without this gradient, layers before the current conv layer could not update their parameters. Mathematically, this gradient is the convolution of the upstream gradient with the filter rotated 180° (a full cross-correlation). PyTorch computes this automatically via transposed convolution in the backward pass.
</details>

## Common mistakes

- Thinking the gradient w.r.t. the filter is computed at a single position — it is the accumulated sum over all output positions in all batch items
- Forgetting that the bias gradient is simply the sum of the upstream gradient over all spatial positions and batch items: $\partial \mathcal{L} / \partial b = \sum_{batch, i, j} \partial \mathcal{L} / \partial y[batch, i, j]$
- Confusing the gradient w.r.t. weights (for updating filters) with the gradient w.r.t. input (for backpropagating to earlier layers) — both are computed in the backward pass but used differently

## Final takeaway

Backpropagation through a conv layer requires two gradient computations: the filter weight gradient (a cross-correlation between upstream gradient and the input) and the input gradient (a cross-correlation between upstream gradient and the flipped filter). Weight sharing means the filter gradient is a sum over all positions where that filter was applied — a key consequence of the shared parameterization that makes convolution parameter-efficient.
