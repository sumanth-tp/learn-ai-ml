---
title: Deep Learning Interview Questions
sidebar_position: 10
sidebar_label: "10 · Deep learning foundations"
---

# Deep Learning Interview Questions

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. After revising these foundations, continue with the [practical topic bank](11-deep-learning-llms.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](23-tools-versions.md) for current behaviour and [coding labs](21-coding-labs.md) for runnable implementations.
:::

100 essential deep learning interview questions with in-depth answers and code examples.

---

<details>
<summary><strong>1. What is a neural network and how does it work?</strong></summary>

**Answer:**
A neural network is a computational model inspired by the brain, consisting of layers of interconnected nodes (neurons). Each neuron computes a weighted sum of inputs, adds a bias, then applies a non-linear activation function.

```python
import numpy as np

# Single neuron
def neuron(inputs, weights, bias, activation="relu"):
    z = np.dot(weights, inputs) + bias
    if activation == "relu":
        return max(0, z)
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    return z

inputs = np.array([0.5, 0.3, 0.8])
weights = np.array([0.4, -0.2, 0.7])
bias = 0.1
output = neuron(inputs, weights, bias, activation="relu")
print(f"Neuron output: {output:.4f}")

# Simple 2-layer network from scratch
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def relu(self, z): return np.maximum(0, z)
    def sigmoid(self, z): return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

net = TwoLayerNet(3, 4, 1)
X = np.random.randn(5, 3)
out = net.forward(X)
print(f"Network output shape: {out.shape}, values: {out.ravel().round(4)}")
```

**Interview Tip:** Emphasize depth (multiple layers) enables learning hierarchical representations. More layers = more abstract features. Universal Approximation Theorem: a single hidden layer with enough neurons can approximate any continuous function.

</details>

<details>
<summary><strong>2. What is backpropagation?</strong></summary>

**Answer:**
Backpropagation computes gradients of the loss with respect to all parameters using the chain rule of calculus, propagating errors backward from output to input. Enables efficient gradient computation in O(n) instead of O(n²).

```python
import numpy as np

# Backprop from scratch for a simple network
np.random.seed(42)

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# Initialize
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(a): return a * (1 - a)

lr = 0.5
for epoch in range(5000):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Loss (binary cross-entropy)
    loss = -np.mean(y * np.log(a2 + 1e-8) + (1-y) * np.log(1-a2 + 1e-8))

    # Backward pass (chain rule)
    dL_da2 = -(y / (a2 + 1e-8) - (1-y) / (1-a2 + 1e-8))
    da2_dz2 = sigmoid_deriv(a2)
    dz2 = dL_da2 * da2_dz2                # dL/dz2

    dW2 = a1.T @ dz2 / len(X)
    db2 = dz2.mean(axis=0, keepdims=True)
    da1 = dz2 @ W2.T

    dz1 = da1 * sigmoid_deriv(a1)         # dL/dz1
    dW1 = X.T @ dz1 / len(X)
    db1 = dz1.mean(axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")

print(f"Predictions: {a2.ravel().round(2)}")  # Should be ~[0, 1, 1, 0]
```

**Interview Tip:** Backprop is just the chain rule applied efficiently via dynamic programming. Forward pass stores intermediate activations; backward pass uses them to compute gradients. Vanishing/exploding gradients are the main challenges.

</details>

<details>
<summary><strong>3. What are activation functions? Compare ReLU, sigmoid, and tanh.</strong></summary>

**Answer:**
Activation functions introduce non-linearity. Without them, deep networks collapse to linear transformations.

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)
def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)
def elu(z, alpha=1.0): return np.where(z > 0, z, alpha * (np.exp(z) - 1))
def swish(z): return z * sigmoid(z)
def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

z = np.array([-3, -1, 0, 1, 3], dtype=float)

print("z:          ", z)
print("sigmoid:    ", sigmoid(z).round(3))    # (0,1), saturates
print("tanh:       ", tanh(z).round(3))       # (-1,1), saturates
print("relu:       ", relu(z).round(3))       # [0,inf), dying neurons
print("leaky_relu: ", leaky_relu(z).round(3)) # allows negative gradients
print("swish:      ", swish(z).round(3))      # smooth, non-monotonic
print("gelu:       ", gelu(z).round(3))       # used in BERT, GPT

# Derivatives (for backprop)
def sigmoid_grad(a): return a * (1 - a)   # where a = sigmoid(z)
def tanh_grad(a): return 1 - a**2
def relu_grad(z): return (z > 0).astype(float)  # 0 or 1
```

| Function | Range | Pros | Cons |
|----------|-------|------|------|
| Sigmoid | (0,1) | Output as prob | Vanishing gradient, not zero-centered |
| Tanh | (-1,1) | Zero-centered | Vanishing gradient |
| ReLU | [0,∞) | Fast, no vanishing | Dying ReLU |
| Leaky ReLU | (-∞,∞) | No dying | Extra hyperparameter |
| GELU | ~(-0.17,∞) | Best for transformers | Expensive |

**Interview Tip:** ReLU default for CNNs. GELU for transformers. Sigmoid/tanh for output layers (binary/tanh output). Dying ReLU: neurons outputting 0 always — fixed by Leaky ReLU or careful initialization.

</details>

<details>
<summary><strong>4. What is vanishing and exploding gradients?</strong></summary>

**Answer:**
In deep networks, gradients are multiplied through many layers. If weights are small, gradients shrink exponentially (vanishing). If large, they grow exponentially (exploding). Both prevent effective training.

```python
import numpy as np

# Demonstrate vanishing gradient with sigmoid
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_grad(a): return a * (1 - a)

# For sigmoid: max derivative is 0.25 (at z=0)
# After n layers, gradient <= 0.25^n
depths = [5, 10, 20, 50]
for n in depths:
    vanishing_factor = 0.25 ** n
    print(f"Depth {n:2d}: gradient factor = {vanishing_factor:.2e}")

# With ReLU: gradient is 1 for positive activations
# After n layers with ReLU: gradient stays ~1 (no vanishing)
for n in depths:
    relu_factor = 1.0 ** n
    print(f"Depth {n:2d}: ReLU gradient = {relu_factor:.2f}")

# Solutions:
print("\nSolutions:")
print("1. ReLU and variants (avoid saturating activations)")
print("2. Residual connections (skip connections)")
print("3. Batch Normalization")
print("4. Careful initialization (Xavier, He)")
print("5. Gradient clipping (for exploding)")
print("6. LSTM/GRU gates (for RNNs)")

# Gradient clipping (prevents explosion)
def clip_gradients(gradients, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        return [g * clip_coef for g in gradients]
    return gradients

grads = [np.array([100.0, 200.0]), np.array([50.0])]
clipped = clip_gradients(grads, max_norm=1.0)
print(f"\nOriginal grad norm: {np.sqrt(sum(np.sum(g**2) for g in grads)):.1f}")
print(f"Clipped grad norm:  {np.sqrt(sum(np.sum(g**2) for g in clipped)):.3f}")
```

**Interview Tip:** LSTM gates solve vanishing gradients for RNNs by maintaining a separate cell state. ResNets solve it for deep CNNs with skip connections. Gradient clipping (by norm, not value) is standard in NLP training.

</details>

<details>
<summary><strong>5. What is batch normalization and why does it help?</strong></summary>

**Answer:**
Batch normalization normalizes layer inputs to zero mean and unit variance, then scales and shifts with learned parameters. Stabilizes training, enables higher learning rates, acts as regularizer.

```python
import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)   # learnable scale
        self.beta = np.zeros(num_features)   # learnable shift
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, X, training=True):
        if training:
            mean = X.mean(axis=0)
            var = X.var(axis=0)
            # Update running statistics (for inference)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        X_norm = (X - mean) / np.sqrt(var + self.eps)
        return self.gamma * X_norm + self.beta

# Demonstrate normalization
np.random.seed(42)
X = np.random.randn(32, 64) * 5 + 10  # batch of 32, 64 features

bn = BatchNorm(64)
X_bn = bn.forward(X, training=True)
print(f"Before BN: mean={X.mean():.2f}, std={X.std():.2f}")
print(f"After BN:  mean={X_bn.mean():.4f}, std={X_bn.std():.4f}")

# PyTorch usage
"""
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),       # after linear, before activation
    nn.ReLU(),
    nn.Linear(64, 10)
)
"""
```

**Interview Tip:** Place BatchNorm between linear layer and activation. For CNNs, use BatchNorm2d. Layer Norm is used in Transformers (normalizes over feature dimension per sample). Instance Norm for style transfer.

</details>

<details>
<summary><strong>6. What is dropout and how does it work?</strong></summary>

**Answer:**
Dropout randomly sets a fraction p of neuron outputs to zero during training. Forces the network to learn redundant representations. At inference, all neurons are active but scaled by (1-p). Acts as regularization.

```python
import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p  # probability of DROPPING (zeroing out)
        self.mask = None

    def forward(self, X, training=True):
        if not training:
            return X  # no dropout at test time

        # Inverted dropout: scale during training so no scaling needed at test time
        self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask  # same mask used in forward

np.random.seed(42)
X = np.ones((4, 8))  # batch=4, features=8

dropout = Dropout(p=0.5)

train_out = dropout.forward(X, training=True)
test_out = dropout.forward(X, training=False)

print(f"Training output (some zeros):\n{train_out}")
print(f"Test output (all ones):\n{test_out}")
print(f"Training mean: {train_out.mean():.2f} (close to 1.0 due to inverted scaling)")

# Dropout in PyTorch
"""
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # only active in model.train() mode
        return self.fc2(x)
"""
```

**Interview Tip:** Dropout is not applied during evaluation — use `model.eval()` in PyTorch. Higher p = more regularization. Spatial dropout for CNNs (drops entire feature maps). Dropout after fully-connected layers; less common after conv layers.

</details>

<details>
<summary><strong>7. What are optimizers? SGD vs Adam vs RMSprop.</strong></summary>

**Answer:**
Optimizers update model parameters using gradients. SGD: simple, needs tuning. RMSprop: adapts learning rate per-parameter. Adam: combines momentum + adaptive learning rate, most popular default.

```python
import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
            params[key] += self.velocity[key]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # first moment (momentum)
        self.v = {}  # second moment (adaptive lr)
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

            m_hat = self.m[key] / (1 - self.beta1**self.t)  # bias correction
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Compare on simple loss landscape
def loss_fn(w): return w**2 + 5*np.sin(w)  # non-convex
def grad_fn(w): return 2*w + 5*np.cos(w)

for name, opt_class, lr in [("SGD", SGD, {"lr": 0.1}), ("Adam", Adam, {"lr": 0.1})]:
    w = {"w": np.array([5.0])}
    opt = opt_class(**lr)
    for i in range(100):
        g = {"w": grad_fn(w["w"])}
        opt.update(w, g)
    print(f"{name}: final w={w['w'][0]:.4f}, loss={loss_fn(w['w'][0]):.4f}")
```

**Interview Tip:** Adam is default for most tasks. SGD with momentum often achieves better generalization in practice (especially for CNNs). AdamW (Adam + weight decay decoupled) is preferred over Adam for transformers. Learning rate warmup is essential for large models.

</details>

<details>
<summary><strong>8. What are convolutional neural networks (CNNs)?</strong></summary>

**Answer:**
CNNs use convolutional layers to learn local spatial features via shared filter weights. Key components: convolution (feature extraction), pooling (spatial reduction), and fully-connected (classification).

```python
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """2D convolution from scratch"""
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)))

    kh, kw = kernel.shape
    ih, iw = image.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1

    output = np.zeros((oh, ow))
    for i in range(0, oh):
        for j in range(0, ow):
            output[i, j] = np.sum(image[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)
    return output

# Example: edge detection kernel
image = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
], dtype=float)

# Sobel vertical edge detector
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
edges = conv2d(image, kernel, padding=1)
print("Edge detected output:")
print(edges.round(1))

# PyTorch CNN
"""
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))
"""
```

**Interview Tip:** Parameter sharing is key — one filter learns one feature across entire image. Stride controls spatial resolution reduction. Valid padding = no padding, Same padding = output size = input size. MaxPool reduces spatial size, increases receptive field.

</details>

<details>
<summary><strong>9. What are recurrent neural networks (RNNs) and LSTMs?</strong></summary>

**Answer:**
RNNs process sequences by maintaining hidden state across time steps. Standard RNNs suffer from vanishing gradients. LSTMs use gates (forget, input, output) to maintain long-term dependencies via a separate cell state.

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.Wx = np.random.randn(input_size, hidden_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b = np.zeros(hidden_size)

    def forward(self, X, h0=None):
        T, input_size = X.shape
        hidden_size = self.Wh.shape[0]
        h = h0 if h0 is not None else np.zeros(hidden_size)
        hiddens = []

        for t in range(T):
            h = np.tanh(X[t] @ self.Wx + h @ self.Wh + self.b)
            hiddens.append(h.copy())

        return np.array(hiddens), h

# LSTM equations:
def lstm_cell(x, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    concat = np.concatenate([x, h_prev])

    f = 1 / (1 + np.exp(-(concat @ Wf + bf)))  # forget gate
    i = 1 / (1 + np.exp(-(concat @ Wi + bi)))  # input gate
    c_tilde = np.tanh(concat @ Wc + bc)         # candidate cell
    o = 1 / (1 + np.exp(-(concat @ Wo + bo)))  # output gate

    c = f * c_prev + i * c_tilde                # cell state
    h = o * np.tanh(c)                          # hidden state
    return h, c

# Demo
rnn = SimpleRNN(input_size=5, hidden_size=10)
sequence = np.random.randn(20, 5)  # 20 time steps, 5 features
hiddens, final_h = rnn.forward(sequence)
print(f"Hidden states shape: {hiddens.shape}")  # (20, 10)
print(f"Final hidden: {final_h.round(3)}")

# PyTorch usage
"""
import torch.nn as nn

lstm = nn.LSTM(input_size=5, hidden_size=10, num_layers=2,
               batch_first=True, dropout=0.3, bidirectional=True)
"""
```

**Interview Tip:** LSTM: forget gate decides what to remove from cell state, input gate what to add, output gate controls what to pass as hidden state. GRU is simpler (2 gates), nearly as effective. Bidirectional RNNs use both past and future context.

</details>

<details>
<summary><strong>10. What is the attention mechanism?</strong></summary>

**Answer:**
Attention computes a weighted sum of values, where weights reflect the relevance of each position to the query. Enables models to focus on relevant parts of input, overcoming the bottleneck of fixed-size RNN encodings.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: queries  (batch, heads, seq_q, d_k)
    K: keys     (batch, heads, seq_k, d_k)
    V: values   (batch, heads, seq_v, d_v)
    """
    d_k = Q.shape[-1]
    # Scores: how much each query attends to each key
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (b, h, sq, sk)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # mask out future tokens

    # Softmax to get attention weights
    scores -= scores.max(axis=-1, keepdims=True)  # numerical stability
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    # Weighted sum of values
    output = weights @ V  # (b, h, sq, d_v)
    return output, weights

# Single-head example
np.random.seed(42)
batch, seq_len, d_k = 2, 5, 8

Q = np.random.randn(batch, 1, seq_len, d_k)
K = np.random.randn(batch, 1, seq_len, d_k)
V = np.random.randn(batch, 1, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")  # (2, 1, 5, 8)
print(f"Attention weights (sum=1): {attn_weights[0, 0, 0].round(3)}")
print(f"Sum of weights: {attn_weights[0, 0, 0].sum():.4f}")

# Self-attention: Q=K=V (all from same sequence)
X = np.random.randn(batch, 1, seq_len, d_k)
self_out, self_weights = scaled_dot_product_attention(X, X, X)
print(f"Self-attention output: {self_out.shape}")
```

**Interview Tip:** "Queries look at keys to get weights, then use weights to sum values." Self-attention: Q, K, V all from same sequence. Cross-attention: Q from decoder, K,V from encoder. Scaled by sqrt(d_k) to prevent large dot products that push softmax into saturation.

</details>

<details>
<summary><strong>11. What is the Transformer architecture?</strong></summary>

**Answer:**
Transformers replace RNNs with multi-head self-attention + feed-forward layers + residual connections + layer normalization. Enables parallelization and captures long-range dependencies efficiently.

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Projection weights
        self.Wq = np.random.randn(d_model, d_model) * 0.1
        self.Wk = np.random.randn(d_model, d_model) * 0.1
        self.Wv = np.random.randn(d_model, d_model) * 0.1
        self.Wo = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    def forward(self, Q_input, K_input, V_input):
        batch = Q_input.shape[0]
        Q = self.split_heads(Q_input @ self.Wq, batch)
        K = self.split_heads(K_input @ self.Wk, batch)
        V = self.split_heads(V_input @ self.Wv, batch)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        weights = softmax(scores, axis=-1)
        out = weights @ V  # (batch, heads, seq, d_k)

        out = out.transpose(0, 2, 1, 3).reshape(batch, -1, self.d_model)
        return out @ self.Wo

# Transformer block structure
"""
TransformerLayer:
  1. Multi-Head Self-Attention(x) -> attn_out
  2. Add & Norm: x = LayerNorm(x + attn_out)
  3. Feed-Forward: ff_out = FFN(x)  [Linear -> ReLU/GELU -> Linear]
  4. Add & Norm: x = LayerNorm(x + ff_out)

Full Transformer:
  Encoder: N x TransformerLayer
  Decoder: N x (Masked Self-Attn + Cross-Attn + FFN)
"""

batch, seq, d_model, heads = 2, 10, 64, 8
mha = MultiHeadAttention(d_model, heads)
X = np.random.randn(batch, seq, d_model)
out = mha.forward(X, X, X)
print(f"MHA output: {out.shape}")  # (2, 10, 64)
```

**Interview Tip:** Key innovations: (1) Attention captures all pairwise token relationships, (2) Positional encoding adds sequence order information, (3) Residual connections + LayerNorm enable training very deep models. O(n²) attention complexity is the main limitation for long sequences.

</details>

<details>
<summary><strong>12. What is BERT and how does it work?</strong></summary>

**Answer:**
BERT (Bidirectional Encoder Representations from Transformers) pre-trains a transformer encoder bidirectionally on two tasks: Masked Language Modeling (predict masked tokens) and Next Sentence Prediction.

```python
# pip install transformers
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Tokenize input
text = "Deep learning is fascinating."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
print("Token IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token = sentence embedding

print(f"Last hidden shape: {last_hidden.shape}")
print(f"[CLS] embedding shape: {cls_embedding.shape}")

# Fine-tuning for classification
clf_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# BERT tokenization specifics
tokens = tokenizer.tokenize("Hello, world!")
print(f"Tokens: {tokens}")  # ['hello', ',', 'world', '!']
# [CLS] at start, [SEP] at end, [MASK] for MLM

# Sentence pair encoding
pair = tokenizer("First sentence.", "Second sentence.", return_tensors="pt")
print(f"Token type IDs: {pair['token_type_ids']}")  # 0 for first, 1 for second
```

**Interview Tip:** BERT uses WordPiece tokenization. [CLS] token embedding used for classification. Fine-tuning: add task head, train all layers or just head. RoBERTa removes NSP (useless) and trains longer with more data.

</details>

<details>
<summary><strong>13. What is GPT and how does it differ from BERT?</strong></summary>

**Answer:**
GPT uses a decoder-only transformer with causal (unidirectional) attention, pre-trained with next-token prediction. BERT: encoder-only, bidirectional, masked prediction. GPT: generative, autoregressive; BERT: discriminative, non-generative.

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Text generation
input_text = "Deep learning is"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    # Greedy decoding
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=30,
        do_sample=False,           # greedy
    )
print("Greedy:", tokenizer.decode(output_ids[0], skip_special_tokens=True))

with torch.no_grad():
    # Sampling with temperature
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=30,
        do_sample=True,
        temperature=0.7,           # lower = more focused
        top_p=0.9,                 # nucleus sampling
        top_k=50,
    )
print("Sampled:", tokenizer.decode(output_ids[0], skip_special_tokens=True))

# GPT vs BERT summary
"""
BERT (Encoder-only):
- Bidirectional attention (sees whole sequence)
- Pre-training: MLM + NSP
- Best for: classification, NER, QA
- Input: [CLS] text [SEP]

GPT (Decoder-only):
- Causal (left-to-right) attention
- Pre-training: next token prediction
- Best for: generation, completion
- Autoregressive inference
"""
```

**Interview Tip:** GPT-4 is decoder-only. T5/BART are encoder-decoder. BERT-style models dominate classification; GPT-style dominate generation. "Instruction tuning" (RLHF, DPO) aligns GPT models to follow instructions.

</details>

<details>
<summary><strong>14. What is transfer learning in deep learning?</strong></summary>

**Answer:**
Transfer learning uses weights from a model pre-trained on a large dataset as initialization for a downstream task. Fine-tuning updates pre-trained weights; feature extraction freezes them and trains only the new head.

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.optim import Adam

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Feature extraction: freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer with new head for 5 classes
num_features = model.fc.in_features  # 2048
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 5)
)

# Only new head parameters are trained
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

optimizer = Adam(model.fc.parameters(), lr=1e-3)

# Fine-tuning: unfreeze last block
for param in model.layer4.parameters():
    param.requires_grad = True

# Different learning rates for different layers (common pattern)
optimizer = Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

# Transforms for pre-trained ImageNet models
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])
print("Transfer learning setup complete")
```

**Interview Tip:** Pre-trained models expect specific preprocessing (ImageNet normalization). Lower LR for pre-trained layers, higher for new layers. Domain similarity determines how much to fine-tune — similar domain: fine-tune just head; different domain: fine-tune more layers.

</details>

<details>
<summary><strong>15. What is the difference between ResNet and VGG?</strong></summary>

**Answer:**
VGG: deep sequential CNN with small 3x3 filters, no skip connections — limited depth (~19 layers). ResNet: introduces residual (skip) connections that add input to output, solving vanishing gradients — enables 50, 101, 152+ layers.

```python
import torch
import torch.nn as nn

# VGG block (sequential)
def vgg_block(in_ch, out_ch, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(in_ch if i==0 else out_ch, out_ch, kernel_size=3, padding=1),
                   nn.BatchNorm2d(out_ch), nn.ReLU()]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ResNet residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # SKIP CONNECTION: adds input to output
        return torch.relu(out)

# Why skip connections work:
# Even if conv layers learn zero, gradient flows through skip connection
# H(x) = F(x) + x, so dL/dx = dL/dH * (dF/dx + 1)
# The "+1" prevents vanishing gradient

block = ResidualBlock(64)
x = torch.randn(2, 64, 32, 32)
out = block(x)
print(f"ResBlock input: {x.shape}, output: {out.shape}")

# Model size comparison
vgg16 = torch.hub.load('pytorch/vision', 'vgg16', pretrained=False, verbose=False)
resnet50 = models.resnet50(pretrained=False)

print(f"VGG-16 params: {sum(p.numel() for p in vgg16.parameters()):,}")
print(f"ResNet-50 params: {sum(p.numel() for p in resnet50.parameters()):,}")
```

**Interview Tip:** ResNet-50 uses bottleneck blocks (1x1-3x3-1x1 conv) for efficiency. Skip connections are the most important architectural innovation. DenseNet extends this by connecting each layer to ALL previous layers.

</details>

<details>
<summary><strong>16. What is weight initialization and why does it matter?</strong></summary>

**Answer:**
Poor initialization causes vanishing/exploding gradients before training even begins. Xavier (Glorot) for tanh/sigmoid, He (Kaiming) for ReLU — both scale weights based on layer dimensions.

```python
import numpy as np
import torch
import torch.nn as nn

def xavier_init(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def he_init(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)  # x2 for ReLU (cuts signal in half)
    return np.random.randn(fan_in, fan_out) * std

# Demonstrate variance preservation
np.random.seed(42)
n_layers, n_neurons = 50, 100

# Zero initialization (bad)
x = np.ones(n_neurons)
for _ in range(n_layers):
    W = np.zeros((n_neurons, n_neurons))  # all same gradient -- symmetry breaking fails
    x = np.tanh(x @ W)

# Random initialization (bad -- explodes or vanishes)
x = np.ones(n_neurons)
for _ in range(n_layers):
    W = np.random.randn(n_neurons, n_neurons)  # std=1
    x = np.tanh(x @ W)
print(f"Random init after {n_layers} layers: mean={x.mean():.4f}, std={x.std():.6f}")  # vanishes

# Xavier initialization (good for tanh)
x = np.ones(n_neurons)
for _ in range(n_layers):
    W = xavier_init(n_neurons, n_neurons)
    x = np.tanh(x @ W)
print(f"Xavier init after {n_layers} layers: mean={x.mean():.4f}, std={x.std():.4f}")  # stable

# He initialization (good for ReLU)
x = np.ones(n_neurons)
for _ in range(n_layers):
    W = he_init(n_neurons, n_neurons)
    x = np.maximum(0, x @ W)  # ReLU
print(f"He init after {n_layers} layers: mean={x.mean():.4f}, std={x.std():.4f}")  # stable

# PyTorch initialization
layer = nn.Linear(256, 128)
nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
nn.init.zeros_(layer.bias)
print(f"He init weight std: {layer.weight.std():.4f}")
```

**Interview Tip:** Default PyTorch initializations are generally good. Xavier for tanh/sigmoid, He for ReLU. Batch normalization reduces sensitivity to initialization but doesn't eliminate the need for good initialization.

</details>

<details>
<summary><strong>17. What is learning rate scheduling?</strong></summary>

**Answer:**
Learning rate schedulers adjust LR during training. High initial LR for fast progress, decay for fine-grained convergence. Warmup + cosine decay is standard for transformers.

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, OneCycleLR,
    ReduceLROnPlateau, CosineAnnealingWarmRestarts
)
import numpy as np

model = nn.Linear(10, 1)
optimizer = Adam(model.parameters(), lr=0.1)

# 1. StepLR: decay by gamma every step_size epochs
scheduler_step = StepLR(optimizer, step_size=10, gamma=0.5)

# 2. Cosine annealing
scheduler_cos = CosineAnnealingLR(optimizer, T_max=50)

# 3. Reduce on plateau (adaptive)
scheduler_plateau = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                       patience=5, verbose=True)

# 4. One Cycle (warmup + cosine decay -- popular for fast training)
scheduler_onecycle = OneCycleLR(optimizer, max_lr=0.1, total_steps=100,
                                 pct_start=0.3)  # 30% warmup

# 5. Cosine with warm restarts (SGDR)
scheduler_restart = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

# Manual warmup + cosine decay (common for transformers)
def warmup_cosine_lr(step, warmup_steps, total_steps, min_lr=1e-7, max_lr=1e-4):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

steps = range(1000)
lrs = [warmup_cosine_lr(s, warmup_steps=100, total_steps=1000) for s in steps]
print(f"LR at step 0: {lrs[0]:.2e}")
print(f"LR at step 100 (peak): {lrs[100]:.2e}")
print(f"LR at step 999 (end): {lrs[999]:.2e}")
```

**Interview Tip:** Warmup prevents early unstable gradients with large LR. OneCycleLR achieves superconvergence. For transformers, use `d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))` (original "Attention is All You Need" schedule).

</details>

<details>
<summary><strong>18. What is an autoencoder?</strong></summary>

**Answer:**
An autoencoder learns to compress data into a latent representation (encoder) and reconstruct it (decoder). The bottleneck forces learning a compact, meaningful representation. Used for dimensionality reduction, anomaly detection, and denoising.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # output in [0,1] for image data
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Train on synthetic data
np.random.seed(42)
X = torch.FloatTensor(np.random.rand(1000, 784))  # fake image data

ae = Autoencoder(input_dim=784, latent_dim=32)
optimizer = optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

dataset = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

for epoch in range(5):
    total_loss = 0
    for (batch,) in dataset:
        optimizer.zero_grad()
        x_recon, z = ae(batch)
        loss = criterion(x_recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(dataset):.4f}")

# Anomaly detection with autoencoder
ae.eval()
normal_test = torch.FloatTensor(np.random.rand(10, 784))
anomaly_test = torch.FloatTensor(np.random.rand(10, 784) * 5)  # out of distribution

with torch.no_grad():
    recon_normal, _ = ae(normal_test)
    recon_anomaly, _ = ae(anomaly_test)

normal_mse = nn.MSELoss()(recon_normal, normal_test).item()
anomaly_mse = nn.MSELoss()(recon_anomaly, anomaly_test).item()
print(f"\nReconstruction MSE - Normal: {normal_mse:.4f}, Anomaly: {anomaly_mse:.4f}")
```

**Interview Tip:** Reconstruction loss is MSE for continuous data, BCE for binary. Sparse autoencoders add L1 on latent (encourages sparse activations). Denoising autoencoders add noise to input but reconstruct clean output (more robust features).

</details>

<details>
<summary><strong>19. What is a Variational Autoencoder (VAE)?</strong></summary>

**Answer:**
VAE learns a probabilistic latent space — encoder outputs distribution parameters (mean, variance), decoder samples from it. The loss combines reconstruction loss and KL divergence regularizing the latent space to be Gaussian.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)     # mean
        self.fc_logvar = nn.Linear(256, latent_dim) # log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * std"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std  # differentiable sampling
        return mu  # at inference, use mean directly

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

vae = VAE(784, 20)
x = torch.rand(32, 784)

x_recon, mu, logvar = vae(x)
loss = vae_loss(x_recon, x, mu, logvar)
print(f"VAE loss: {loss.item():.1f}")
print(f"Latent mu range: [{mu.min().item():.2f}, {mu.max().item():.2f}]")

# Generate new samples (pure generation)
with torch.no_grad():
    z = torch.randn(5, 20)  # sample from prior N(0,I)
    generated = vae.decode(z)
    print(f"Generated shape: {generated.shape}")
```

**Interview Tip:** Reparameterization trick enables backprop through stochastic sampling (key insight). Beta-VAE uses beta > 1 for more disentangled latent space. VAE vs GAN: VAE produces blurrier but more stable generations; GANs sharper but harder to train.

</details>

<details>
<summary><strong>20. What is a Generative Adversarial Network (GAN)?</strong></summary>

**Answer:**
GANs consist of a Generator (creates fake data) and Discriminator (distinguishes real from fake) trained adversarially. Generator minimizes discriminator's ability to detect fakes; discriminator maximizes it.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.Tanh()  # output in [-1, 1]
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

G = Generator(); D = Discriminator()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training loop (simplified)
batch_size, noise_dim = 64, 100
for step in range(3):
    real = torch.randn(batch_size, 784)  # fake "real" data
    z = torch.randn(batch_size, noise_dim)
    fake = G(z).detach()

    # Train Discriminator
    opt_D.zero_grad()
    real_loss = criterion(D(real), torch.ones(batch_size, 1))
    fake_loss = criterion(D(fake), torch.zeros(batch_size, 1))
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward(); opt_D.step()

    # Train Generator
    z = torch.randn(batch_size, noise_dim)
    opt_G.zero_grad()
    g_loss = criterion(D(G(z)), torch.ones(batch_size, 1))  # fool D
    g_loss.backward(); opt_G.step()

    print(f"Step {step}: D_loss={d_loss.item():.3f}, G_loss={g_loss.item():.3f}")
```

**Interview Tip:** GAN training challenges: mode collapse (G produces limited variety), training instability. WGAN uses Wasserstein distance for more stable training. StyleGAN/BigGAN are state-of-the-art for image generation. GAN loss is a minimax game: min_G max_D.

</details>

<details>
<summary><strong>21. What is the difference between a shallow and deep network?</strong></summary>

Shallow: 1-2 hidden layers. Deep: many layers (10, 100+). Depth enables hierarchical feature learning — early layers learn simple features (edges), later layers learn complex concepts (faces, objects). Deep networks are more parameter-efficient than wide shallow ones for complex functions.
</details>

<details>
<summary><strong>22. What is receptive field in CNNs?</strong></summary>

The receptive field is the region of the input that influences a particular output neuron. Grows with depth: two 3x3 convolutions have a 5x5 effective receptive field. Pooling, strided convs, and dilated convs increase it. Larger receptive field = more context.
</details>

<details>
<summary><strong>23. What is max pooling vs average pooling?</strong></summary>

Max pooling: takes maximum in window — captures most prominent feature, translation invariant, reduces spatial size. Average pooling: takes mean — preserves more information, smoother. Global average pooling: reduces spatial dims to 1x1 (used before classifier).
</details>

<details>
<summary><strong>24. What is depthwise separable convolution?</strong></summary>

Splits standard convolution into depthwise (filter per channel) + pointwise (1x1 conv). Reduces computation by ~8-9x for typical settings. Used in MobileNet, EfficientNet, Xception. Same expressive power with much lower cost.

```python
import torch.nn as nn

# Standard conv: C_in * C_out * K * K parameters
std_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
params_std = sum(p.numel() for p in std_conv.parameters())

# Depthwise separable:
dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)  # depthwise
pw = nn.Conv2d(32, 64, kernel_size=1)                         # pointwise
params_dw_sep = sum(p.numel() for p in dw.parameters()) + sum(p.numel() for p in pw.parameters())

print(f"Standard conv params: {params_std}")
print(f"DW-Sep conv params:   {params_dw_sep}")
print(f"Reduction: {params_std/params_dw_sep:.1f}x")
```
</details>

<details>
<summary><strong>25. What is a 1x1 convolution?</strong></summary>

A 1x1 conv applies a linear combination across channels at each spatial location. Used for: channel dimensionality reduction (bottleneck), adding non-linearity between layers, increasing channel count. Used in Inception, ResNet bottlenecks, NiN.
</details>

<details>
<summary><strong>26. What is the difference between valid and same padding?</strong></summary>

Valid: no padding, output smaller than input. Same: pad input so output same size as input. For stride=1, kernel=3: same padding adds 1 pixel each side. "Causal" padding for temporal conv: pad only left side (no future leakage).
</details>

<details>
<summary><strong>27. What is dilated (atrous) convolution?</strong></summary>

Dilated convolution inserts holes in the filter, increasing receptive field without pooling or striding. Dilation rate d: spaces d-1 zeros between filter weights. Used in segmentation (DeepLab), WaveNet for audio, TCN for sequences.
</details>

<details>
<summary><strong>28. What is transposed convolution (deconvolution)?</strong></summary>

Transposed convolution upsamples feature maps by going "backwards" through a convolution. Used in decoder parts of U-Net, segmentation networks, GANs. Often produces checkerboard artifacts — use bilinear upsampling + conv instead.
</details>

<details>
<summary><strong>29. What is U-Net and where is it used?</strong></summary>

U-Net has encoder-decoder structure with skip connections between corresponding encoder and decoder levels. Skip connections preserve fine-grained spatial information. Standard for medical image segmentation. "Contracting path" + "expansive path".
</details>

<details>
<summary><strong>30. What is YOLO?</strong></summary>

YOLO (You Only Look Once) is a single-stage object detector that divides image into grid, predicts bounding boxes and class probabilities per cell in one forward pass. Much faster than two-stage detectors (Faster R-CNN) with competitive accuracy.
</details>

<details>
<summary><strong>31. What is the difference between object detection and semantic segmentation?</strong></summary>

Object detection: bounding boxes + class labels for each object. Semantic segmentation: per-pixel class label (no instance distinction). Instance segmentation: per-pixel masks for each separate object instance (Mask R-CNN).
</details>

<details>
<summary><strong>32. What is self-supervised learning?</strong></summary>

Learning representations from unlabeled data by creating pretext tasks (predict masked tokens, image rotations, contrastive pairs). BERT, GPT, SimCLR, MoCo, MAE are self-supervised. Creates powerful representations without manual labeling.
</details>

<details>
<summary><strong>33. What is contrastive learning (SimCLR, MoCo)?</strong></summary>

Contrastive learning pulls representations of augmented views of the same image closer (positive pairs) and pushes different images apart (negative pairs). NT-Xent loss. Requires large batch or memory bank for negatives.
</details>

<details>
<summary><strong>34. What is knowledge distillation?</strong></summary>

Training a smaller "student" model to mimic the output distribution (soft targets) of a larger "teacher" model. Student matches teacher's softmax outputs (not just one-hot labels) — carries "dark knowledge" about class similarities.

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.5):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean"
    ) * (T ** 2)  # scale by T^2 to normalize

    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```
</details>

<details>
<summary><strong>35. What is model pruning?</strong></summary>

Removing redundant weights (unstructured: individual weights; structured: entire neurons/filters). Reduces model size and inference speed. Magnitude pruning: remove smallest weights. Lottery Ticket Hypothesis: sparse subnetworks exist that train as well as dense networks.
</details>

<details>
<summary><strong>36. What is quantization in deep learning?</strong></summary>

Reducing weight/activation precision from float32 to int8/int4 etc. Post-training quantization: quantize after training. Quantization-aware training: simulate quantization during training for better accuracy. Enables deployment on edge devices.
</details>

<details>
<summary><strong>37. What is neural architecture search (NAS)?</strong></summary>

Automated search for optimal network architectures. Methods: reinforcement learning (NASNet), evolutionary algorithms, differentiable NAS (DARTS). EfficientNet was found via NAS. Expensive but finds architectures humans wouldn't design.
</details>

<details>
<summary><strong>38. What is the difference between layer norm and batch norm?</strong></summary>

Batch Norm: normalizes across batch dimension (per feature). Layer Norm: normalizes across feature dimension (per sample). Layer Norm: independent of batch size, works with batch size 1, preferred for transformers. Batch Norm: preferred for CNNs.
</details>

<details>
<summary><strong>39. What is an embedding layer?</strong></summary>

An embedding layer maps discrete tokens (words, categories) to dense continuous vectors. Trainable lookup table of shape (vocab_size, embedding_dim). More efficient than one-hot encoding for high-cardinality categoricals.

```python
import torch.nn as nn
import torch

vocab_size, embed_dim = 10000, 128
embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
token_ids = torch.tensor([[1, 254, 33, 2], [5, 0, 1, 8]])  # batch of sequences
embedded = embedding(token_ids)
print(f"Embedded shape: {embedded.shape}")  # (2, 4, 128)
```
</details>

<details>
<summary><strong>40. What is positional encoding in transformers?</strong></summary>

Transformers have no inherent sequence order (attention is permutation-invariant), so positional information must be added. Sinusoidal encoding: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...). Learned positional encodings (GPT) or RoPE (Rotary Position Embedding, LLaMA) are alternatives.
</details>

<details>
<summary><strong>41. What is the difference between encoder-only, decoder-only, and encoder-decoder transformers?</strong></summary>

Encoder-only (BERT): bidirectional attention, good for understanding. Decoder-only (GPT): causal attention, good for generation. Encoder-decoder (T5, BART): encoder reads input, decoder generates output — good for translation, summarization.
</details>

<details>
<summary><strong>42. What is RLHF (Reinforcement Learning from Human Feedback)?</strong></summary>

RLHF aligns language models with human preferences: (1) supervised fine-tuning on demonstrations, (2) train reward model from human comparisons, (3) optimize policy with PPO against reward model. Used in InstructGPT, ChatGPT, Claude.
</details>

<details>
<summary><strong>43. What is the difference between fine-tuning and LoRA?</strong></summary>

Full fine-tuning updates all model weights — expensive for large models. LoRA (Low-Rank Adaptation): freezes pre-trained weights, adds trainable low-rank matrices A and B (W_new = W + A*B). Reduces trainable params by 10000x. QLoRA: quantized LoRA for further efficiency.
</details>

<details>
<summary><strong>44. What is sequence-to-sequence learning?</strong></summary>

Encoder maps input sequence to context; decoder generates output sequence one token at a time. Used for translation, summarization, dialogue. Attention between encoder and decoder (cross-attention) is crucial. Beam search improves generation quality.
</details>

<details>
<summary><strong>45. What is beam search?</strong></summary>

Greedy decoding picks the highest probability token at each step. Beam search maintains top-k (beam width) partial sequences, exploring multiple paths. Better than greedy but slower. Beam width=1 is greedy. Typical: beam_width=4-10.
</details>

<details>
<summary><strong>46. What is temperature in language model sampling?</strong></summary>

Temperature T scales logits before softmax: p = softmax(logits/T). T &lt; 1: sharper distribution (more focused/conservative). T &gt; 1: flatter distribution (more diverse/random). T=1: original distribution. T→0: greedy decoding.
</details>

<details>
<summary><strong>47. What is top-k and top-p (nucleus) sampling?</strong></summary>

Top-k: sample only from k highest probability tokens. Top-p (nucleus): sample from smallest set of tokens whose cumulative probability > p. Top-p adapts to distribution shape — better than top-k in practice. Typical: p=0.9, k=50.
</details>

<details>
<summary><strong>48. What is the perplexity metric for language models?</strong></summary>

Perplexity = exp(cross-entropy loss) = 2^(average log2 prob per token). Lower is better — measures how "surprised" the model is by the test data. A perplexity of k means the model is as confused as if choosing uniformly from k words.
</details>

<details>
<summary><strong>49. What is semantic search vs keyword search?</strong></summary>

Keyword search matches exact terms (BM25, TF-IDF). Semantic search embeds queries and documents into vector space, uses cosine similarity. Retrieval Augmented Generation (RAG) combines both: retrieve relevant documents, feed to LLM for answer generation.
</details>

<details>
<summary><strong>50. What is a vector database and how is it used in LLM applications?</strong></summary>

Vector databases (Pinecone, Weaviate, ChromaDB, FAISS) store high-dimensional embeddings and support efficient approximate nearest neighbor search. Used in RAG: embed documents, store in vector DB, embed query, retrieve similar docs, pass to LLM.
</details>

<details>
<summary><strong>51. What is gradient checkpointing?</strong></summary>

Trades computation for memory: instead of storing all activations for backprop, recompute them on-the-fly during backward pass. Reduces memory by ~sqrt(n) at cost of ~33% more computation. Essential for training very deep models or large batch sizes.
</details>

<details>
<summary><strong>52. What is mixed precision training (FP16)?</strong></summary>

Uses float16 (half precision) for forward/backward pass, float32 (master weights) for parameter updates. 2x memory reduction, faster computation on modern GPUs (Tensor Cores). Loss scaling prevents underflow. `torch.cuda.amp.autocast()` in PyTorch.
</details>

<details>
<summary><strong>53. What is gradient accumulation?</strong></summary>

Accumulate gradients over multiple small batches before performing optimizer step, effectively simulating a larger batch size. Used when GPU memory can't fit the desired batch size. `loss.backward()` n times, then `optimizer.step()`.
</details>

<details>
<summary><strong>54. What is data augmentation in deep learning?</strong></summary>

Artificially expand training data by applying transformations: flipping, rotation, cropping, color jitter, cutout, mixup, cutmix. Improves generalization, prevents overfitting. RandAugment automatically searches augmentation policies.
</details>

<details>
<summary><strong>55. What is early stopping in neural networks?</strong></summary>

Monitor validation loss during training, stop when it stops improving for patience epochs, restore best weights. Prevents overfitting. More principled than training for fixed epochs. Common: patience=5-10.
</details>

<details>
<summary><strong>56. What is the difference between pooling and strided convolutions for downsampling?</strong></summary>

Strided conv: learnable downsampling, can learn what to preserve. MaxPool: fixed, preserves maximum activation. Average pool: smooth downsampling. Modern practice: prefer strided conv over pooling (ResNet-style).
</details>

<details>
<summary><strong>57. What are attention masks in transformers?</strong></summary>

Masks control which positions attend to which. Padding mask: ignore padding tokens. Causal mask (lower triangular): prevent attending to future tokens (used in decoder self-attention). Key-value cache at inference stores past computed K,V.
</details>

<details>
<summary><strong>58. What is the KV cache in transformer inference?</strong></summary>

During autoregressive generation, previously computed key (K) and value (V) tensors are cached. Each new token only computes Q for the new position and attends to cached K,V. Reduces computation from O(n²) per step to O(n). Essential for fast LLM inference.
</details>

<details>
<summary><strong>59. What is multi-query attention (MQA) and grouped-query attention (GQA)?</strong></summary>

Multi-head attention (MHA): separate K,V heads for each Q head. MQA: all Q heads share one K,V — reduces KV cache size dramatically. GQA (used in LLaMA-2): groups of Q heads share K,V — balance between MHA quality and MQA speed.
</details>

<details>
<summary><strong>60. What is FlashAttention?</strong></summary>

FlashAttention is an IO-aware exact attention algorithm that computes attention in blocks to minimize memory reads/writes between GPU HBM and SRAM. Same result as standard attention, 2-4x faster, O(n) memory instead of O(n²). Standard in modern LLM training.
</details>

<details>
<summary><strong>61. What is PEFT (Parameter-Efficient Fine-Tuning)?</strong></summary>

Methods to fine-tune large models with minimal trainable parameters: LoRA (low-rank adapters), prefix tuning (trainable prefix tokens), prompt tuning, adapter layers. LoRA is the most popular — adds rank-r matrices only.
</details>

<details>
<summary><strong>62. What is the context window of a language model?</strong></summary>

Maximum number of tokens the model can process at once. GPT-3.5: 4096, GPT-4: 8k/128k, Claude: 200k. Constrained by O(n²) attention complexity and KV cache memory. Sliding window attention, RoPE scaling, and other methods extend context.
</details>

<details>
<summary><strong>63. What is tokenization in NLP?</strong></summary>

Converting text to token IDs. WordPiece (BERT): subword units, handles OOV. BPE (GPT): byte-pair encoding, merges frequent pairs. SentencePiece: language-agnostic subword tokenization. Typical vocab size: 30k-100k tokens.
</details>

<details>
<summary><strong>64. What is named entity recognition (NER)?</strong></summary>

Sequence labeling task that identifies and classifies named entities (person, organization, location, etc.) in text. Uses BIO tagging (Beginning, Inside, Outside). Fine-tune BERT/RoBERTa with token-level classification head.
</details>

<details>
<summary><strong>65. What is text classification vs sequence labeling?</strong></summary>

Text classification: one label per document (sentiment, topic). Sequence labeling: one label per token (NER, POS tagging). Text classification uses [CLS] embedding; sequence labeling uses all token embeddings.
</details>

<details>
<summary><strong>66. What is a language model vs a chat model?</strong></summary>

Base language model: predicts next token, raw pre-trained. Chat model (instruction-tuned): fine-tuned with RLHF/DPO to follow instructions and engage in dialogue. Chat models have system/user/assistant turn structure.
</details>

<details>
<summary><strong>67. What is retrieval augmented generation (RAG)?</strong></summary>

RAG combines retrieval with generation: (1) embed query, (2) retrieve similar documents from vector store, (3) provide retrieved docs as context to LLM, (4) generate answer grounded in retrieved knowledge. Reduces hallucination, enables up-to-date knowledge.
</details>

<details>
<summary><strong>68. What is hallucination in LLMs?</strong></summary>

LLMs generate plausible-sounding but factually incorrect text. Caused by: training data errors, insufficient knowledge, overconfident generation. Mitigated by: RAG (ground in retrieved facts), RLHF (prefer accurate answers), citations, temperature reduction.
</details>

<details>
<summary><strong>69. What is the sliding window attention?</strong></summary>

Each token attends only to a local window of w nearby tokens instead of all tokens — reduces O(n²) to O(n*w). Used in Longformer, BigBird for long documents. Global tokens (like [CLS]) still attend to everything.
</details>

<details>
<summary><strong>70. What is sparse attention?</strong></summary>

Variants of attention that compute only a subset of the full n×n attention matrix. Types: local (sliding window), strided, global+local (Longformer), random (BigBird). Enables processing much longer sequences efficiently.
</details>

<details>
<summary><strong>71. What are graph neural networks (GNNs)?</strong></summary>

NNs for graph-structured data. Message passing: each node aggregates information from neighbors. GCN, GraphSAGE, GAT (attention over neighbors). Used for: molecular property prediction, social networks, knowledge graphs.
</details>

<details>
<summary><strong>72. What is a Mixture of Experts (MoE)?</strong></summary>

MoE replaces dense FFN layers with many "expert" FFN networks. Router network selects top-k experts per token (sparse activation). Only k/N experts active per token — allows much larger model with same compute. Used in Mixtral, GPT-4 (speculated), Switch Transformer.
</details>

<details>
<summary><strong>73. What is speculative decoding?</strong></summary>

Use a small "draft" model to generate candidate tokens, then verify with large model in parallel. Achieves 2-3x speedup on large model inference with identical output. Small model generates cheap candidates; large model accepts/rejects in parallel.
</details>

<details>
<summary><strong>74. What is the difference between LSTM and GRU?</strong></summary>

LSTM: 3 gates (forget, input, output), 2 state vectors (cell state + hidden). GRU: 2 gates (reset, update), 1 state vector. GRU: fewer parameters, faster, similar performance to LSTM. LSTM: better for very long sequences. Both largely replaced by transformers.
</details>

<details>
<summary><strong>75. What is teacher forcing in sequence training?</strong></summary>

During training, feed ground-truth tokens as decoder inputs (instead of model's own predictions). Stabilizes training but creates exposure bias — at inference, model uses its own outputs. Scheduled sampling gradually replaces teacher tokens with model outputs.
</details>

<details>
<summary><strong>76. What is word2vec and how does it work?</strong></summary>

Word2Vec learns word embeddings from distributional similarity. CBOW: predict center word from context. Skip-gram: predict context words from center. Negative sampling makes training efficient. Captures semantic relationships (king - man + woman = queen).
</details>

<details>
<summary><strong>77. What is a feedforward network in transformer context?</strong></summary>

Position-wise FFN applies two linear layers with activation: FFN(x) = max(0, xW1 + b1)W2 + b2. Applied independently to each position. Hidden dim typically 4x model dim. GELU activation used in GPT/BERT. Contains most transformer parameters.
</details>

<details>
<summary><strong>78. What is cross-entropy loss for classification?</strong></summary>

CE = -sum(y_i * log(p_i)) where y is one-hot label, p is softmax output. Equivalent to negative log-likelihood. For binary: BCE = -(y*log(p) + (1-y)*log(1-p)). Focal loss adds (1-p)^gamma weighting for class imbalance (used in object detection).
</details>

<details>
<summary><strong>79. What is gradient tape in TensorFlow?</strong></summary>

TensorFlow's automatic differentiation mechanism. `tf.GradientTape` records operations for automatic gradient computation. Similar to PyTorch's autograd but explicit context manager.

```python
import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1
grad = tape.gradient(y, x)  # dy/dx = 2x + 2 = 8
print(f"Gradient: {grad.numpy()}")
```
</details>

<details>
<summary><strong>80. What is torchscript?</strong></summary>

TorchScript compiles PyTorch models to a serialized, platform-independent format runnable without Python. Two modes: tracing (records tensor operations) and scripting (supports control flow). Used for production deployment.
</details>

<details>
<summary><strong>81. What is ONNX?</strong></summary>

Open Neural Network Exchange — open format for ML models. Export from PyTorch/TensorFlow, run on any compatible runtime (ONNX Runtime, TensorRT). Enables cross-framework deployment and hardware optimization.
</details>

<details>
<summary><strong>82. What is model serving and inference optimization?</strong></summary>

Model serving: exposing model predictions via API (TorchServe, TF Serving, Triton). Inference optimization: quantization (int8), pruning, kernel fusion (TensorRT), batching, async processing, model caching. Latency vs throughput tradeoff.
</details>

<details>
<summary><strong>83. What is attention score/weights vs attention value?</strong></summary>

Scores = Q @ K^T / sqrt(d_k): raw compatibility scores. Weights = softmax(scores): probabilities summing to 1. Values = context vectors. Output = weights @ V: weighted sum of values. "What to attend to" (weights) vs "what information" (values).
</details>

<details>
<summary><strong>84. What is the difference between token-level and sentence-level embeddings?</strong></summary>

Token-level: representation per token from last hidden state (used for NER, QA). Sentence-level: pooled representation for whole sequence — [CLS] token, mean pool, or max pool. Sentence transformers (SBERT) fine-tune specifically for sentence-level similarity.
</details>

<details>
<summary><strong>85. What is reinforcement learning in deep learning context (Deep RL)?</strong></summary>

Deep RL uses neural networks as function approximators for policy (actor) and/or value function (critic). Key algorithms: DQN (Q-learning with NN), PPO (policy gradient, RLHF), A3C, SAC. Applications: games (AlphaGo), robotics, LLM alignment.
</details>

<details>
<summary><strong>86. What is curriculum learning?</strong></summary>

Training on examples ordered from easy to hard, mimicking human learning. Can improve convergence speed and final performance. Used in NLP (shorter sentences first), image classification (cleaner images first), multi-task learning.
</details>

<details>
<summary><strong>87. What is label noise robustness?</strong></summary>

Training with noisy labels leads to overfitting to noise. Methods: label smoothing, robust loss functions (MAE, symmetric CE), noise-aware training (Mixup), sample selection (small loss trick — clean samples have smaller loss early in training).
</details>

<details>
<summary><strong>88. What are skip connections in detail?</strong></summary>

Skip connections (residual) add layer input to output: H(x) = F(x) + x. Enables gradient flow through identity mapping. Variants: dense connections (DenseNet, all previous layers), gated (highway networks), attention-based.
</details>

<details>
<summary><strong>89. What is the difference between global and local self-attention?</strong></summary>

Global: every token attends to every other (standard). Local (sliding window): attends only to nearby tokens — O(n*w). Combined (Longformer, BigBird): most tokens use local window, some "global" tokens (like [CLS]) attend to all.
</details>

<details>
<summary><strong>90. What is a causal language model vs masked language model?</strong></summary>

Causal LM (GPT-style): predict next token, left-to-right only, causal mask. Masked LM (BERT-style): predict masked tokens, bidirectional. Causal: better for generation. Masked: better for understanding tasks. Unified (prefix LM): causal on some, bidirectional on prefix.
</details>

<details>
<summary><strong>91. What are the main components of a training loop?</strong></summary>

Forward pass -> compute loss -> backward pass (compute gradients) -> optimizer step (update weights) -> zero gradients. Plus: data loading, learning rate scheduling, logging, checkpointing, gradient clipping.
</details>

<details>
<summary><strong>92. What is model checkpointing?</strong></summary>

Saving model weights during training at regular intervals or when validation loss improves. Enables: resuming from crash, using best model, ensemble of checkpoints. Save: state_dict, optimizer state, scheduler state, epoch.
</details>

<details>
<summary><strong>93. What is distributed training?</strong></summary>

Training across multiple GPUs/machines. Data parallelism (DDP): same model on each GPU, split batch — gradients averaged. Model parallelism: split model across GPUs — for models too large for one GPU. Pipeline parallelism: split layers across GPUs.
</details>

<details>
<summary><strong>94. What is mixed precision and AMP?</strong></summary>

Automatic Mixed Precision (AMP) automatically casts operations to float16 where safe. `torch.cuda.amp.GradScaler` scales loss to avoid underflow in fp16 gradients. ~2x speedup on modern GPUs with Tensor Cores.
</details>

<details>
<summary><strong>95. What is the XLNet model?</strong></summary>

XLNet (Yang et al., 2019) uses permutation language modeling — trains on all possible permutations of token order, enabling bidirectional context without masking. Combines benefits of autoregressive (GPT) and autoencoding (BERT) models.
</details>

<details>
<summary><strong>96. What is SentenceBERT (SBERT)?</strong></summary>

SBERT fine-tunes BERT with siamese/triplet network on sentence pairs for semantic similarity. Produces semantically meaningful sentence embeddings suitable for cosine similarity comparison. Much faster than comparing all pairs via cross-encoder.
</details>

<details>
<summary><strong>97. What is chain-of-thought prompting?</strong></summary>

Prompting LLMs to generate intermediate reasoning steps ("Think step by step") before the final answer. Dramatically improves performance on multi-step reasoning tasks (math, logic). Works best with large models (100B+).
</details>

<details>
<summary><strong>98. What is prompt engineering?</strong></summary>

Designing input prompts to elicit desired behavior from LLMs without changing model weights. Techniques: few-shot examples, chain-of-thought, system prompts, structured output formatting, role assignment. Complements fine-tuning.
</details>

<details>
<summary><strong>99. What is the difference between pretraining and fine-tuning costs?</strong></summary>

Pretraining: extremely expensive (GPT-3: ~$4-12M, Llama-2: ~$3M compute cost) on massive datasets. Fine-tuning: much cheaper (days on fewer GPUs). LoRA/QLoRA: fine-tune 7B+ models on a single GPU in hours. Inference: ongoing cost, optimization critical.
</details>

<details>
<summary><strong>100. What are the key challenges in deploying deep learning models in production?</strong></summary>

Latency requirements, throughput, model size vs accuracy, distribution shift, explainability, fairness, monitoring for drift, versioning, A/B testing, hardware constraints, cold-start, batch vs real-time inference, continuous retraining pipelines.
</details>

---

## Architecture Timeline

| Year | Model | Innovation |
|------|-------|-----------|
| 2012 | AlexNet | Deep CNN + ReLU + Dropout |
| 2014 | GAN | Adversarial training |
| 2014 | VGG | Very deep small filters |
| 2015 | ResNet | Skip connections |
| 2015 | LSTM | Gated RNN |
| 2017 | Transformer | Self-attention |
| 2018 | BERT | Bidirectional pretraining |
| 2018 | GPT | Autoregressive pretraining |
| 2020 | GPT-3 | Scale + few-shot learning |
| 2022 | ChatGPT | RLHF alignment |
| 2023 | LLaMA | Open-source LLM |
| 2023 | Mistral/Mixtral | MoE + efficient inference |

## Summary in simple points

- Neural networks compose differentiable transformations; backpropagation applies the chain rule and an optimiser updates parameters.
- Activation, initialisation, skip connections, normalisation and gradient controls affect signal propagation. Inspect vanishing/exploding gradients rather than guessing.
- Dropout, augmentation, early stopping and regularisation target generalisation. Training and evaluation modes can change behaviour.
- Convolutions have explicit channels, receptive fields, stride, padding and dilation. Depthwise, pointwise and transposed convolutions serve different purposes.
- Pooling and strided convolution reduce spatial resolution. Classification, detection and segmentation need different outputs and metrics.
- RNNs, LSTMs, GRUs and sequence-to-sequence models manage temporal state; teacher forcing changes what inputs training observes.
- Attention uses queries, keys and values. Multi-head attention, positions and masks determine which tokens can influence each output.
- Encoder, decoder and encoder-decoder transformers expose different context. Masked and causal objectives train different prediction tasks.
- Tokenisation, embedding layers, word2vec and sentence embeddings determine representation granularity; pooling is not automatically a good semantic encoder.
- Greedy, beam, temperature, top-k and top-p decoding change generation behaviour. Perplexity needs comparable tokenisation and data.
- Autoencoders, VAEs, GANs and other generative models use different objectives and latent assumptions. Self-supervised and contrastive learning require suitable training signals.
- Transfer learning, full fine-tuning, LoRA/PEFT, distillation, pruning and quantisation trade capacity, memory and cost differently.
- RLHF and other feedback-based training depend on reward/preference quality. Deep RL also needs explicit environment and exploration assumptions.
- Gradient checkpointing, accumulation and mixed precision change memory/compute behaviour. Save optimiser and runtime state needed to resume training.
- KV caching, MQA/GQA, FlashAttention, sparse/local attention, MoE and speculation optimise different parts of inference; benchmark actual workloads.
- RAG uses external evidence but does not remove hallucination. Search, vector storage and answer validation remain separate stages.
- Framework autograd, export/compilation formats and serving runtimes have version-specific support. Test exported model parity rather than assuming it.
- Distributed training, monitoring, deployment, prompts and checkpoint management need reproducibility and failure recovery. Curriculum and noisy-label techniques still require independent evaluation.
