"""Projected multihead self-attention in NumPy; no training framework required."""
import numpy as np

def attention(x, wq, wk, wv, wo, heads=1, valid=None, causal=True):
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, time, embedding)")
    batch, length, width = x.shape
    if heads < 1 or width % heads or length == 0:
        raise ValueError("nonempty sequence and width divisible by heads required")
    if any(np.shape(w) != (width, width) for w in (wq, wk, wv, wo)):
        raise ValueError("projection matrices must be square embedding width")
    valid = np.ones((batch, length), dtype=bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != (batch, length):
        raise ValueError("padding mask must be (batch, time)")
    depth = width // heads
    def split(w):
        return (x @ w).reshape(batch, length, heads, depth).transpose(0, 2, 1, 3)
    q, k, v = map(split, (wq, wk, wv))
    logits = q @ k.swapaxes(-1, -2) / np.sqrt(depth)
    allowed = np.broadcast_to(valid[:, None, None, :], (batch, heads, length, length)).copy()
    if causal:
        allowed &= np.tril(np.ones((length, length), dtype=bool))
    allowed &= valid[:, None, :, None]
    # Fully masked queries must produce zeros instead of softmax(-inf,...,-inf).
    masked = np.where(allowed, logits, -np.inf)
    maximum = np.max(masked, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    weights = np.exp(masked - maximum)
    denominator = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    merged = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
    return merged @ wo, weights

if __name__ == "__main__":
    x = np.arange(24, dtype=float).reshape(1, 3, 8) / 24
    output, weights = attention(x, *([np.eye(8)] * 4), heads=2)
    print("output", output.shape, "weights", weights.shape)
