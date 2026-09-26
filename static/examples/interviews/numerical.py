"""Numerical reference implementations for gradient and kernel interviews."""
import numpy as np

def logistic_loss_gradient(x, y, w):
    x, y, w = map(lambda a: np.asarray(a, dtype=float), (x, y, w))
    z = x @ w
    # logaddexp avoids log(1 + exp(z)) overflow.
    loss = np.mean(np.logaddexp(0, z) - y * z)
    sigmoid = np.exp(-np.logaddexp(0, -z))
    gradient = x.T @ (sigmoid - y) / len(y)
    return float(loss), gradient

def finite_difference(fn, w, eps=1e-5):
    w = np.asarray(w, dtype=float)
    gradient = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = eps
        gradient[i] = (fn(w + step) - fn(w - step)) / (2 * eps)
    return gradient

def conv2d(x, kernel, stride=1, padding=0):
    """NCHW input; OIHW kernel; cross-correlation as used by DL libraries."""
    x, kernel = np.asarray(x), np.asarray(kernel)
    if x.ndim != 4 or kernel.ndim != 4 or stride < 1 or padding < 0:
        raise ValueError("invalid dimensions, stride, or padding")
    batch, channels, height, width = x.shape
    out_channels, kernel_channels, kh, kw = kernel.shape
    if channels != kernel_channels:
        raise ValueError("input channel mismatch")
    oh = (height + 2 * padding - kh) // stride + 1
    ow = (width + 2 * padding - kw) // stride + 1
    if min(oh, ow) < 1:
        raise ValueError("kernel does not fit")
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.zeros((batch, out_channels, oh, ow), dtype=np.result_type(x, kernel, float))
    for n in range(batch):
        for channel in range(out_channels):
            for row in range(oh):
                for col in range(ow):
                    patch = padded[n, :, row * stride:row * stride + kh, col * stride:col * stride + kw]
                    output[n, channel, row, col] = np.sum(patch * kernel[channel])
    return output

if __name__ == "__main__":
    x, y, w = np.array([[1., 2.], [1., -1.], [1., .5]]), np.array([1., 0., 1.]), np.array([.2, -.3])
    loss, grad = logistic_loss_gradient(x, y, w)
    reference = finite_difference(lambda p: logistic_loss_gradient(x, y, p)[0], w)
    print("loss", round(loss, 6), "gradient max error", float(np.max(np.abs(grad - reference))))
    print("convolution", conv2d(np.arange(9).reshape(1, 1, 3, 3), np.array([[[[1, 0], [0, -1]]]])).tolist())
