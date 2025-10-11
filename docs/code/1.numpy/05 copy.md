---
title: NumPy Broadcasting
sidebar_label: NumPy Broadcasting
sidebar_position: 15
---

# NumPy Broadcasting

NumPy broadcasting is a powerful mechanism that allows NumPy to work with arrays
of different shapes during arithmetic operations. It enables vectorized array
operations without making unnecessary copies of data.

## Basic Concept

Broadcasting automatically expands smaller arrays to match the shape of larger
arrays for element-wise operations.

## Broadcasting Rules

NumPy follows these rules when determining if two arrays are broadcastable:

1. **Rule 1**: If the two arrays differ in their number of dimensions, the shape
   of the one with fewer dimensions is padded with ones on its leading (left)
   side.

2. **Rule 2**: If the shape of the two arrays does not match in any dimension,
   the array with shape equal to 1 in that dimension is stretched to match the
   other shape.

3. **Rule 3**: If in any dimension the sizes disagree and neither is equal to 1,
   an error is raised.

## Examples

```python
import numpy as np

# Example 1: Scalar broadcasting
a = np.array([1, 2, 3])
b = 2
result = a + b  # b is broadcast to [2, 2, 2]
print(result)  # [3, 4, 5]

# Example 2: 1D array with 2D array
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape: (2, 3)
b = np.array([10, 20, 30])  # shape: (3,)
result = a + b  # b is broadcast to [[10, 20, 30], [10, 20, 30]]
print(result)
# [[11, 22, 33]
#  [14, 25, 36]]

# Example 3: Different dimension broadcasting
a = np.array([[1], [2], [3]])  # shape: (3, 1)
b = np.array([10, 20])         # shape: (2,)
result = a + b  # a broadcast to [[1, 1], [2, 2], [3, 3]]
                # b broadcast to [[10, 20], [10, 20], [10, 20]]
print(result)
# [[11, 21]
#  [12, 22]
#  [13, 23]]
```

## Step-by-step Broadcasting Process

```python
# Let's trace through a complex example
a = np.ones((2, 3, 4))  # shape: (2, 3, 4)
b = np.array([1, 2, 3])  # shape: (3,)

# Step 1: Align shapes (Rule 1)
# a: (2, 3, 4)
# b: (1, 3)  # padded with ones on the left

# Step 2: Compare dimensions (Rule 2)
# Dimension 0: 2 vs 1 -> stretch b to 2
# Dimension 1: 3 vs 3 -> compatible
# Dimension 2: 4 vs 1 -> stretch b to 4

# Final broadcasted shapes:
# a: (2, 3, 4) unchanged
# b: (2, 3, 4) after broadcasting

result = a + b
print(result.shape)  # (2, 3, 4)
```

## Practical Applications

### 1. Data Normalization

```python
# Normalize each feature (column) to have zero mean and unit variance
data = np.random.randn(100, 5)  # 100 samples, 5 features
mean = data.mean(axis=0)        # shape: (5,)
std = data.std(axis=0)          # shape: (5,)
normalized_data = (data - mean) / std
```

### 2. Outer Products

```python
# Compute outer product using broadcasting
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# Instead of np.outer(a, b), use broadcasting:
result = a[:, np.newaxis] * b  # shape: (3, 1) * (3,) -> (3, 3)
print(result)
# [[ 4,  5,  6]
#  [ 8, 10, 12]
#  [12, 15, 18]]
```

### 3. Distance Calculation

```python
# Calculate pairwise distances between points
points = np.array([[1, 2], [3, 4], [5, 6]])  # shape: (3, 2)
centers = np.array([[0, 0], [10, 10]])       # shape: (2, 2)

# Using broadcasting to compute distances
diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]  # shape: (3, 2, 2)
distances = np.sqrt(np.sum(diff**2, axis=2))
print(distances)
# [[ 2.236, 12.727]
#  [ 5.000,  9.220]
#  [ 7.810,  5.657]]
```

### 4. Image Processing

```python
# Apply different filters to each color channel
image = np.random.rand(256, 256, 3)  # RGB image
filters = np.array([0.2989, 0.5870, 0.1140])  # grayscale coefficients
grayscale = np.sum(image * filters, axis=2)  # broadcasting applied
```

## Broadcasting with np.newaxis

```python
# Explicitly adding dimensions for broadcasting
a = np.array([1, 2, 3])  # shape: (3,)

# Add new axis to make it 2D
a_col = a[:, np.newaxis]   # shape: (3, 1)
a_row = a[np.newaxis, :]   # shape: (1, 3)

print("Column vector:")
print(a_col)
# [[1]
#  [2]
#  [3]]

print("Row vector:")
print(a_row)
# [[1, 2, 3]]
```

## Performance Benefits

```python
import time

# Without broadcasting (slower)
def without_broadcasting(a, b):
    result = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[i, j] = a[i, j] + b[j]
    return result

# With broadcasting (faster)
def with_broadcasting(a, b):
    return a + b

# Test performance
a = np.random.rand(1000, 1000)
b = np.random.rand(1000)

start = time.time()
result1 = without_broadcasting(a, b)
print(f"Without broadcasting: {time.time() - start:.4f}s")

start = time.time()
result2 = with_broadcasting(a, b)
print(f"With broadcasting: {time.time() - start:.4f}s")

print("Results are equal:", np.allclose(result1, result2))
```

## Common Broadcasting Patterns

```python
# Pattern 1: Row-wise operations
matrix = np.random.rand(4, 3)
row_sums = matrix.sum(axis=1)  # shape: (4,)
# To divide each row by its sum:
normalized = matrix / row_sums[:, np.newaxis]

# Pattern 2: Column-wise operations
col_means = matrix.mean(axis=0)  # shape: (3,)
centered = matrix - col_means  # automatically broadcasts

# Pattern 3: Pairwise operations
vectors = np.random.rand(5, 10)
similarity = vectors @ vectors.T  # (5,10) @ (10,5) -> (5,5) pairwise dot products
```

## When Broadcasting Fails

```python
# This will raise a ValueError
a = np.array([[1, 2, 3]])  # shape: (1, 3)
b = np.array([[4, 5]])     # shape: (1, 2)

try:
    result = a + b
except ValueError as e:
    print(f"Error: {e}")
    # Error: operands could not be broadcast together with shapes (1,3) (1,2)
```

Broadcasting is one of NumPy's most powerful features, enabling concise and
efficient code while maintaining readability and performance.
