---
title: NumPy Interview Questions (100)
sidebar_position: 31
sidebar_label: "Reference · Numpy"
---

# NumPy Interview Questions (100)

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. Start with the [practical topic bank](03-deep-learning-llms.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](99-tools-versions.md) for current behaviour and [coding labs](11-coding-labs.md) for runnable implementations.
:::

## Beginner Questions (1-30)

<details>
<summary><strong>1. What is NumPy and its key advantages?</strong></summary>

**Answer:**
NumPy is Python library for numerical computing with N-dimensional arrays.

**Key advantages**:
- **Performance**: 50-100x faster than Python lists (C implementation)
- **Memory efficiency**: Homogeneous data uses less memory
- **Broadcasting**: Automatic operations across different shapes
- **Vectorization**: Avoid explicit loops
- **Foundation**: Base for pandas, scikit-learn, TensorFlow

```python
import numpy as np
import time

# NumPy is much faster
np_array = np.arange(1000000)
py_list = list(range(1000000))

start = time.time()
result = np_array * 2
np_time = time.time() - start

start = time.time()
result = [x * 2 for x in py_list]
py_time = time.time() - start

print(f"NumPy: {np_time:.6f}s, Python: {py_time:.6f}s")
# NumPy is typically 50-100x faster
```

**Interview Tip**: Benchmark and explain vectorization benefits for ML.
</details>

<details>
<summary><strong>2. Explain array properties (shape, dtype, ndim).</strong></summary>

**Answer:**
NumPy arrays have fixed shape and data type properties.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr.shape)    # (5,) - dimensions
print(arr.ndim)     # 1 - number of dimensions
print(arr.dtype)    # int64 - data type
print(arr.size)     # 5 - total elements
print(arr.itemsize) # 8 - bytes per element

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)
print(matrix.ndim)   # 2

# Specifying dtype
arr_float = np.array([1, 2, 3], dtype=np.float32)
arr_complex = np.array([1, 2, 3], dtype=np.complex128)
```

**Interview Tip**: Know dtypes and memory implications.
</details>

<details>
<summary><strong>3. How do you create NumPy arrays?</strong></summary>

**Answer:**
Multiple creation methods: from lists, special functions, random.

```python
import numpy as np

# From Python lists
arr = np.array([1, 2, 3])

# Special functions
zeros = np.zeros((3, 4))           # All zeros
ones = np.ones((2, 3))             # All ones
empty = np.empty((2, 2))           # Uninitialized
identity = np.eye(4)               # Identity matrix
diagonal = np.diag([1, 2, 3])      # Diagonal matrix

# Range and linspace
range_arr = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)  # 5 points

# Random arrays
random_uniform = np.random.uniform(0, 1, size=(3, 3))
random_normal = np.random.normal(0, 1, size=(1000,))
random_int = np.random.randint(1, 100, size=(10,))

# From existing
copy = arr.copy()
view = arr.view()
reshaped = arr.reshape(1, -1)
```

**Interview Tip**: Know copy vs view difference; understand reshape with -1.
</details>

<details>
<summary><strong>4. What is broadcasting?</strong></summary>

**Answer:**
Broadcasting automatically aligns arrays of different shapes for operations.

**Rules**:
1. Pad dimensions on left if needed
2. Dimensions must be equal or one is 1
3. Broadcast 1 to match larger dimension

```python
import numpy as np

# Scalar to array
arr = np.array([1, 2, 3])
result = arr + 5  # [6, 7, 8]

# 1D to 2D
arr1 = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
arr2 = np.array([1, 2, 3])               # (3,)
result = arr1 + arr2  # arr2 broadcasts to (2, 3)

# Column to matrix
col = np.array([[1], [2], [3]])      # (3, 1)
row = np.array([1, 2, 3])            # (3,) → (1, 3)
result = col + row.reshape(1, -1)

# Mean normalization (ML use case)
data = np.array([[1, 2, 3], [4, 5, 6]])
means = data.mean(axis=1, keepdims=True)  # [[2], [5]]
centered = data - means  # Broadcasting subtracts each row mean
```

**Interview Tip**: Draw dimension diagrams; essential for ML.
</details>

<details>
<summary><strong>5. Explain indexing and slicing.</strong></summary>

**Answer:**
NumPy supports integer, boolean, and fancy indexing.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Integer indexing
print(arr[0])      # 10
print(arr[-1])     # 50
print(arr[1:3])    # [20, 30]
print(arr[::2])    # [10, 30, 50]

# 2D indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0])        # [1, 2, 3]
print(matrix[0, 1])     # 2
print(matrix[:, 1])     # [2, 5, 8]
print(matrix[1:, :2])   # [[4, 5], [7, 8]]

# Boolean indexing
mask = arr > 25
print(arr[mask])  # [30, 40, 50]

# Fancy indexing
indices = [0, 2, 4]
print(arr[indices])  # [10, 30, 50]

# Multiple conditions
result = matrix[(matrix > 2) & (matrix < 6)]  # [3, 4, 5]
```

**Interview Tip**: Slicing returns views, fancy indexing returns copies.
</details>

<details>
<summary><strong>6. What are universal functions (ufuncs)?</strong></summary>

**Answer:**
ufuncs are vectorized functions that operate element-wise on arrays.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Mathematical ufuncs
print(np.sqrt(arr))       # [1, 1.41, 1.73, 2, 2.24]
print(np.exp(arr))        # exponential
print(np.log(arr))        # natural log
print(np.sin(arr))        # sine
print(np.abs(np.array([-1, -2, 3])))  # absolute

# Comparison ufuncs
result = np.greater(arr, 2)  # [False, False, True, True, True]
result = np.equal(arr, 3)    # [False, False, True, False, False]

# Arithmetic ufuncs
result = np.add(arr, 10)     # Add
result = np.multiply(arr, 2) # Multiply
result = np.divide(arr, 2)   # Divide

# Aggregate functions (not strictly ufuncs)
print(np.sum(arr))           # 15
print(np.mean(arr))          # 3
print(np.std(arr))           # Standard deviation
```

**Interview Tip**: Know common ufuncs and why they're faster than loops.
</details>

<details>
<summary><strong>7. Explain aggregation functions.</strong></summary>

**Answer:**
Functions that reduce arrays to single values (or along axes).

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Basic aggregation
print(np.sum(arr))         # 15
print(np.mean(arr))        # 3
print(np.median(arr))      # 3
print(np.std(arr))         # Standard deviation
print(np.var(arr))         # Variance
print(np.min(arr))         # 1
print(np.max(arr))         # 5

# Along axes
print(np.sum(matrix, axis=0))    # [5, 7, 9] - column sums
print(np.sum(matrix, axis=1))    # [6, 15] - row sums
print(np.mean(matrix, axis=1))   # [2, 5] - row means

# With keepdims (important for broadcasting)
result = np.mean(matrix, axis=1, keepdims=True)
print(result.shape)  # (2, 1) instead of (2,)

# Other aggregation functions
print(np.cumsum(arr))      # Cumulative sum
print(np.argmax(arr))      # Index of maximum (4)
print(np.argmin(arr))      # Index of minimum (0)
```

**Interview Tip**: Know axis parameter and keepdims for ML workflows.
</details>

<details>
<summary><strong>8. What is matrix multiplication?</strong></summary>

**Answer:**
Matrix operations for linear algebra, crucial for ML.

```python
import numpy as np

# Vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
dot_product = v1 @ v2  # Same using @ operator

# Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)   # A @ B
result = A @ B     # Modern syntax

# Element-wise multiplication (different!)
element_wise = A * B

# Transpose
A_T = A.T

# Inverse
A_inv = np.linalg.inv(A)

# Determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solving linear systems (Ax = b)
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

**Interview Tip**: Know @ operator, transpose, and linalg functions for ML.
</details>

<details>
<summary><strong>9. How do you handle random numbers?</strong></summary>

**Answer:**
NumPy provides comprehensive random number generation.

```python
import numpy as np

# Seeding for reproducibility
np.random.seed(42)

# Uniform distribution (0-1)
random_uniform = np.random.uniform(0, 1, size=(1000,))

# Normal distribution (Gaussian)
random_normal = np.random.normal(loc=0, scale=1, size=(1000,))

# Integers
random_int = np.random.randint(1, 100, size=(10,))

# Random choice
arr = np.array([1, 2, 3, 4, 5])
samples = np.random.choice(arr, size=10)  # With replacement
samples = np.random.choice(arr, size=3, replace=False)  # Without replacement

# Permutation (shuffle)
shuffled = np.random.permutation(arr)

# Probability distributions
poisson = np.random.poisson(lam=5, size=(1000,))
exponential = np.random.exponential(scale=2, size=(1000,))
binomial = np.random.binomial(n=10, p=0.5, size=(1000,))

# Modern approach (newer NumPy)
rng = np.random.default_rng(seed=42)
random_normal = rng.normal(size=(1000,))
```

**Interview Tip**: Always mention seeding for reproducibility in ML.
</details>

<details>
<summary><strong>10. How do you reshape and manipulate arrays?</strong></summary>

**Answer:**
Reshaping changes array dimensions without changing data.

```python
import numpy as np

arr = np.arange(12)  # [0, 1, ..., 11]

# Reshape
reshaped = arr.reshape(3, 4)      # 2D array
reshaped = arr.reshape(2, 3, 2)   # 3D array

# Flatten
flat = reshaped.flatten()  # Returns copy
flat = reshaped.ravel()    # Returns view (if possible)

# Transpose
matrix = np.arange(6).reshape(2, 3)
transposed = matrix.T

# Concatenate
a = np.array([1, 2])
b = np.array([3, 4])
result = np.concatenate([a, b])  # [1, 2, 3, 4]

# Stack
stacked = np.stack([a, b])  # [[1, 2], [3, 4]]

# Split
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, 3)  # [[1, 2], [3, 4], [5, 6]]

# Squeeze (remove dimensions of size 1)
arr = np.array([[[1], [2]], [[3], [4]]])
squeezed = np.squeeze(arr)

# Expand dimensions
arr = np.array([1, 2, 3])
expanded = np.expand_dims(arr, axis=0)  # [[1, 2, 3]]
```

**Interview Tip**: Know difference between view (reshape, ravel) and copy (flatten).
</details>

<details>
<summary><strong>11. What are NumPy universal functions (ufuncs)?</strong></summary>

**Answer:**
Element-wise operations on arrays, much faster than Python loops.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Mathematical ufuncs
np.sqrt(arr)           # [1.0, 1.41, 1.73, 2.0, 2.24]
np.exp(arr)            # exponential
np.log(arr)            # natural logarithm
np.sin(arr)            # sine
np.cos(arr)            # cosine

# Comparison ufuncs
result = np.greater(arr, 2)  # [False, False, True, True, True]
result = np.equal(arr, 3)    # [False, False, True, False, False]

# Arithmetic ufuncs
np.add(arr, 10)        # [11, 12, 13, 14, 15]
np.multiply(arr, 2)    # [2, 4, 6, 8, 10]
np.divide(arr, 2)      # [0.5, 1.0, 1.5, 2.0, 2.5]
np.power(arr, 2)       # [1, 4, 9, 16, 25]

# Absolute value
np.abs(np.array([-1, -2, 3]))  # [1, 2, 3]

# Custom ufuncs (advanced)
def double(x):
    return x * 2

ufunc = np.frompyfunc(double, 1, 1)
result = ufunc(arr)
```

**Interview Tip**: Explain why ufuncs are faster (C implementation, vectorized).
</details>

<details>
<summary><strong>12. What are aggregation functions?</strong></summary>

**Answer:**
Reduce arrays to single values or along axes.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Basic aggregation
np.sum(arr)         # 15
np.mean(arr)        # 3.0
np.median(arr)      # 3.0
np.std(arr)         # Standard deviation
np.var(arr)         # Variance
np.min(arr)         # 1
np.max(arr)         # 5

# Along axes
np.sum(matrix, axis=0)    # [5, 7, 9] - column sums
np.sum(matrix, axis=1)    # [6, 15] - row sums
np.mean(matrix, axis=1)   # [2.0, 5.0] - row means

# With keepdims (for broadcasting)
result = np.mean(matrix, axis=1, keepdims=True)
print(result.shape)  # (2, 1) instead of (2,)

# Other aggregation functions
np.cumsum(arr)      # [1, 3, 6, 10, 15] - cumulative sum
np.argmax(arr)      # 4 (index of maximum)
np.argmin(arr)      # 0 (index of minimum)
np.prod(arr)        # 120 (product)

# Percentile
np.percentile(arr, 25)   # 25th percentile
np.percentile(arr, 75)   # 75th percentile
```

**Interview Tip**: Know axis parameter and keepdims for ML workflows.
</details>

<details>
<summary><strong>13. What is matrix multiplication and linear algebra?</strong></summary>

**Answer:**
Linear algebra operations fundamental to ML.

```python
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
dot_product = v1 @ v2         # Same (Python 3.5+)

# Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)              # or A @ B
# [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]

# Element-wise multiplication (different!)
element_wise = A * B

# Transpose
A_T = A.T

# Inverse
A_inv = np.linalg.inv(A)

# Determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solving linear systems (Ax = b)
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)  # Solution to Ax = b

# Norm (vector length)
norm = np.linalg.norm(v1)  # sqrt(1^2 + 2^2 + 3^2) = 3.74
```

**Interview Tip**: Know @ operator and common linalg functions for ML.
</details>

<details>
<summary><strong>14. What is random number generation?</strong></summary>

**Answer:**
NumPy provides comprehensive random number generation.

```python
import numpy as np

# Seeding for reproducibility
np.random.seed(42)

# Uniform distribution (0-1)
random_uniform = np.random.uniform(0, 1, size=(1000,))

# Normal distribution (Gaussian)
random_normal = np.random.normal(loc=0, scale=1, size=(1000,))

# Integers
random_int = np.random.randint(1, 100, size=(10,))

# Random choice
arr = np.array([1, 2, 3, 4, 5])
samples = np.random.choice(arr, size=10)  # With replacement
samples = np.random.choice(arr, size=3, replace=False)  # Without

# Permutation (shuffle)
shuffled = np.random.permutation(arr)

# Probability distributions
poisson = np.random.poisson(lam=5, size=(1000,))
exponential = np.random.exponential(scale=2, size=(1000,))
binomial = np.random.binomial(n=10, p=0.5, size=(1000,))

# Modern approach (newer NumPy)
rng = np.random.default_rng(seed=42)
random_normal = rng.normal(size=(1000,))
random_int = rng.integers(1, 100, size=(10,))

# Reproducibility important for ML/debugging
def train_model():
    np.random.seed(42)  # Set seed at start
    # ... training code
```

**Interview Tip**: Always mention seeding for reproducibility.
</details>

<details>
<summary><strong>15. What is array reshaping?</strong></summary>

**Answer:**
Change array dimensions without changing data.

```python
import numpy as np

arr = np.arange(12)  # [0, 1, ..., 11]

# Reshape
reshaped = arr.reshape(3, 4)      # 2D array (3x4)
reshaped = arr.reshape(2, 3, 2)   # 3D array (2x3x2)

# Flatten (returns copy)
flat = reshaped.flatten()  # [0, 1, 2, ..., 11]

# Ravel (returns view if possible, usually faster)
flat = reshaped.ravel()    # [0, 1, 2, ..., 11]

# Transpose
matrix = np.arange(6).reshape(2, 3)
transposed = matrix.T  # (3, 2)

# Concatenate
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
result = np.concatenate([a, b], axis=0)  # (3, 2)

# Stack
stacked = np.stack([a, b], axis=0)

# Split
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, 3)  # [[1, 2], [3, 4], [5, 6]]

# Squeeze (remove dimensions of size 1)
arr = np.array([[[1], [2]], [[3], [4]]])
squeezed = np.squeeze(arr)  # (2, 2)

# Expand dimensions
arr = np.array([1, 2, 3])
expanded = np.expand_dims(arr, axis=0)  # [[1, 2, 3]]

# Using -1 to auto-calculate dimension
arr = np.arange(12)
reshaped = arr.reshape(3, -1)  # (3, 4) - -1 calculated as 4
```

**Interview Tip**: Know difference between view (reshape, ravel) and copy (flatten).
</details>

<details>
<summary><strong>16. How do you sort arrays and use argsort?</strong></summary>

**Answer:**
`np.sort` returns sorted copy; `argsort` returns indices that would sort the array.

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Sort ascending (returns copy)
print(np.sort(arr))          # [1 1 2 3 4 5 6 9]

# Sort descending
print(np.sort(arr)[::-1])    # [9 6 5 4 3 2 1 1]

# In-place sort
arr.sort()

# argsort: indices that sort the array
arr = np.array([3, 1, 4, 1, 5])
idx = np.argsort(arr)
print(idx)          # [1 3 0 2 4]
print(arr[idx])     # [1 1 3 4 5]  sorted via fancy indexing

# Sort 2D by column
matrix = np.array([[3, 1], [1, 4], [2, 2]])
sorted_rows = matrix[matrix[:, 0].argsort()]  # sort by first column
print(sorted_rows)  # [[1 4], [2 2], [3 1]]

# searchsorted: find insertion point in sorted array (binary search)
a = np.array([1, 3, 5, 7, 9])
print(np.searchsorted(a, 4))   # 2  (insert before index 2)
print(np.searchsorted(a, 5))   # 2  (left-side default)
print(np.searchsorted(a, 5, side='right'))  # 3

# np.where: conditional selection
arr = np.array([1, -2, 3, -4, 5])
print(np.where(arr > 0, arr, 0))    # [1 0 3 0 5]  (replace negatives with 0)
print(np.where(arr > 0))            # (array([0, 2, 4]),)  indices of True
```
</details>

<details>
<summary><strong>17. What are unique, repeat, and tile operations?</strong></summary>

**Answer:**
`unique` finds distinct values; `repeat`/`tile` replicate data.

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])

# unique values
print(np.unique(arr))                          # [1 2 3 4 5 6 9]

# unique with counts
vals, counts = np.unique(arr, return_counts=True)
print(dict(zip(vals, counts)))                 # {1:2, 2:1, 3:1, 4:1, 5:2, 6:1, 9:1}

# unique with inverse indices (reconstruct original)
vals, inv = np.unique(arr, return_inverse=True)
print(vals[inv])  # reconstructed original

# repeat: repeat each element N times
print(np.repeat([1, 2, 3], 3))         # [1 1 1 2 2 2 3 3 3]
print(np.repeat([1, 2, 3], [1, 2, 3])) # [1 2 2 3 3 3]  variable repeats

# tile: repeat the whole array N times
print(np.tile([1, 2, 3], 3))           # [1 2 3 1 2 3 1 2 3]
print(np.tile([1, 2, 3], (2, 3)))      # 2x3 repetition:
# [[1 2 3 1 2 3 1 2 3]
#  [1 2 3 1 2 3 1 2 3]]
```
</details>

<details>
<summary><strong>18. What set operations does NumPy support?</strong></summary>

**Answer:**
NumPy provides fast set operations on 1D arrays.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])

print(np.union1d(a, b))        # [1 2 3 4 5 6 7]  — union
print(np.intersect1d(a, b))    # [3 4 5]           — intersection
print(np.setdiff1d(a, b))      # [1 2]             — in a but not b
print(np.setxor1d(a, b))       # [1 2 6 7]         — symmetric difference

# Check membership
print(np.isin(a, b))           # [F F T T T]
print(np.isin(a, b, invert=True))  # [T T F F F]

# in1d (legacy, prefer isin)
print(np.in1d([1, 2, 3], [2, 4]))  # [F T F]
```
</details>

<details>
<summary><strong>19. How do you compute histograms and bin data?</strong></summary>

**Answer:**
`np.histogram` computes frequency counts; `np.digitize` assigns bin labels.

```python
import numpy as np

data = np.random.normal(0, 1, 1000)

# Histogram: counts and bin edges
counts, edges = np.histogram(data, bins=10)
print(counts)   # frequency per bin
print(edges)    # 11 edges for 10 bins

# Custom bin edges
counts, edges = np.histogram(data, bins=[-3, -2, -1, 0, 1, 2, 3])

# 2D histogram
x = np.random.randn(1000)
y = np.random.randn(1000)
H, xedges, yedges = np.histogram2d(x, y, bins=20)

# digitize: assign each value to a bin index
bins = np.array([0, 25, 50, 75, 100])
ages = np.array([15, 30, 60, 80, 45])
bucket = np.digitize(ages, bins)
print(bucket)   # [1 2 3 4 3]  (1-indexed bin number)

# Bin statistics with np.bincount
values = np.array([2, 0, 1, 3, 1, 0, 0])
print(np.bincount(values))        # [3 2 1 1]  counts per integer
print(np.bincount(values) / len(values))  # frequencies
```
</details>

<details>
<summary><strong>20. How does NumPy handle polynomials?</strong></summary>

**Answer:**
`np.polyfit` fits polynomials; `np.polyval` evaluates; `np.poly1d` wraps coefficients.

```python
import numpy as np

# Fit polynomial (least squares)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Degree-2 polynomial fit
coeffs = np.polyfit(x, y, 2)       # [a, b, c] for ax^2+bx+c
print(coeffs)

# Evaluate polynomial at new points
y_pred = np.polyval(coeffs, x)
y_new  = np.polyval(coeffs, [6, 7])

# poly1d wrapper (more OOP)
p = np.poly1d(coeffs)
print(p(3))          # evaluate at x=3
print(p.roots)       # roots of polynomial
print(p.deriv())     # derivative polynomial
print(p.integ())     # antiderivative

# Polynomial arithmetic
p1 = np.poly1d([1, 2])   # x + 2
p2 = np.poly1d([1, -1])  # x - 1
print(p1 * p2)            # (x+2)(x-1) = x^2+x-2
print(np.polymul([1, 2], [1, -1]))  # same via coefficients

# np.roots: find roots given coefficients
coeffs = [1, -3, 2]   # x^2 - 3x + 2 = (x-1)(x-2)
print(np.roots(coeffs))   # [2. 1.]
```
</details>

<details>
<summary><strong>21. How do you save and load NumPy arrays?</strong></summary>

**Answer:**
Use `.npy` for single arrays, `.npz` for multiple, and text formats for interoperability.

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Binary format (.npy) — fast, preserves dtype
np.save('array.npy', arr)
loaded = np.load('array.npy')
print(loaded.dtype, loaded.shape)   # int64, (2,3)

# Multiple arrays in one file (.npz)
x = np.arange(10)
y = np.random.randn(10)
np.savez('data.npz', x=x, y=y)

npz = np.load('data.npz')
print(npz['x'], npz['y'])
print(list(npz.files))  # ['x', 'y']

# Compressed (smaller file, slower)
np.savez_compressed('data_compressed.npz', x=x, y=y)

# Text format (CSV-like) — interoperable but larger
np.savetxt('array.csv', arr, delimiter=',', fmt='%.2f')
loaded_csv = np.loadtxt('array.csv', delimiter=',')

# Memory-mapped files (for arrays too large to fit in RAM)
mmap = np.memmap('large.dat', dtype='float32', mode='w+', shape=(10000, 1000))
mmap[:100] = np.random.randn(100, 1000)
del mmap   # flush to disk

mmap_read = np.memmap('large.dat', dtype='float32', mode='r', shape=(10000, 1000))
print(mmap_read[:100].mean())
```
</details>

<details>
<summary><strong>22. What are NumPy dtypes and type conversions?</strong></summary>

**Answer:**
NumPy supports many numeric types; choosing the right dtype saves memory and prevents overflow.

```python
import numpy as np

# Common dtypes
arr_int8  = np.array([1, 2, 3], dtype=np.int8)    # -128 to 127
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
arr_int64 = np.array([1, 2, 3], dtype=np.int64)
arr_f32   = np.array([1.0], dtype=np.float32)      # single precision
arr_f64   = np.array([1.0], dtype=np.float64)      # double precision
arr_bool  = np.array([True, False], dtype=bool)
arr_cmplx = np.array([1+2j], dtype=np.complex128)

# Memory sizes
for dtype in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]:
    print(f"{dtype.__name__}: {np.dtype(dtype).itemsize} bytes")

# Type conversion (astype creates a copy)
arr = np.array([1.7, 2.9, 3.1])
print(arr.astype(np.int32))    # [1 2 3]  truncates, not rounds!
print(arr.astype(np.float32))  # precision reduced

# Check type
print(np.issubdtype(arr.dtype, np.floating))  # True
print(np.issubdtype(arr.dtype, np.integer))   # False

# Overflow behavior
a = np.array([200], dtype=np.int8)
print(a + 100)   # wraps around: [-56]  (200+100=300; 300%256=44; 44-128=-84... careful!)

# Safe casting
try:
    np.array([300], dtype=np.int8)
except OverflowError:
    pass

# Dtype from string
dt = np.dtype('float32')
dt = np.dtype('<f4')   # little-endian float32
```
</details>

<details>
<summary><strong>23. What are masked arrays?</strong></summary>

**Answer:**
Masked arrays handle missing or invalid data by attaching a boolean mask.

```python
import numpy as np
import numpy.ma as ma

data = np.array([1.0, -999.0, 3.0, -999.0, 5.0])

# Create masked array (mask=True means "invalid/hidden")
masked = ma.masked_where(data == -999.0, data)
print(masked)           # [1.0 -- 3.0 -- 5.0]
print(masked.mean())    # 3.0  (ignores masked values)
print(masked.data)      # original data
print(masked.mask)      # [F T F T F]

# Create directly
masked2 = ma.array(data, mask=[0, 1, 0, 1, 0])

# Mask values outside range
masked3 = ma.masked_outside(data, 0, 10)  # mask values not in [0,10]

# Fill masked values for output
filled = masked.filled(fill_value=0.0)    # replace masked with 0
print(filled)   # [1. 0. 3. 0. 5.]

# Operations: masked values propagate
a = ma.array([1, 2, 3], mask=[0, 1, 0])
b = ma.array([4, 5, 6], mask=[0, 0, 1])
print(a + b)    # [5 -- --]  (any masked input -> masked output)

# Compress: remove masked values
print(masked.compressed())   # [1. 3. 5.]
```

**Use case**: sensor data with sentinel missing values, image processing with ROI masks.
</details>

<details>
<summary><strong>24. Explain fancy indexing and advanced indexing.</strong></summary>

**Answer:**
Fancy indexing uses integer arrays or boolean arrays as indices — always returns a copy.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Integer array indexing (fancy indexing)
idx = np.array([0, 2, 4])
print(arr[idx])          # [10 30 50]
print(arr[[1, 3, 1]])    # [20 40 20]  can repeat indices

# 2D fancy indexing
matrix = np.arange(16).reshape(4, 4)
rows = np.array([0, 1, 2])
cols = np.array([1, 2, 3])
print(matrix[rows, cols])   # [1 6 11]  diagonal-like

# All combinations: use ix_
print(matrix[np.ix_([0, 2], [1, 3])])
# [[1  3]
#  [9 11]]

# Boolean indexing (masking)
data = np.array([1, -2, 3, -4, 5])
mask = data > 0
print(data[mask])            # [1 3 5]  — returns copy
data[data < 0] = 0           # in-place via boolean index
print(data)                  # [1 0 3 0 5]

# Combined boolean
a = np.arange(12).reshape(3, 4)
print(a[(a > 3) & (a < 9)])  # [4 5 6 7 8]

# Fancy index assignment
arr = np.zeros(5)
arr[[1, 3]] = 99
print(arr)   # [0. 99. 0. 99. 0.]
```

**Key**: fancy indexing always copies; basic slicing produces a view.
</details>

<details>
<summary><strong>25. What is C-order vs Fortran-order memory layout?</strong></summary>

**Answer:**
C-order (row-major) stores rows contiguously; Fortran-order (column-major) stores columns contiguously. Matters for performance.

```python
import numpy as np

# C-order (default): rows are contiguous in memory
c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(c.flags['C_CONTIGUOUS'])   # True
print(c.strides)                 # (24, 8) bytes — move 24 bytes to next row

# Fortran-order: columns are contiguous
f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f.flags['F_CONTIGUOUS'])   # True
print(f.strides)                 # (8, 16) bytes — move 8 bytes to next column

# Reshape preserves data; order affects layout
arr = np.arange(6)
c_matrix = arr.reshape(2, 3, order='C')   # row-major fill
f_matrix = arr.reshape(2, 3, order='F')   # column-major fill
print(c_matrix)  # [[0 1 2], [3 4 5]]
print(f_matrix)  # [[0 2 4], [1 3 5]]

# Performance: iterate in memory-contiguous direction
import time
big = np.random.randn(5000, 5000)
# Row iteration (C-order): fast
t0 = time.time(); _ = big.sum(axis=1); print(f"row sum: {time.time()-t0:.3f}s")

# Convert order
f_order = np.asfortranarray(big)
c_order = np.ascontiguousarray(f_order)

# Useful for BLAS/LAPACK interop (Fortran-based libraries prefer F-order)
```
</details>

<details>
<summary><strong>26. What is the difference between a view and a copy?</strong></summary>

**Answer:**
A view shares the same memory as the original; modifying one affects the other. A copy is independent.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Slicing creates a VIEW
view = arr[1:4]
view[0] = 99
print(arr)    # [1 99 3 4 5]  — original changed!

# Check if view or copy
print(view.base is arr)    # True  — view
print(view.flags['OWNDATA'])  # False — doesn't own data

# Explicit copy
copy = arr[1:4].copy()
copy[0] = 0
print(arr)    # unchanged

# Fancy indexing always copies
fancy = arr[[0, 2, 4]]
fancy[0] = -1
print(arr)    # unchanged (fancy indexing gave a copy)

# reshape may return view or copy
a = np.arange(6)
b = a.reshape(2, 3)
print(b.base is a)   # True — reshape is often a view

# flatten (copy) vs ravel (view when possible)
flat_copy = b.flatten()   # always copy
flat_view = b.ravel()     # view if possible

flat_view[0] = 99
print(a[0])   # 99 — ravel returned a view

# np.may_share_memory
print(np.may_share_memory(arr, view))   # True
print(np.shares_memory(arr, view))      # True (definitive)
```
</details>

<details>
<summary><strong>27. How does broadcasting work in detail?</strong></summary>

**Answer:**
Broadcasting aligns shapes right-to-left, expanding dimensions of size 1 to match.

```python
import numpy as np

# Rules:
# 1. If arrays have different ndim, prepend 1s to smaller shape
# 2. Dimensions of size 1 are stretched to match the other
# 3. Dimensions must match or one must be 1

# Example: (3,) + (4,3)
a = np.array([1, 2, 3])          # shape (3,)
b = np.ones((4, 3))              # shape (4,3)
print((a + b).shape)             # (4,3): a broadcast to (4,3)

# Example: column vector + row vector
col = np.array([[1], [2], [3]])  # shape (3,1)
row = np.array([10, 20, 30])     # shape (3,)   -> (1,3) -> (3,3)
print(col + row)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]

# Subtract column mean from each column (normalize)
data = np.random.randn(100, 5)
col_mean = data.mean(axis=0)     # shape (5,)
normalized = data - col_mean     # broadcasts (5,) to (100,5)

# Outer product via broadcasting
a = np.array([1, 2, 3])[:, np.newaxis]  # (3,1)
b = np.array([10, 20, 30])              # (3,)
print(a * b)  # outer product (3,3)

# np.newaxis = None (adds dimension)
arr = np.array([1, 2, 3])        # (3,)
print(arr[:, np.newaxis].shape)  # (3,1)
print(arr[np.newaxis, :].shape)  # (1,3)
```
</details>

<details>
<summary><strong>28. How do you compute FFT (Fast Fourier Transform)?</strong></summary>

**Answer:**
`np.fft` module provides FFT for signal processing and frequency analysis.

```python
import numpy as np

# Signal: sum of two sine waves
fs = 1000        # sampling frequency (Hz)
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# FFT
fft_vals = np.fft.fft(signal)
freqs    = np.fft.fftfreq(len(signal), d=1/fs)

# Magnitude spectrum (only positive frequencies)
pos_mask = freqs > 0
magnitude = np.abs(fft_vals[pos_mask])
pos_freqs = freqs[pos_mask]

# Peak frequencies
peak_idx = np.argsort(magnitude)[-2:][::-1]
print(pos_freqs[peak_idx])   # [50. 120.]  detected frequencies

# Inverse FFT
reconstructed = np.fft.ifft(fft_vals).real
print(np.allclose(signal, reconstructed))   # True

# 2D FFT (image processing)
image = np.random.randn(256, 256)
fft2d = np.fft.fft2(image)
fft2d_shifted = np.fft.fftshift(fft2d)  # center zero frequency

# Power spectral density
psd = np.abs(fft_vals)**2 / len(signal)
```
</details>

<details>
<summary><strong>29. How do you do linear algebra with NumPy?</strong></summary>

**Answer:**
`np.linalg` provides solvers, decompositions, and matrix operations.

```python
import numpy as np

A = np.array([[2, 1], [5, 3]], dtype=float)
b = np.array([8, 13], dtype=float)

# Solve Ax = b
x = np.linalg.solve(A, b)
print(x)                          # [11. -14.]
print(np.allclose(A @ x, b))      # True

# Inverse (prefer solve over inverse for numerical stability)
A_inv = np.linalg.inv(A)
print(A_inv @ A)                  # identity (approximately)

# Determinant
print(np.linalg.det(A))           # 1.0

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)   # eigenvalues
# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    print(np.allclose(A @ v, eigenvalues[i] * v))   # True

# SVD
U, S, Vt = np.linalg.svd(A)
# Reconstruct: A = U @ np.diag(S) @ Vt
print(np.allclose(U @ np.diag(S) @ Vt, A))   # True

# Rank and condition number
print(np.linalg.matrix_rank(A))   # 2
print(np.linalg.cond(A))          # condition number

# Norm
print(np.linalg.norm(b))          # L2 norm = sqrt(8^2 + 13^2)
print(np.linalg.norm(A, 'fro'))   # Frobenius norm
```
</details>

<details>
<summary><strong>30. How do you use np.einsum for tensor operations?</strong></summary>

**Answer:**
`einsum` expresses complex tensor contractions in Einstein notation — often faster than chained operations.

```python
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.random.randn(3, 5)

# Matrix multiplication: ij,jk->ik
print(np.allclose(np.einsum('ij,jk->ik', A, B), A @ B))   # True

# Dot product: i,i->
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.einsum('i,i->', a, b))   # 32  (scalar)

# Outer product: i,j->ij
print(np.einsum('i,j->ij', a, b))  # (3,3) outer product

# Trace: ii->
M = np.eye(3) * 5
print(np.einsum('ii->', M))        # 15.0 (sum of diagonal)

# Transpose: ij->ji
print(np.einsum('ij->ji', A).shape)  # (4,3)

# Batch matrix multiply: bij,bjk->bik
batch_A = np.random.randn(10, 3, 4)
batch_B = np.random.randn(10, 4, 5)
result = np.einsum('bij,bjk->bik', batch_A, batch_B)
print(result.shape)   # (10, 3, 5)

# Hadamard product + sum: ij,ij->
print(np.einsum('ij,ij->', A, A))   # sum of element-wise squares = Frobenius^2

# Performance: optimize=True rewrites expression for speed
np.einsum('ij,jk,kl->il', A, B[:, :3], C, optimize=True)
```
</details>

---

## Intermediate Questions (31-70)

<details>
<summary><strong>31. How do you optimize NumPy performance?</strong></summary>

**Answer:**
Use vectorization, broadcasting, and avoid Python loops.

```python
import numpy as np

# SLOW: Python loop
data = np.arange(1000000)
result = []
for x in data:
    result.append(x ** 2)

# FAST: Vectorized
result = data ** 2

# SLOW: Element-wise operations
matrix = np.random.randn(1000, 1000)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        matrix[i, j] = matrix[i, j] ** 2

# FAST: Vectorized
matrix = matrix ** 2

# Using numexpr for complex operations
import numexpr
a = np.random.randn(1000000)
b = np.random.randn(1000000)

# NumPy
result = np.sin(a) * np.cos(b) + np.sqrt(a)

# numexpr (faster for complex expressions)
result = numexpr.evaluate('sin(a) * cos(b) + sqrt(a)')
```

**Interview Tip**: Explain why vectorization is faster; mention memory access patterns.
</details>

<details>
<summary><strong>32. How do you create meshgrids and grids?</strong></summary>

**Answer:**
`np.meshgrid` creates coordinate grids for vectorized evaluation over 2D/3D spaces.

```python
import numpy as np

x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)

# meshgrid: each output has shape (len(y), len(x))
X, Y = np.meshgrid(x, y)
print(X.shape, Y.shape)   # (5,5), (5,5)

# Evaluate function on grid
Z = np.sin(X) * np.cos(Y)

# indexing='ij' gives matrix indexing (X[i,j] = x[i])
X_ij, Y_ij = np.meshgrid(x, y, indexing='ij')   # shape (5,5)

# ogrid / mgrid: open/dense grid (memory efficient)
yg, xg = np.mgrid[-2:2:5j, -2:2:5j]  # 5j = 5 points
print(yg.shape, xg.shape)   # (5,5), (5,5)

# ogrid: sparse (no broadcasting cost)
yg, xg = np.ogrid[-2:2:5j, -2:2:5j]
print(yg.shape, xg.shape)   # (5,1), (1,5)  — broadcast on demand
Z = np.sin(xg) * np.cos(yg)   # shape (5,5) via broadcast

# linspace variants
print(np.linspace(0, 1, 5))           # [0. .25 .5 .75 1.]  inclusive
print(np.arange(0, 1, 0.25))          # [0. .25 .5 .75]  exclusive end
print(np.logspace(0, 3, 4))           # [1, 10, 100, 1000] log-spaced
print(np.geomspace(1, 1000, 4))       # same as logspace
```
</details>

<details>
<summary><strong>33. How do you do statistical analysis with NumPy?</strong></summary>

**Answer:**
NumPy provides comprehensive statistical functions with axis support.

```python
import numpy as np

data = np.random.randn(100, 5)  # 100 samples, 5 features

# Central tendency
print(np.mean(data, axis=0))       # column means (shape 5,)
print(np.median(data, axis=0))     # column medians
print(np.average(data, weights=np.random.rand(100), axis=0))  # weighted mean

# Spread
print(np.std(data, axis=0))        # std (ddof=0 default)
print(np.std(data, axis=0, ddof=1)) # sample std (ddof=1)
print(np.var(data, axis=0))        # variance
print(np.ptp(data, axis=0))        # peak-to-peak (max - min)

# Percentiles and quantiles
print(np.percentile(data, [25, 50, 75], axis=0))  # quartiles
print(np.quantile(data, 0.9, axis=0))              # 90th percentile

# Correlation
X = np.random.randn(5, 100)   # 5 features, 100 samples
cov_matrix = np.cov(X)         # (5,5) covariance matrix
corr_matrix = np.corrcoef(X)   # (5,5) correlation matrix

# Cumulative operations
arr = np.array([1, 2, 3, 4, 5])
print(np.cumsum(arr))    # [1  3  6 10 15]
print(np.cumprod(arr))   # [1  2  6 24 120]
print(np.diff(arr))      # [1 1 1 1]  first differences
print(np.diff(arr, n=2)) # [0 0 0]    second differences
```
</details>

<details>
<summary><strong>34. How do you interpolate data with NumPy?</strong></summary>

**Answer:**
`np.interp` does 1D piecewise linear interpolation.

```python
import numpy as np

# Known data points
x_known = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y_known = np.array([0.0, 1.0, 4.0, 9.0, 16.0])   # y = x^2

# Interpolate at new points
x_new = np.linspace(0, 4, 20)
y_interp = np.interp(x_new, x_known, y_new)

# Clamp behavior outside range
x_out = np.array([-1.0, 5.0])
print(np.interp(x_out, x_known, y_known))          # [0. 16.]  clamp to edges
print(np.interp(x_out, x_known, y_known, left=-1, right=100))  # custom fill

# Multi-linear interpolation via RegularGridInterpolator (scipy)
# But NumPy alone: use np.searchsorted + manual weight computation
def linear_interp_manual(x_known, y_known, x_query):
    idx = np.searchsorted(x_known, x_query, side='right') - 1
    idx = np.clip(idx, 0, len(x_known) - 2)
    t = (x_query - x_known[idx]) / (x_known[idx+1] - x_known[idx])
    return y_known[idx] * (1 - t) + y_known[idx+1] * t

print(linear_interp_manual(x_known, y_known, np.array([1.5, 2.5])))
# [2.5 6.5]  (linear between known points)
```
</details>

<details>
<summary><strong>35. How do you apply functions along axes with np.apply_along_axis?</strong></summary>

**Answer:**
`apply_along_axis` applies a 1D function along a specified axis — convenient but slower than vectorized alternatives.

```python
import numpy as np

data = np.random.randn(5, 4)

# Apply along axis=0 (each column)
col_ranges = np.apply_along_axis(
    func1d=lambda col: col.max() - col.min(),
    axis=0,
    arr=data
)
print(col_ranges.shape)   # (4,)

# Apply along axis=1 (each row)
row_norms = np.apply_along_axis(np.linalg.norm, 1, data)
print(row_norms.shape)    # (5,)

# Vectorized alternatives are faster:
# Instead of apply_along_axis, prefer:
print(data.max(axis=0) - data.min(axis=0))   # 100x faster

# apply_over_axes: apply reduction over multiple axes
result = np.apply_over_axes(np.sum, data, axes=[0, 1])
print(result)   # total sum, shape (1,1)

# frompyfunc: create ufunc from Python function
safe_log = np.frompyfunc(lambda x: np.log(x) if x > 0 else np.nan, 1, 1)
arr = np.array([-1, 0, 1, np.e, 10])
print(safe_log(arr).astype(float))

# vectorize: similar to frompyfunc but with dtype support
vfunc = np.vectorize(lambda x, y: x + y if x > 0 else y)
print(vfunc(np.array([-1, 2]), np.array([3, 4])))  # [3 6]
```
</details>

<details>
<summary><strong>36. How do you compute numerical derivatives and integrals?</strong></summary>

**Answer:**
NumPy provides finite differences for derivatives and the trapezoid rule for integration.

```python
import numpy as np

x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

# Numerical derivative: np.gradient (central differences)
dy = np.gradient(y, x)              # dy/dx
print(np.allclose(dy, np.cos(x), atol=0.01))  # True (approx cosine)

# 2D gradient
Z = np.random.randn(10, 10)
dZ_dy, dZ_dx = np.gradient(Z)       # gradient in each direction

# Forward difference manually
dx = x[1] - x[0]
dy_forward = np.diff(y) / dx        # shape (999,) — one shorter

# Numerical integration: trapezoid rule
integral = np.trapz(y, x)           # integral of sin(x) from 0 to 2pi ≈ 0
print(round(integral, 6))           # ~0.0

# Integral of sin from 0 to pi should be 2
x2 = np.linspace(0, np.pi, 1000)
print(np.trapz(np.sin(x2), x2))    # ~2.0

# Cumulative integral (running sum)
cumulative = np.cumsum(y) * dx
# Or more accurately:
from numpy import trapz
cumulative_trap = np.array([trapz(y[:i+1], x[:i+1]) for i in range(len(x))])
```
</details>

<details>
<summary><strong>37. How do you work with complex numbers in NumPy?</strong></summary>

**Answer:**
NumPy fully supports complex arithmetic with `complex64` and `complex128` dtypes.

```python
import numpy as np

# Create complex arrays
z1 = np.array([1+2j, 3+4j, 5+6j])
z2 = np.array([1j, 2+0j, 0+3j])

# Arithmetic
print(z1 + z2)       # element-wise
print(z1 * z2)       # complex multiplication
print(z1 / z2)       # complex division

# Components
print(z1.real)       # [1. 3. 5.]
print(z1.imag)       # [2. 4. 6.]
print(np.abs(z1))    # [2.236 5.0 7.81]  magnitude = sqrt(re^2 + im^2)
print(np.angle(z1))  # phase angle in radians
print(np.conj(z1))   # conjugate: [1-2j, 3-4j, 5-6j]

# Polar form
r     = np.abs(z1)
theta = np.angle(z1)
z_reconstructed = r * np.exp(1j * theta)
print(np.allclose(z1, z_reconstructed))   # True

# Complex dtype
z = np.zeros(5, dtype=complex)   # complex128 by default
z = np.zeros(5, dtype=np.complex64)

# FFT returns complex output
signal = np.sin(2 * np.pi * np.linspace(0, 1, 100))
spectrum = np.fft.fft(signal)    # complex array
print(spectrum.dtype)            # complex128
```
</details>

<details>
<summary><strong>38. How do you profile and benchmark NumPy code?</strong></summary>

**Answer:**
Use `timeit`, `%timeit` in Jupyter, and avoid common anti-patterns.

```python
import numpy as np
import time

# timeit module
import timeit

arr = np.random.randn(1_000_000)

# Compare approaches
t1 = timeit.timeit(lambda: sum(arr),         number=10) / 10  # Python sum
t2 = timeit.timeit(lambda: np.sum(arr),      number=10) / 10  # NumPy sum
t3 = timeit.timeit(lambda: arr.sum(),        number=10) / 10  # method form
print(f"Python sum: {t1*1000:.2f}ms")
print(f"np.sum:     {t2*1000:.4f}ms")
print(f"arr.sum():  {t3*1000:.4f}ms")

# Memory profiling
import sys
py_list = list(range(1_000_000))
np_arr  = np.arange(1_000_000)
print(f"Python list: {sys.getsizeof(py_list) / 1e6:.1f} MB")
print(f"NumPy array: {np_arr.nbytes / 1e6:.1f} MB")

# Common performance anti-patterns:

# BAD: loop over elements
def slow(arr): return [x**2 for x in arr]

# GOOD: vectorized
def fast(arr): return arr**2

# BAD: repeated concatenation
def slow_concat(n):
    result = np.array([])
    for i in range(n): result = np.append(result, i)  # O(n^2)!
    return result

# GOOD: pre-allocate
def fast_concat(n):
    result = np.empty(n)
    for i in range(n): result[i] = i
    return result

# BETTER: vectorized
def best_concat(n): return np.arange(n)
```
</details>

<details>
<summary><strong>39. How does NumPy integrate with pandas?</strong></summary>

**Answer:**
pandas is built on NumPy; you can convert freely between DataFrames and arrays.

```python
import numpy as np
import pandas as pd

# DataFrame from NumPy
data = np.random.randn(100, 4)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

# Extract underlying NumPy array
arr = df.values           # deprecated but works
arr = df.to_numpy()       # preferred
arr = df['A'].to_numpy()  # single column

# Apply NumPy function to DataFrame
df['A_scaled'] = (df['A'] - df['A'].mean()) / df['A'].std()

# NumPy operations on DataFrame columns
corr = np.corrcoef(df[['A', 'B']].to_numpy().T)   # correlation matrix

# Boolean indexing (same pattern)
mask = df['A'] > 0
print(df[mask].shape)            # rows where A > 0
print(df.to_numpy()[mask.to_numpy()])  # same via NumPy

# GroupBy -> NumPy aggregation
groups = df.groupby(df['A'] > 0)['B'].apply(lambda x: np.percentile(x, 75))

# Create Series/DataFrame from array
idx = np.array(['x', 'y', 'z'])
s = pd.Series(np.array([1, 2, 3]), index=idx)

# Memory layout note: pandas may not always be C-contiguous
arr = df.to_numpy()
if not arr.flags['C_CONTIGUOUS']:
    arr = np.ascontiguousarray(arr)
```
</details>

<details>
<summary><strong>40. What are common NumPy pitfalls and how do you avoid them?</strong></summary>

**Answer:**
Key pitfalls: mutable default arguments, silent overflow, unexpected views vs copies, and floating-point comparisons.

```python
import numpy as np

# Pitfall 1: Integer overflow (silent!)
a = np.array([200], dtype=np.int8)
print(a * 2)   # [-56]  wraps around!
# Fix: use larger dtype
a = np.array([200], dtype=np.int32)
print(a * 2)   # [400]

# Pitfall 2: Modifying a view unexpectedly
arr = np.arange(10)
sub = arr[2:5]
sub[:] = 0       # modifies original!
print(arr)       # [0 0 0 0 0 5 6 7 8 9]
# Fix: sub = arr[2:5].copy()

# Pitfall 3: Floating-point comparison
a = 0.1 + 0.2
print(a == 0.3)           # False!
print(np.isclose(a, 0.3)) # True  — use this
print(np.allclose(np.array([0.1, 0.2]), np.array([0.1, 0.2])))  # True

# Pitfall 4: Concatenate in loop (O(n^2))
# BAD:
result = np.array([])
for i in range(100): result = np.append(result, i)

# GOOD: collect in list, convert once
parts = []
for i in range(100): parts.append(i)
result = np.array(parts)

# Pitfall 5: Boolean vs integer indexing
arr = np.array([10, 20, 30, 40])
print(arr[[True, False, True, False]])  # [10 30]  boolean fancy indexing
print(arr[[1, 0, 1, 0]])               # [20 10 20 10]  integer fancy indexing

# Pitfall 6: axis=None by default
data = np.arange(12).reshape(3, 4)
print(data.sum())          # 66 (all elements)
print(data.sum(axis=0))    # [12 15 18 21]  column sums
print(data.sum(axis=1))    # [6 22 38]  row sums

# Pitfall 7: Forgetting that np.random is not reproducible without seed
np.random.seed(42)  # fix seed for reproducibility
# Modern way:
rng = np.random.default_rng(42)
data = rng.standard_normal(1000)
```
</details>

<details>
<summary><strong>41. Stride tricks and sliding windows</strong></summary>

```python

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Efficient sliding window without copy
arr = np.array([1, 2, 3, 4, 5, 6, 7])
windows = sliding_window_view(arr, window_shape=3)
print(windows)
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]]

rolling_mean = windows.mean(axis=1)   # O(n) rolling mean

# Manual stride tricks (low-level)
from numpy.lib.stride_tricks import as_strided
arr = np.arange(10, dtype=float)
# Create (4,3) sliding window using strides
window_shape = (4, 3)
strides = (arr.strides[0], arr.strides[0])  # stride by 1 element in both dims
view = as_strided(arr, shape=window_shape, strides=strides)
print(view)  # read-only window; careful — write can corrupt memory
```
</details>

<details>
<summary><strong>42. np.lib.stride_tricks for convolution</strong></summary>

```python

import numpy as np

def conv1d_manual(signal, kernel):
    k = len(kernel)
    n = len(signal) - k + 1
    windows = np.lib.stride_tricks.sliding_window_view(signal, k)
    return windows @ kernel[::-1]   # vectorized dot product

signal = np.array([1., 2., 3., 4., 5., 6.])
kernel = np.array([0.25, 0.5, 0.25])   # smoothing kernel
print(conv1d_manual(signal, kernel))    # [2. 3. 4. 5.]
```
</details>

<details>
<summary><strong>43. Sparse operations without scipy</strong></summary>

```python

import numpy as np

# COO format manually
rows = np.array([0, 1, 2])
cols = np.array([1, 2, 0])
vals = np.array([5., 3., 2.])
shape = (3, 3)

# Dense reconstruction
dense = np.zeros(shape)
dense[rows, cols] = vals

# Sparse matrix-vector multiply
x = np.array([1., 2., 3.])
result = np.zeros(shape[0])
np.add.at(result, rows, vals * x[cols])
```
</details>

<details>
<summary><strong>44. np.ufunc methods: reduce, accumulate, outer</strong></summary>

```python

import numpy as np
arr = np.array([1, 2, 3, 4, 5])

# reduce: collapse to scalar using ufunc
print(np.add.reduce(arr))          # 15  (sum)
print(np.multiply.reduce(arr))     # 120 (product)
print(np.maximum.reduce(arr))      # 5   (max)

# accumulate: cumulative ufunc
print(np.add.accumulate(arr))      # [1  3  6 10 15]  (cumsum)
print(np.multiply.accumulate(arr)) # [1  2  6 24 120] (cumprod)

# outer: all combinations
print(np.add.outer([1,2,3], [10,20]))
# [[11 21]
#  [12 22]
#  [13 23]]
```
</details>

<details>
<summary><strong>45. Vectorizing distance computations</strong></summary>

```python

import numpy as np

# Euclidean distance matrix: all pairs
X = np.random.randn(100, 5)   # 100 points in 5D

# Efficient via broadcasting
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (100,100,5)
dist_matrix = np.sqrt((diff**2).sum(axis=-1))       # (100,100)

# Even faster: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
sq_norms = (X**2).sum(axis=1)                             # (100,)
dist2 = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
dist_matrix2 = np.sqrt(np.maximum(dist2, 0))              # clip numerical errors
```
</details>

<details>
<summary><strong>46. PCA from scratch using NumPy</strong></summary>

```python

import numpy as np

def pca(X, n_components=2):
    # Center
    X_centered = X - X.mean(axis=0)
    # Covariance matrix
    cov = np.cov(X_centered.T)
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Project
    components = eigenvectors[:, :n_components]
    X_pca = X_centered @ components
    explained_var = eigenvalues[:n_components] / eigenvalues.sum()
    return X_pca, explained_var

X = np.random.randn(200, 5)
X_2d, var = pca(X, n_components=2)
print(X_2d.shape)    # (200, 2)
print(var.sum())     # fraction of variance explained
```
</details>

<details>
<summary><strong>47. One-hot encoding with NumPy</strong></summary>

```python

import numpy as np

labels = np.array([0, 2, 1, 2, 0])
n_classes = 3

# Method 1: fancy indexing
ohe = np.zeros((len(labels), n_classes))
ohe[np.arange(len(labels)), labels] = 1

# Method 2: eye indexing (one-liner)
ohe2 = np.eye(n_classes)[labels]
print(ohe2)
# [[1 0 0]
#  [0 0 1]
#  [0 1 0]
#  [0 0 1]
#  [1 0 0]]
```
</details>

<details>
<summary><strong>48. Softmax and log-sum-exp (numerically stable)</strong></summary>

```python

import numpy as np

def softmax(x):
    x_shifted = x - x.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(x_shifted)
    return e / e.sum(axis=-1, keepdims=True)

def log_softmax(x):
    x_shifted = x - x.max(axis=-1, keepdims=True)
    return x_shifted - np.log(np.exp(x_shifted).sum(axis=-1, keepdims=True))

logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
print(softmax(logits))
print(softmax(logits).sum(axis=1))   # [1. 1.]  each row sums to 1
```
</details>

<details>
<summary><strong>49. Sliding statistics for time series</strong></summary>

```python

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

prices = np.random.randn(252).cumsum() + 100   # simulated price series
window = 20

windows = sliding_window_view(prices, window)   # (233, 20)
rolling_mean = windows.mean(axis=1)
rolling_std  = windows.std(axis=1)
rolling_max  = windows.max(axis=1)

# Bollinger Bands
upper = rolling_mean + 2 * rolling_std
lower = rolling_mean - 2 * rolling_std
```
</details>

<details>
<summary><strong>50. np.random Generator API (modern)</strong></summary>

```python

import numpy as np

# Modern: Generator object (reproducible, better stats)
rng = np.random.default_rng(seed=42)

# Distributions
print(rng.standard_normal(5))
print(rng.integers(0, 10, size=5))
print(rng.uniform(0, 1, size=5))
print(rng.choice(np.arange(10), size=5, replace=False))

# Shuffle and permutation
arr = np.arange(10)
rng.shuffle(arr)   # in-place
print(rng.permutation(10))   # returns new shuffled array

# Multivariate
mean = [0, 0]
cov  = [[1, 0.8], [0.8, 1]]
samples = rng.multivariate_normal(mean, cov, size=1000)
print(np.corrcoef(samples.T))   # ~[[1, 0.8], [0.8, 1]]
```
</details>

<details>
<summary><strong>51. Batch matrix operations</strong></summary>

```python

import numpy as np

# Batched matrix multiply (3D arrays)
A = np.random.randn(10, 3, 4)   # batch of 10 matrices (3x4)
B = np.random.randn(10, 4, 5)   # batch of 10 matrices (4x5)

# matmul supports broadcasting
C = A @ B                        # (10, 3, 5) — batched matmul
C2 = np.matmul(A, B)             # same

# Batched inverse (using linalg)
M = np.random.randn(5, 3, 3)
M_inv = np.linalg.inv(M)         # (5, 3, 3)

# Batched dot product (einsum)
result = np.einsum('bij,bij->b', A[:, :3, :3], M)  # batched trace-like
```
</details>

<details>
<summary><strong>52. Polynomial regression pipeline</strong></summary>

```python

import numpy as np

# Generate data
x = np.linspace(-3, 3, 50)
y = 2*x**2 - x + np.random.randn(50)

# Vandermonde matrix for degree-d polynomial
def poly_features(x, degree):
    return np.vstack([x**i for i in range(degree+1)]).T

X = poly_features(x, 2)   # (50, 3): [1, x, x^2]
# Solve via least squares
coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
print(coeffs)   # ~ [0, -1, 2]  (intercept, x, x^2 coefficients)

y_pred = X @ coeffs
mse = np.mean((y - y_pred)**2)
```
</details>

<details>
<summary><strong>53. Sparse matrix encoding</strong></summary>

```python

import numpy as np

# CSR format: efficient row slicing
row_ptr = np.array([0, 2, 3, 5])   # row i has entries row_ptr[i]:row_ptr[i+1]
col_idx = np.array([0, 2, 1, 0, 3])
values  = np.array([1., 2., 3., 4., 5.])

# Reconstruct dense row
def get_row(row_ptr, col_idx, values, n_cols, i):
    row = np.zeros(n_cols)
    row[col_idx[row_ptr[i]:row_ptr[i+1]]] = values[row_ptr[i]:row_ptr[i+1]]
    return row

print(get_row(row_ptr, col_idx, values, 4, 0))  # [1. 0. 2. 0.]
```
</details>

<details>
<summary><strong>54. N-dimensional indexing</strong></summary>

```python

import numpy as np

# 3D array indexing
arr = np.arange(24).reshape(2, 3, 4)

# Slice: first matrix, all rows, columns 1-3
print(arr[0, :, 1:3])   # (3, 2)

# Boolean on 3D
mask = arr > 10
print(arr[mask])         # 1D array of values > 10

# np.ndindex: iterate over all indices
for idx in np.ndindex(2, 3):
    pass   # idx = (0,0), (0,1), ... (1,2)

# np.unravel_index: convert flat index to multi-dim
flat_idx = arr.argmax()
multi_idx = np.unravel_index(flat_idx, arr.shape)
print(multi_idx, arr[multi_idx])   # position and value of max
```
</details>

<details>
<summary><strong>55. Memory-efficient operations with out parameter</strong></summary>

```python

import numpy as np

a = np.random.randn(1_000_000)
b = np.random.randn(1_000_000)
out = np.empty_like(a)

# Avoid creating temporary arrays
np.add(a, b, out=out)       # a + b stored directly in out
np.multiply(a, 2, out=out)  # in-place style via ufunc out=
np.sqrt(np.abs(a), out=out) # chain: abs first, then sqrt

# Pre-allocate for repeated operations
results = np.empty((100, 1_000_000))
for i in range(100):
    np.multiply(a, i, out=results[i])
```
</details>

<details>
<summary><strong>56. Implementing KMeans with NumPy</strong></summary>

```python

import numpy as np

def kmeans(X, k, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    # Initialize centroids randomly from data
    centroids = X[rng.choice(len(X), k, replace=False)]
    
    for _ in range(max_iter):
        # Assign: squared distances to each centroid
        dist2 = ((X[:, np.newaxis] - centroids[np.newaxis])**2).sum(axis=2)
        labels = dist2.argmin(axis=1)   # (n_samples,)
        
        # Update centroids
        new_centroids = np.array([X[labels==j].mean(axis=0) for j in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

X = np.vstack([np.random.randn(50, 2) + c for c in [[0,0], [5,5], [0,5]]])
labels, centers = kmeans(X, k=3)
```
</details>

<details>
<summary><strong>57. Window functions for signal processing</strong></summary>

```python

import numpy as np

N = 64
# Common window functions
rect    = np.ones(N)
hann    = np.hanning(N)    # 0.5*(1 - cos(2pi*n/(N-1)))
hamming = np.hamming(N)
blackman= np.blackman(N)   # best sidelobe suppression
bartlett= np.bartlett(N)   # triangular

# Apply window before FFT
signal = np.sin(2*np.pi*5*np.linspace(0,1,N)) + 0.5*np.sin(2*np.pi*12*np.linspace(0,1,N))
windowed = signal * hann
spectrum = np.abs(np.fft.rfft(windowed))
freqs    = np.fft.rfftfreq(N, d=1/N)
```
</details>

<details>
<summary><strong>58. Gram-Schmidt orthogonalization</strong></summary>

```python

import numpy as np

def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=float)
    for i in range(A.shape[1]):
        v = A[:, i].astype(float)
        for j in range(i):
            v -= np.dot(Q[:, j], A[:, i]) * Q[:, j]
        norm = np.linalg.norm(v)
        Q[:, i] = v / norm if norm > 1e-12 else v
    return Q

A = np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])
Q = gram_schmidt(A)
print(np.allclose(Q.T @ Q, np.eye(3)))   # True (orthonormal)
```
</details>

<details>
<summary><strong>59. Confusion matrix and classification metrics</strong></summary>

```python

import numpy as np

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm

y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 2, 0, 0, 1])
cm = confusion_matrix(y_true, y_pred, 3)
print(cm)

# Metrics from confusion matrix
def classification_report_np(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1
```
</details>

<details>
<summary><strong>60. Image operations with NumPy</strong></summary>

```python

import numpy as np

# Simulate grayscale image
img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

# Brightness and contrast
def adjust(img, brightness=0, contrast=1.0):
    img_f = img.astype(np.float32)
    return np.clip(img_f * contrast + brightness, 0, 255).astype(np.uint8)

# Channel separation for RGB
rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

# Grayscale: weighted average
gray = (0.299*R + 0.587*G + 0.114*B).astype(np.uint8)

# Flip/rotate
flipped_h = img[:, ::-1]           # horizontal flip
flipped_v = img[::-1, :]           # vertical flip
rotated   = np.rot90(img)          # 90 degrees CCW
rotated_k = np.rot90(img, k=2)    # 180 degrees

# Simple 2D convolution (correlation)
def conv2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = np.pad(img.astype(float), ((ph,ph),(pw,pw)), mode='reflect')
    h, w = img.shape
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return (windows * kernel).sum(axis=(-2,-1))

# Gaussian blur kernel
k = 5
x = np.linspace(-2, 2, k)
gauss_1d = np.exp(-x**2 / 2)
gauss_1d /= gauss_1d.sum()
kernel = np.outer(gauss_1d, gauss_1d)
blurred = conv2d(img, kernel)
```
</details>

<details>
<summary><strong>61. Batch normalization from scratch</strong></summary>

```python

import numpy as np

def batch_norm(X, gamma, beta, eps=1e-5):
    mu  = X.mean(axis=0)
    var = X.var(axis=0)
    X_hat = (X - mu) / np.sqrt(var + eps)
    return gamma * X_hat + beta, mu, var

X = np.random.randn(32, 64)   # batch of 32, 64 features
gamma = np.ones(64)
beta  = np.zeros(64)
out, mu, var = batch_norm(X, gamma, beta)
print(out.mean(axis=0).max())   # ~0
print(out.std(axis=0).min())    # ~1
```
</details>

<details>
<summary><strong>62. Attention mechanism from scratch</strong></summary>

```python

import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # (seq, seq)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)  # softmax
    return weights @ V                      # (seq, d_v)

seq_len, d_k, d_v = 10, 64, 64
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)
out = scaled_dot_product_attention(Q, K, V)
print(out.shape)   # (10, 64)
```
</details>

<details>
<summary><strong>63. Moving average and exponential smoothing</strong></summary>

```python

import numpy as np

data = np.random.randn(100).cumsum()

# Simple moving average
def sma(arr, window):
    return sliding_window_view(arr, window).mean(axis=1)

from numpy.lib.stride_tricks import sliding_window_view
sma_20 = sma(data, 20)   # shape (81,)

# Exponential moving average (EMA)
def ema(arr, alpha=0.1):
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

ema_values = ema(data, alpha=0.1)
```
</details>

<details>
<summary><strong>64. np.linalg.lstsq for overdetermined systems</strong></summary>

```python

import numpy as np

# Linear regression: solve Xw = y (overdetermined)
X = np.column_stack([np.ones(50), np.random.randn(50)])   # design matrix
true_w = np.array([2.0, 3.0])
y = X @ true_w + np.random.randn(50) * 0.5

# Least squares solution
w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
print(w)   # ~ [2. 3.]

# Equivalent: normal equations
w_normal = np.linalg.inv(X.T @ X) @ X.T @ y
# (lstsq more numerically stable via SVD)

# R-squared
y_pred = X @ w
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot
print(f"R² = {r2:.4f}")
```
</details>

<details>
<summary><strong>65. Cholesky decomposition for fast sampling</strong></summary>

```python

import numpy as np

# Sample from multivariate Gaussian using Cholesky
def multivariate_normal_sample(mean, cov, n_samples, seed=42):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)     # cov = L @ L.T
    z = rng.standard_normal((len(mean), n_samples))
    return (L @ z).T + mean

mean = np.array([1.0, 2.0])
cov  = np.array([[2.0, 1.0], [1.0, 3.0]])
samples = multivariate_normal_sample(mean, cov, 1000)
print(samples.mean(axis=0))   # ~ [1., 2.]
print(np.cov(samples.T))      # ~ [[2, 1], [1, 3]]
```
</details>

<details>
<summary><strong>66. np.pad for convolutions and images</strong></summary>

```python

import numpy as np

arr = np.array([[1, 2], [3, 4]])

# Constant padding (zero-padding)
print(np.pad(arr, 1, mode='constant', constant_values=0))
# [[0 0 0 0]
#  [0 1 2 0]
#  [0 3 4 0]
#  [0 0 0 0]]

# Reflection padding (better for images)
print(np.pad(arr, 1, mode='reflect'))

# Edge/replicate padding
print(np.pad(arr, 1, mode='edge'))

# Wrap-around (circular)
print(np.pad(arr, 1, mode='wrap'))

# Different pad widths per axis: ((top, bottom), (left, right))
print(np.pad(arr, ((1, 2), (0, 1)), mode='constant'))
```
</details>

<details>
<summary><strong>67. np.where for conditional logic</strong></summary>

```python

import numpy as np

x = np.linspace(-3, 3, 100)

# ReLU
relu = np.where(x > 0, x, 0)

# Leaky ReLU
leaky = np.where(x > 0, x, 0.01 * x)

# Clip (same as np.clip)
clipped = np.where(x > 2, 2, np.where(x < -2, -2, x))

# Multi-condition (nested np.where)
labels = np.where(x < -1, 'low', np.where(x > 1, 'high', 'mid'))

# np.select: cleaner for multiple conditions
conditions = [x < -1, x > 1]
choices    = [-1, 1]
result = np.select(conditions, choices, default=0)
```
</details>

<details>
<summary><strong>68. Vectorized string operations with np.char</strong></summary>

```python

import numpy as np

arr = np.array(['Hello', 'World', 'NumPy'])

print(np.char.upper(arr))         # ['HELLO' 'WORLD' 'NUMPY']
print(np.char.lower(arr))         # ['hello' 'world' 'numpy']
print(np.char.add(arr, '!'))      # ['Hello!' 'World!' 'NumPy!']
print(np.char.startswith(arr, 'H'))  # [True False False]
print(np.char.find(arr, 'o'))     # [4 1 -1]  index of 'o' (-1 if not found)
print(np.char.count(arr, 'l'))    # [2 1 0]
print(np.char.replace(arr, 'l', 'L'))  # ['HeLLo' 'WorLd' 'NumPy']
```
</details>

<details>
<summary><strong>69. Numerical stability techniques</strong></summary>

```python

import numpy as np

# Log-sum-exp trick (avoids overflow in softmax)
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

x = np.array([1000., 1001., 1002.])
print(np.exp(x))           # overflow!
print(logsumexp(x))        # 1002.408...  stable

# Stable variance computation (Welford's online algorithm)
def welford_variance(data):
    n = 0; mean = 0.0; M2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    return M2 / n if n > 1 else 0.0

# vs naive (can have catastrophic cancellation for large numbers)
data = np.array([1e8 + 1, 1e8 + 2, 1e8 + 3])
print(np.var(data))             # 0.666... (correct)
print(welford_variance(data))   # 0.666... (stable)
```
</details>

<details>
<summary><strong>70. Building neural network forward pass with NumPy</strong></summary>

```python

import numpy as np

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forward(X, weights):
    h = X
    for W, b, activation in weights:
        h = activation(h @ W + b)
    return h

# Initialize weights
np.random.seed(42)
layers = [
    (np.random.randn(4, 8) * 0.1, np.zeros(8),  relu),
    (np.random.randn(8, 4) * 0.1, np.zeros(4),  relu),
    (np.random.randn(4, 1) * 0.1, np.zeros(1),  sigmoid),
]

X = np.random.randn(100, 4)
preds = forward(X, layers)
print(preds.shape)   # (100, 1)
print(preds.min(), preds.max())   # in [0,1] from sigmoid
```
</details>

---

## Advanced Questions (71-100)

<details>
<summary><strong>71. Explain structured arrays and record arrays.</strong></summary>

**Answer:**
Structured arrays allow heterogeneous data types per field.

```python
import numpy as np

# Define structured dtype
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f8')])
data = np.array([('Alice', 25, 60.5), ('Bob', 30, 75.0)], dtype=dt)

print(data['name'])    # ['Alice' 'Bob']
print(data['age'])     # [25 30]
print(data[0])         # ('Alice', 25, 60.5)

# Record arrays (easier access)
records = np.rec.array([('Alice', 25, 60.5), ('Bob', 30, 75.0)],
                       dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f8')])
print(records.name)    # ['Alice' 'Bob']
```

**Interview Tip**: Useful for tabular data, but pandas is often better.
</details>

<details>
<summary><strong>72. How do you implement gradient descent with NumPy?</strong></summary>

**Answer:**
Pure NumPy lets you implement optimizers from scratch for deep understanding.

```python
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradient_descent(X, y, lr=0.01, n_iter=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for _ in range(n_iter):
        y_pred = X @ w + b
        error  = y_pred - y
        # Gradients
        dw = (2 / n) * X.T @ error
        db = (2 / n) * error.sum()
        # Update
        w -= lr * dw
        b -= lr * db
        losses.append(mse_loss(y, y_pred))

    return w, b, losses

# Adam optimizer from scratch
def adam(grad_fn, w0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, n_iter=100):
    w = w0.copy()
    m = np.zeros_like(w)   # first moment
    v = np.zeros_like(w)   # second moment
    for t in range(1, n_iter + 1):
        g = grad_fn(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)   # bias correction
        v_hat = v / (1 - beta2**t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w
```
</details>

<details>
<summary><strong>73. How do you implement convolution from scratch?</strong></summary>

**Answer:**
Manual 2D convolution using stride tricks — same as what CNNs do internally.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def conv2d_valid(image, kernel):
    kh, kw = kernel.shape
    windows = sliding_window_view(image, (kh, kw))   # (oh, ow, kh, kw)
    return np.einsum('ijkl,kl->ij', windows, kernel)

def conv2d_same(image, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='constant')
    return conv2d_valid(padded, kernel)

# Sobel edge detection
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
sobel_y = sobel_x.T

img = np.random.randint(0, 256, (64, 64), dtype=np.uint8).astype(float)
edges_x = conv2d_same(img, sobel_x)
edges_y = conv2d_same(img, sobel_y)
edges   = np.hypot(edges_x, edges_y)   # magnitude
```
</details>

<details>
<summary><strong>74. How do you use np.linalg.eigh vs eig?</strong></summary>

**Answer:**
`eigh` is for symmetric/Hermitian matrices — faster and more numerically stable than `eig`.

```python
import numpy as np

# Symmetric positive definite matrix
A = np.random.randn(5, 5)
S = A.T @ A   # symmetric PD

# eigh: assumes symmetric, returns REAL eigenvalues sorted ascending
eigenvalues_h, eigenvectors_h = np.linalg.eigh(S)
print(eigenvalues_h)        # real, sorted ascending
print(np.all(eigenvalues_h >= 0))   # True (PD)

# eig: general, returns COMPLEX eigenvalues (unordered)
eigenvalues_g, eigenvectors_g = np.linalg.eig(S)
print(eigenvalues_g.imag.max())     # ~0 (imaginary part noise)

# For PCA/spectral methods: always use eigh
# 10x faster, numerically stable for symmetric matrices

# Verify eigenvectors are orthonormal (eigh guarantees this)
print(np.allclose(eigenvectors_h.T @ eigenvectors_h, np.eye(5)))   # True

# Get top-k eigenvectors (descending order)
k = 2
top_k = eigenvectors_h[:, -k:][:, ::-1]   # eigh returns ascending
print(top_k.shape)   # (5, 2)
```
</details>

<details>
<summary><strong>75. How do you compute the pseudo-inverse?</strong></summary>

**Answer:**
`np.linalg.pinv` computes the Moore-Penrose pseudo-inverse using SVD — works for non-square and rank-deficient matrices.

```python
import numpy as np

# Overdetermined system (more equations than unknowns)
A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
b = np.array([2, 4, 5], dtype=float)

# Least-squares solution via pseudo-inverse
A_pinv = np.linalg.pinv(A)        # shape (2, 3)
x = A_pinv @ b                    # least-squares solution
print(x)

# Equivalent to lstsq
x2, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print(np.allclose(x, x2))   # True

# Singular matrix (rank deficient)
S = np.array([[1., 2.], [2., 4.]])   # rank 1
print(np.linalg.matrix_rank(S))      # 1
S_pinv = np.linalg.pinv(S)
print(S @ S_pinv @ S)               # should equal S (pinv property)

# Relation to SVD
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
threshold = 1e-10
sigma_inv = np.where(sigma > threshold, 1/sigma, 0)
pinv_manual = Vt.T @ np.diag(sigma_inv) @ U.T
print(np.allclose(A_pinv, pinv_manual))   # True
```
</details>

<details>
<summary><strong>76. How do you implement cross-validation splits with NumPy?</strong></summary>

**Answer:**
Manual k-fold cross-validation using index manipulation.

```python
import numpy as np

def k_fold_indices(n, k, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx

# Use it
X = np.random.randn(100, 5)
y = np.random.randn(100)

scores = []
for train_idx, val_idx in k_fold_indices(len(X), k=5):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # fit/predict...
    scores.append(np.mean((y_val - y_val.mean())**2))   # dummy score

print(f"CV MSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Stratified split for classification
def stratified_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)
```
</details>

<details>
<summary><strong>77. How do you handle large arrays with memory mapping?</strong></summary>

**Answer:**
`np.memmap` maps a file on disk to a NumPy array — enables working with data larger than RAM.

```python
import numpy as np

# Create large memory-mapped array (writes to disk)
shape = (10_000, 1_000)   # 80MB for float64
mmap = np.memmap('large_data.dat', dtype='float64', mode='w+', shape=shape)

# Write in chunks (doesn't load all into RAM)
chunk = 1000
for i in range(0, shape[0], chunk):
    mmap[i:i+chunk] = np.random.randn(chunk, shape[1])
    mmap.flush()   # write to disk

del mmap   # close file

# Read back (lazy — only loads accessed pages)
mmap_read = np.memmap('large_data.dat', dtype='float64', mode='r', shape=shape)
print(mmap_read[0, :5])    # only loads page containing row 0
print(mmap_read.mean())    # computes in chunks via OS paging

# Column mean without loading all data
col_means = np.empty(shape[1])
for i in range(0, shape[0], chunk):
    col_means += mmap_read[i:i+chunk].sum(axis=0)
col_means /= shape[0]

# Use mode='r+' to read and write existing file
# Use mode='c' (copy-on-write) for read with local modifications
```
</details>

<details>
<summary><strong>78. How do you use np.vectorize and when is it NOT faster?</strong></summary>

**Answer:**
`np.vectorize` is syntactic sugar — it loops internally and is no faster than a Python loop. Use it only for convenience.

```python
import numpy as np
import timeit

# Python function with if/else
def classify(x):
    if x < -1:   return 'low'
    elif x > 1:  return 'high'
    else:        return 'mid'

arr = np.random.randn(10_000)

# np.vectorize: convenient but NOT faster than loop
v_classify = np.vectorize(classify)
t1 = timeit.timeit(lambda: v_classify(arr), number=10)

# Vectorized with np.select: TRUE vectorization (much faster)
def classify_vec(arr):
    return np.select([arr < -1, arr > 1], ['low', 'high'], default='mid')
t2 = timeit.timeit(lambda: classify_vec(arr), number=10)

print(f"vectorize: {t1:.3f}s")
print(f"np.select: {t2:.3f}s")   # 10-50x faster

# When vectorize IS useful:
# - Functions returning variable-length outputs
# - Functions using Python builtins that have no ufunc equivalent
# - Rapid prototyping before optimizing

# Better alternatives:
# np.where, np.select, np.piecewise
result = np.piecewise(arr,
    [arr < -1, arr > 1],
    [lambda x: x**2, lambda x: x**3, lambda x: x])
```
</details>

<details>
<summary><strong>79. How do you implement a rolling window correlation?</strong></summary>

**Answer:**
Efficient rolling correlation using stride tricks and vectorized statistics.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def rolling_correlation(x, y, window):
    wx = sliding_window_view(x, window)   # (n-w+1, w)
    wy = sliding_window_view(y, window)

    # Vectorized: compute correlation for each window
    xm = wx - wx.mean(axis=1, keepdims=True)
    ym = wy - wy.mean(axis=1, keepdims=True)

    num = (xm * ym).sum(axis=1)
    den = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1)) + 1e-12
    return num / den

# Rolling z-score (normalize each window)
def rolling_zscore(x, window):
    windows = sliding_window_view(x, window)
    mu  = windows.mean(axis=1)
    std = windows.std(axis=1) + 1e-12
    return (x[window-1:] - mu) / std

np.random.seed(42)
x = np.cumsum(np.random.randn(200))
y = x + np.random.randn(200) * 0.5
corr = rolling_correlation(x, y, window=20)
print(corr[:5])   # rolling correlation values
```
</details>

<details>
<summary><strong>80. How do you implement matrix factorization for recommendations?</strong></summary>

**Answer:**
Alternating Least Squares (ALS) for collaborative filtering — a core recommendation algorithm.

```python
import numpy as np

def als_matrix_factorization(R, k=10, n_iter=20, reg=0.1, seed=42):
    """
    R: (n_users, n_items) ratings matrix with 0 for missing
    k: latent factors
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    U = rng.standard_normal((n_users, k)) * 0.1  # user factors
    V = rng.standard_normal((n_items, k)) * 0.1  # item factors

    mask = R > 0   # observed ratings

    for iteration in range(n_iter):
        # Fix V, update U
        for i in range(n_users):
            obs_items = np.where(mask[i])[0]
            if len(obs_items) == 0: continue
            Vi = V[obs_items]                    # (n_obs, k)
            ri = R[i, obs_items]                 # (n_obs,)
            A = Vi.T @ Vi + reg * np.eye(k)
            b = Vi.T @ ri
            U[i] = np.linalg.solve(A, b)

        # Fix U, update V
        for j in range(n_items):
            obs_users = np.where(mask[:, j])[0]
            if len(obs_users) == 0: continue
            Uj = U[obs_users]
            rj = R[obs_users, j]
            A = Uj.T @ Uj + reg * np.eye(k)
            b = Uj.T @ rj
            V[j] = np.linalg.solve(A, b)

        R_pred = U @ V.T
        mse = np.mean((R[mask] - R_pred[mask])**2)
        if iteration % 5 == 0:
            print(f"Iter {iteration}: MSE = {mse:.4f}")

    return U, V

# Simulate sparse ratings matrix
R = np.zeros((50, 100))
for i in range(50):
    items = np.random.choice(100, 20, replace=False)
    R[i, items] = np.random.randint(1, 6, 20).astype(float)

U, V = als_matrix_factorization(R, k=5, n_iter=10)
```
</details>

<details>
<summary><strong>81. Custom ufuncs with np.frompyfunc and np.vectorize</strong></summary>

```python

import numpy as np

# frompyfunc: wraps Python function as ufunc (returns object array)
safe_log = np.frompyfunc(lambda x: np.log(x) if x > 0 else np.nan, nin=1, nout=1)
arr = np.array([-1, 0, 1, np.e, 10])
print(safe_log(arr).astype(float))   # [nan nan 0.0 1.0 2.302]

# vectorize: like frompyfunc but with dtype control
safe_div = np.vectorize(lambda a, b: a/b if b != 0 else np.inf, otypes=[float])
print(safe_div(np.array([1,2,3]), np.array([0,2,1])))   # [inf 1. 3.]
```
</details>

<details>
<summary><strong>82. np.einsum for multi-head attention</strong></summary>

```python

import numpy as np

# Multi-head attention einsum patterns
batch, heads, seq, d_k = 2, 4, 10, 16
Q = np.random.randn(batch, heads, seq, d_k)
K = np.random.randn(batch, heads, seq, d_k)
V = np.random.randn(batch, heads, seq, d_k)

# Attention scores: Q @ K.T / sqrt(d_k)
scores = np.einsum('bhid,bhjd->bhij', Q, K) / np.sqrt(d_k)  # (batch, heads, seq, seq)

# Softmax (numerically stable)
scores -= scores.max(axis=-1, keepdims=True)
weights = np.exp(scores)
weights /= weights.sum(axis=-1, keepdims=True)

# Weighted sum of values
out = np.einsum('bhij,bhjd->bhid', weights, V)   # (batch, heads, seq, d_k)
print(out.shape)   # (2, 4, 10, 16)
```
</details>

<details>
<summary><strong>83. Sparse reward computation in RL</strong></summary>

```python

import numpy as np

def compute_discounted_returns(rewards, gamma=0.99):
    n = len(rewards)
    returns = np.zeros(n)
    G = 0
    for t in reversed(range(n)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# Vectorized version using stride tricks + cumsum trick
def discounted_returns_vec(rewards, gamma=0.99):
    n = len(rewards)
    gammas = gamma ** np.arange(n)
    # G[t] = sum_{k=0}^{n-t-1} gamma^k * r[t+k]
    returns = np.array([
        np.dot(rewards[t:], gammas[:n-t]) for t in range(n)
    ])
    return returns
```
</details>

<details>
<summary><strong>84. Histogram equalization for image enhancement</strong></summary>

```python

import numpy as np

def histogram_equalization(img):
    # img: 2D uint8 grayscale
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    lut = np.interp(np.arange(256), np.arange(256), cdf_normalized).astype(np.uint8)
    return lut[img]   # apply lookup table

img = np.random.randint(0, 128, (256, 256), dtype=np.uint8)  # dark image
eq_img = histogram_equalization(img)
print(eq_img.min(), eq_img.max())   # better contrast
```
</details>

<details>
<summary><strong>85. np.diff for feature engineering</strong></summary>

```python

import numpy as np

# Time series features
prices = np.array([100., 102., 101., 105., 103., 108.])

# Returns
returns = np.diff(prices) / prices[:-1]

# Momentum (rate of change over n periods)
n = 2
momentum = np.diff(prices, n=n)   # n-th order difference

# Log returns (preferred in finance)
log_returns = np.diff(np.log(prices))

# Acceleration
velocity     = np.diff(prices)
acceleration = np.diff(velocity)

# Pad to keep original length
returns_padded = np.concatenate([[np.nan], returns])
```
</details>

<details>
<summary><strong>86. Implementing DBSCAN density computation</strong></summary>

```python

import numpy as np

def dbscan_region_query(X, point_idx, eps):
    # Find all points within eps of point_idx
    dists = np.linalg.norm(X - X[point_idx], axis=1)
    return np.where(dists <= eps)[0]

def dbscan(X, eps=0.5, min_samples=5):
    n = len(X)
    labels = np.full(n, -1)   # -1 = noise
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1: continue
        neighbors = dbscan_region_query(X, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1   # noise
            continue
        labels[i] = cluster_id
        stack = list(neighbors)
        while stack:
            j = stack.pop()
            if labels[j] == -1: labels[j] = cluster_id
            if labels[j] != -1: continue
            labels[j] = cluster_id
            nb = dbscan_region_query(X, j, eps)
            if len(nb) >= min_samples:
                stack.extend(nb)
        cluster_id += 1
    return labels
```
</details>

<details>
<summary><strong>87. Matrix exponentiation for Markov chains</strong></summary>

```python

import numpy as np

# Transition matrix (rows sum to 1)
P = np.array([[0.9, 0.1],
              [0.2, 0.8]])

# State after n steps: initial_state @ P^n
def matrix_power(P, n):
    result = np.eye(len(P))
    while n > 0:
        if n % 2 == 1:
            result = result @ P
        P = P @ P
        n //= 2
    return result

# Stationary distribution: P^inf
P_inf = matrix_power(P, 1000)
print(P_inf[0])   # stationary: [0.667 0.333]

# Exact stationary: solve pi @ P = pi, sum(pi) = 1
A = (P.T - np.eye(len(P)))
A[-1] = 1   # replace last row with normalization constraint
b = np.zeros(len(P)); b[-1] = 1
pi = np.linalg.solve(A, b)
print(pi)   # [0.667 0.333]
```
</details>

<details>
<summary><strong>88. Wavelet-like multi-scale features</strong></summary>

```python

import numpy as np

def multi_scale_features(signal, scales=[4, 8, 16, 32]):
    features = [signal]
    for s in scales:
        kernel = np.ones(s) / s
        # Convolve (smooth) then downsample
        padded = np.pad(signal, s//2, mode='reflect')
        smoothed = np.convolve(padded, kernel, mode='valid')[:len(signal)]
        detail = signal - smoothed   # high-frequency component
        features.append(smoothed)
        features.append(detail)
    return np.stack(features, axis=1)   # (n, 1+2*len(scales))

signal = np.sin(np.linspace(0, 4*np.pi, 200)) + 0.3*np.random.randn(200)
X = multi_scale_features(signal)
print(X.shape)   # (200, 9)
```
</details>

<details>
<summary><strong>89. Solving linear recurrences vectorized</strong></summary>

```python

import numpy as np

# Fast Fibonacci using matrix exponentiation
def fib_matrix(n):
    M = np.array([[1, 1], [1, 0]], dtype=object)
    result = np.eye(2, dtype=object)
    while n > 0:
        if n % 2: result = result @ M
        M = M @ M
        n //= 2
    return int(result[0, 1])

# Vectorized: all Fibonacci up to n
def fibs(n):
    a = np.zeros(n+1, dtype=object)
    a[0] = 0; a[1] = 1
    for i in range(2, n+1):
        a[i] = a[i-1] + a[i-2]
    return a

print(fib_matrix(50))   # 12586269025
```
</details>

<details>
<summary><strong>90. Gradient computation using finite differences</strong></summary>

```python

import numpy as np

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus  = x.copy(); x_plus[i]  += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Test on a known function: f(x) = x1^2 + 2*x2^2
f = lambda x: x[0]**2 + 2*x[1]**2
x0 = np.array([3.0, 4.0])
grad = numerical_gradient(f, x0)
print(grad)   # [6.0, 16.0]  (analytic: [2*x1, 4*x2])

# Check gradient of neural network layer
def linear_layer(W, x, b):
    return W @ x + b

W = np.random.randn(3, 4)
x = np.random.randn(4)
b = np.random.randn(3)
f_layer = lambda w_flat: linear_layer(w_flat.reshape(3,4), x, b).sum()
grad_W = numerical_gradient(f_layer, W.ravel())
print(grad_W.reshape(3, 4))   # should match analytical gradient: outer(ones, x)
```
</details>

<details>
<summary><strong>91. Non-maximum suppression (NMS) for object detection</strong></summary>

```python

import numpy as np

def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4) array of [x1, y1, x2, y2]
    scores: (N,) confidence scores
    Returns indices of kept boxes.
    """
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep)
```
</details>

<details>
<summary><strong>92. Implement Naive Bayes from scratch</strong></summary>

```python

import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_  = np.array([np.mean(y==c) for c in self.classes_])
        self.means_   = np.array([X[y==c].mean(axis=0) for c in self.classes_])
        self.vars_    = np.array([X[y==c].var(axis=0) + 1e-9 for c in self.classes_])

    def _log_likelihood(self, X):
        log_ll = []
        for mu, var in zip(self.means_, self.vars_):
            ll = -0.5 * np.sum(np.log(2*np.pi*var) + (X-mu)**2/var, axis=1)
            log_ll.append(ll)
        return np.stack(log_ll, axis=1)   # (n, n_classes)

    def predict(self, X):
        log_posterior = self._log_likelihood(X) + np.log(self.priors_)
        return self.classes_[log_posterior.argmax(axis=1)]

X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
nb = GaussianNaiveBayes()
nb.fit(X[:150], y[:150])
print((nb.predict(X[150:]) == y[150:]).mean())   # accuracy
```
</details>

<details>
<summary><strong>93. Fast nearest neighbor search with kdtree ideas</strong></summary>

```python

import numpy as np

def brute_force_knn(X_train, X_query, k=5):
    # All-pairs squared distance: (a-b)^2 = a^2 + b^2 - 2ab
    sq_train = (X_train**2).sum(axis=1)                 # (n_train,)
    sq_query = (X_query**2).sum(axis=1)                 # (n_query,)
    dist2 = sq_query[:, None] + sq_train[None, :] - 2 * X_query @ X_train.T
    dist2 = np.maximum(dist2, 0)                         # numerical clamp
    knn_idx = np.argpartition(dist2, k, axis=1)[:, :k]  # partial sort O(n)
    knn_dist = np.sqrt(dist2[np.arange(len(X_query))[:, None], knn_idx])
    # Sort the k results
    sort_order = knn_dist.argsort(axis=1)
    return knn_idx[np.arange(len(knn_idx))[:, None], sort_order]

X_train = np.random.randn(500, 10)
X_query = np.random.randn(20,  10)
neighbors = brute_force_knn(X_train, X_query, k=5)
print(neighbors.shape)   # (20, 5)
```
</details>

<details>
<summary><strong>94. Implementing TFIDF from scratch</strong></summary>

```python

import numpy as np
from collections import Counter

def tfidf(corpus):
    # corpus: list of token lists
    vocab = {w for doc in corpus for w in doc}
    vocab = sorted(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    n_docs, n_vocab = len(corpus), len(vocab)

    # TF: term frequency (normalized)
    tf = np.zeros((n_docs, n_vocab))
    for d, doc in enumerate(corpus):
        counts = Counter(doc)
        total = len(doc)
        for w, c in counts.items():
            tf[d, word2idx[w]] = c / total

    # IDF: log(N / df + 1)
    df = (tf > 0).sum(axis=0)
    idf = np.log((n_docs + 1) / (df + 1)) + 1   # sklearn smooth IDF

    tfidf_matrix = tf * idf
    # L2 normalize each document
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True) + 1e-12
    return tfidf_matrix / norms

corpus = [['cat', 'sat', 'mat'], ['dog', 'sat', 'floor'], ['cat', 'dog', 'sat']]
X = tfidf(corpus)
print(X.shape)   # (3, n_vocab)
print(np.allclose(np.linalg.norm(X, axis=1), 1))   # True, L2 normalized
```
</details>

<details>
<summary><strong>95. Kalman filter with NumPy</strong></summary>

```python

import numpy as np

def kalman_filter(observations, F, H, Q, R, x0, P0):
    """
    F: state transition, H: observation model
    Q: process noise, R: observation noise
    x0: initial state, P0: initial covariance
    """
    n_obs = len(observations)
    n_state = len(x0)
    
    filtered_states = np.zeros((n_obs, n_state))
    x, P = x0.copy(), P0.copy()

    for t, z in enumerate(observations):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update
        S = H @ P @ H.T + R         # innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
        x = x + K @ (z - H @ x)    # state update
        P = (np.eye(n_state) - K @ H) @ P  # covariance update
        filtered_states[t] = x

    return filtered_states
```
</details>

<details>
<summary><strong>96. Efficient rolling operations without pandas</strong></summary>

```python

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def rolling_stats(arr, window):
    windows = sliding_window_view(arr, window)   # (n-w+1, w)
    return {
        'mean':   windows.mean(axis=1),
        'std':    windows.std(axis=1),
        'min':    windows.min(axis=1),
        'max':    windows.max(axis=1),
        'median': np.median(windows, axis=1),
    }

prices = np.random.randn(500).cumsum() + 100
stats = rolling_stats(prices, window=20)
print(stats['mean'].shape)   # (481,)
```
</details>

<details>
<summary><strong>97. np.tensordot for arbitrary contractions</strong></summary>

```python

import numpy as np

A = np.random.randn(3, 4, 5)
B = np.random.randn(4, 5, 6)

# Contract over axes (1,2) of A with axes (0,1) of B
result = np.tensordot(A, B, axes=([1, 2], [0, 1]))
print(result.shape)   # (3, 6)

# Equivalent with einsum
result2 = np.einsum('ijk,jkl->il', A, B)
print(np.allclose(result, result2))   # True

# Outer product: no contraction
outer = np.tensordot(np.array([1,2,3]), np.array([4,5]), axes=0)
print(outer.shape)   # (3, 2) — outer product
```
</details>

<details>
<summary><strong>98. Simulating random processes</strong></summary>

```python

import numpy as np

rng = np.random.default_rng(42)

# Geometric Brownian Motion (stock prices)
def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_prices = np.cumsum(increments, axis=1)
    return S0 * np.exp(log_prices)

paths = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1, n_steps=252, n_paths=1000)
print(paths.shape)         # (1000, 252)
print(paths[:, -1].mean()) # expected: ~100 * exp(0.05*1) ≈ 105

# Monte Carlo option pricing (European call)
K = 105   # strike
r = 0.02  # risk-free rate
T = 1.0
payoffs = np.maximum(paths[:, -1] - K, 0)
option_price = np.exp(-r * T) * payoffs.mean()
print(f"Call price: {option_price:.4f}")
```
</details>

<details>
<summary><strong>99. Zero-copy data sharing between NumPy and bytes</strong></summary>

```python

import numpy as np

# Convert bytes to NumPy array (zero-copy)
data_bytes = b'\x01\x00\x02\x00\x03\x00\x04\x00'
arr = np.frombuffer(data_bytes, dtype=np.int16)
print(arr)   # [1 2 3 4]

# NumPy array to bytes (view, no copy)
arr = np.array([1, 2, 3, 4], dtype=np.int16)
bdata = arr.tobytes()   # copy (safe)
bdata2 = bytes(arr.data)  # also copy

# memoryview: true zero-copy
mv = memoryview(arr)
arr2 = np.frombuffer(mv, dtype=np.int16)
arr[0] = 99
print(arr2[0])   # 99 — shared memory!

# Useful for: network serialization, shared memory across processes
import multiprocessing
shm = multiprocessing.shared_memory.SharedMemory(create=True, size=arr.nbytes)
shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
shared_arr[:] = arr
```
</details>

<details>
<summary><strong>100. NumPy best practices and production tips</strong></summary>

```python

import numpy as np

# 1. Use appropriate dtypes (float32 for ML, saves memory and often faster on GPU)
X = np.random.randn(1000, 100).astype(np.float32)

# 2. Prefer views over copies; use .copy() only when needed
arr = np.arange(100)
sub = arr[10:20]        # view (fast)
sub_copy = arr[10:20].copy()  # copy (only if you need independence)

# 3. Pre-allocate output arrays
result = np.empty(1000, dtype=np.float32)
np.multiply(X[:, 0], X[:, 1], out=result[:1000])

# 4. Avoid Python loops: use vectorized ops, einsum, ufuncs
# 5. Use numpy.random.default_rng (new API) for reproducibility

# 6. Be careful with in-place ops on views
a = np.array([1., 2., 3.])
b = a[:2]
a += 1     # modifies b too if b is a view!

# 7. Use np.allclose for float comparison
assert np.allclose(0.1 + 0.2, 0.3)   # not ==

# 8. Profile with timeit before optimizing
# 9. Consider numba for jit-compiled loops when vectorization isn't possible
# from numba import njit
# @njit
# def fast_loop(arr): ...

# 10. Memory layout: ensure C-contiguous before passing to C extensions
arr = np.ascontiguousarray(arr)

print("NumPy essentials for ML:")
print("- np.einsum: tensor contractions")
print("- np.linalg: linear algebra")
print("- np.fft: frequency analysis")
print("- sliding_window_view: rolling operations")
print("- np.memmap: out-of-core data")
```
</details>
---

## NumPy Quick Reference

| Operation | Function | Complexity |
|-----------|----------|-----------|
| Sort | `np.sort`, `np.argsort` | O(n log n) |
| Search | `np.searchsorted` | O(log n) |
| FFT | `np.fft.fft` | O(n log n) |
| Matrix multiply | `@` or `np.matmul` | O(n³) |
| SVD | `np.linalg.svd` | O(min(m,n)²·max(m,n)) |
| Eigh | `np.linalg.eigh` | O(n³) |
| Rolling stats | `sliding_window_view` | O(n·w) |
| Histogram | `np.histogram` | O(n) |

## Summary in simple points

- Read shape, axes, dtype, strides and memory layout before reasoning about an array. Estimate allocations before broadcasting.
- Create, reshape, transpose, concatenate and index arrays with explicit view/copy expectations; advanced indexing usually allocates a copy.
- Use ufuncs, reductions, accumulations, outer operations and conditional selection to express numerical work clearly.
- Separate elementwise multiplication from matrix products; use einsum and tensordot when their axis contracts are clear.
- Choose solve, least squares, eigenvalue routines, decompositions and pseudoinverses according to matrix structure and conditioning.
- Use Generator for explicit random state and record seeds, splits and environment. Randomness alone does not guarantee reproducibility.
- Sorting, argsort, unique, set operations, histograms and binning transform data in different ways; preserve ID alignment.
- Handle missing values, masks, structured arrays and dtype conversion deliberately. Integer overflow and floating-point comparison need tests.
- Use stable softmax, log-sum-exp and gradient checks. A mathematically equivalent expression may be numerically unsafe.
- Window views, padding, differences, interpolation, numerical integration, FFT and smoothing need boundary and sampling assumptions.
- Use memory mapping, output buffers and appropriate layout for large arrays. np.vectorize wraps Python calls and is not automatically a speed optimisation.
- Reference implementations of regression, PCA, k-means, neural layers, attention and convolution should be compared against independently calculated small cases.
- Ranking, confusion matrices, TF-IDF, Naive Bayes and matrix factorisation need correct normalisation and evaluation semantics.
- Image kernels, equalisation and nonmaximum suppression require coordinate, channel and threshold conventions.
- Markov chains, Kalman filters, orthogonalisation, multiscale features and simulated processes need model assumptions and numerical checks.
- Treat sparse encodings and zero-copy buffers as explicit memory contracts; unsafe writes to shared views can corrupt other calculations.
- Save arrays with schema/version context, benchmark representative workloads, and test nonfinite values, empty shapes and degenerate inputs.
