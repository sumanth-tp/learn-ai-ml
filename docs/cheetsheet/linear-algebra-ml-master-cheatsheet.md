---
title: Linear Algebra for ML Master Cheatsheet
sidebar_position: 23
---

# Linear Algebra for ML Master Cheatsheet

Linear algebra is the language of ML: every layer, every optimizer step, every embedding is a matrix operation. Covers the operations you'll actually use, with `numpy.linalg` and `torch.linalg` APIs and the geometric intuitions behind each.

## Vectors and norms

| Method | Description | Code example |
|---|---|---|
| Vector creation | `np.array([1, 2, 3])` — 1D array. Shape `(d,)`. | `import numpy as np`<br/>`v = np.array([1.0, 2.0, 3.0])`<br/>`print(v.shape)  # (3,)` |
| Dot product | `np.dot(a, b)` or `a @ b` — Returns scalar for two 1D vectors: $\sum_i a_i b_i$. | `a = np.array([1, 2, 3])`<br/>`b = np.array([4, 5, 6])`<br/>`print(a @ b)  # 32` |
| L2 norm (Euclidean) | `np.linalg.norm(x, ord=2, axis=None, keepdims=False)` — $\|x\|_2 = \sqrt{\sum_i x_i^2}$. Default. | `print(np.linalg.norm([3, 4]))  # 5.0` |
| L1 norm (Manhattan) | `np.linalg.norm(x, ord=1)` — Sum of absolute values. Used in L1 regularization (Lasso). | `print(np.linalg.norm([3, -4], ord=1))  # 7.0` |
| L∞ norm (max) | `np.linalg.norm(x, ord=np.inf)` — Maximum absolute element. | `print(np.linalg.norm([3, -4, 2], ord=np.inf))  # 4.0` |
| Lp norm (general) | `np.linalg.norm(x, ord=p)` — $(\sum_i |x_i|^p)^{1/p}$. | `print(np.linalg.norm([1, 2, 3], ord=3))` |
| Unit vector | `v / np.linalg.norm(v)` — Normalize to length 1. | `u = v / np.linalg.norm(v)`<br/>`print(np.linalg.norm(u))  # 1.0` |
| Angle between vectors | `arccos(dot(a,b) / (norm(a)*norm(b)))` — In radians. 0 = parallel, π/2 = orthogonal. | `cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))`<br/>`angle = np.arccos(np.clip(cos, -1, 1))` |
| Cosine similarity | $\cos\theta = a \cdot b / (\|a\|\|b\|)$ — Standard similarity metric for embeddings. | `def cos_sim(a, b):`<br/>`    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))` |
| Cross product (3D only) | `np.cross(a, b)` — Returns vector perpendicular to both. | `print(np.cross([1, 0, 0], [0, 1, 0]))  # [0, 0, 1]` |
| Outer product | `np.outer(a, b)` — Returns matrix `M[i,j] = a[i] * b[j]` of shape `(len(a), len(b))`. | `M = np.outer([1, 2], [3, 4])`<br/>`print(M)  # [[3, 4], [6, 8]]` |
| Projection of a onto b | $\text{proj}_b a = \frac{a \cdot b}{b \cdot b} b$. | `def project(a, b):`<br/>`    return (a @ b) / (b @ b) * b` |
| Orthogonality test | Two vectors are orthogonal iff dot product is zero. | `print(np.isclose(np.array([1,0]) @ np.array([0,1]), 0))  # True` |

## Matrix creation and basics

| Method | Description | Code example |
|---|---|---|
| Matrix from list | `np.array([[1, 2], [3, 4]])` — 2D array. Shape `(rows, cols)`. | `A = np.array([[1, 2], [3, 4]])`<br/>`print(A.shape)  # (2, 2)` |
| Identity matrix | `np.eye(N, M=None, k=0, dtype=float)` — `k` offsets the diagonal. | `I = np.eye(3)` |
| Zero / ones matrix | `np.zeros((r, c), dtype=float)` / `np.ones((r, c), dtype=float)`. | `Z = np.zeros((3, 4))` |
| Diagonal matrix | `np.diag(v)` (from 1D vector) / `np.diag(M)` (extract diagonal). | `D = np.diag([1, 2, 3])  # 3x3 with diag entries`<br/>`print(np.diag(D))  # [1, 2, 3]` |
| Triangular matrices | `np.triu(M, k=0)` (upper) / `np.tril(M, k=0)` (lower). | `print(np.triu(A))` |
| Random matrix | `rng.normal(0, 1, size=(M, N))` or `rng.uniform(low, high, size=(M, N))`. | `rng = np.random.default_rng(0)`<br/>`X = rng.normal(0, 1, (4, 3))` |
| Reshape | `A.reshape(*shape)` — Same data, new shape. Use `-1` for inferred dim. | `A = np.arange(12).reshape(3, 4)` |
| Transpose | `A.T` or `np.transpose(A, axes=None)` — Swap rows and columns. | `print(A.T.shape)  # (4, 3)` |
| Stack vertically / horizontally | `np.vstack([A, B])` (axis=0) / `np.hstack([A, B])` (axis=1). | `C = np.vstack([A, B])` |

## Matrix multiplication

| Method | Description | Code example |
|---|---|---|
| Matmul (`@`) | `A @ B` or `np.matmul(A, B)` — Standard matrix product. `(m, k) @ (k, n) → (m, n)`. | `A = np.random.rand(2, 3)`<br/>`B = np.random.rand(3, 4)`<br/>`print((A @ B).shape)  # (2, 4)` |
| Element-wise multiply | `A * B` — Hadamard product. Shapes must broadcast. NOT matrix multiplication. | `A * B  # same shape, element-wise` |
| Batched matmul | `np.matmul(A, B)` with 3D+ arrays — Multiplies last 2 dims, broadcasts leading dims. | `A = np.random.rand(8, 2, 3)  # batch of 8`<br/>`B = np.random.rand(8, 3, 4)`<br/>`print((A @ B).shape)  # (8, 2, 4)` |
| Einsum | `np.einsum(subscripts, *operands, optimize=False)` — Declarative tensor ops. | `# Standard matmul:`<br/>`np.einsum("ik,kj->ij", A, B)`<br/>`# Batched matmul:`<br/>`np.einsum("bik,bkj->bij", A, B)`<br/>`# Trace:`<br/>`np.einsum("ii->", A)` |
| Tensor dot | `np.tensordot(a, b, axes=2)` — Generalized dot over specified axes. | `np.tensordot(A, B, axes=([1], [0]))` |
| Kronecker product | `np.kron(A, B)` — Block matrix where each entry of `A` multiplies the entire `B`. | `print(np.kron([[1, 0], [0, 1]], [[1, 2], [3, 4]]))` |
| Outer product (1D) | `np.outer(a, b)` — Returns `(len(a), len(b))` matrix. | `print(np.outer([1, 2, 3], [4, 5]))` |

## Determinant, rank, trace, norms

| Method | Description | Code example |
|---|---|---|
| Determinant | `np.linalg.det(A)` — Scalar; nonzero iff `A` invertible. Geometrically: signed volume scale factor. | `print(np.linalg.det([[1, 2], [3, 4]]))  # -2.0` |
| Sign + log-determinant | `np.linalg.slogdet(A)` — Returns `(sign, log|det|)`. Stable for large/small dets. | `sign, log_det = np.linalg.slogdet(A)` |
| Trace | `np.trace(A, offset=0, dtype=None)` — Sum of diagonal: $\text{tr}(A) = \sum_i A_{ii}$. | `print(np.trace([[1, 2], [3, 4]]))  # 5` |
| Rank | `np.linalg.matrix_rank(M, tol=None, hermitian=False)` — Number of independent rows/cols. | `print(np.linalg.matrix_rank([[1, 2], [2, 4]]))  # 1` |
| Frobenius norm | `np.linalg.norm(A, ord='fro')` — $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$. | `print(np.linalg.norm(A, ord="fro"))` |
| Spectral norm | `np.linalg.norm(A, ord=2)` — Largest singular value. Bounds amplification. | `print(np.linalg.norm(A, ord=2))` |
| Nuclear norm | `np.linalg.norm(A, ord='nuc')` — Sum of singular values. Used in matrix completion. | `print(np.linalg.norm(A, ord="nuc"))` |
| Condition number | `np.linalg.cond(A, p=None)` — Ratio of largest to smallest singular value. Large = ill-conditioned. | `print(np.linalg.cond([[1, 2], [3, 4]]))` |

## Solving linear systems

| Method | Description | Code example |
|---|---|---|
| `np.linalg.solve()` | `np.linalg.solve(a, b)` — Solve $Ax = b$. Far more stable than `inv(A) @ b`. | `A = np.array([[3, 1], [1, 2]])`<br/>`b = np.array([9, 8])`<br/>`x = np.linalg.solve(A, b)`<br/>`print(x)  # [2., 3.]` |
| `np.linalg.lstsq()` | `np.linalg.lstsq(a, b, rcond=None)` — Least-squares for over/under-determined systems. Returns `(x, residuals, rank, singular_values)`. | `x, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)` |
| `np.linalg.inv()` | `np.linalg.inv(a)` — Matrix inverse. **Avoid** for solving — use `solve()`. | `A_inv = np.linalg.inv(A)` |
| `np.linalg.pinv()` | `np.linalg.pinv(a, rcond=1e-15, hermitian=False)` — Moore-Penrose pseudo-inverse. Handles rectangular and rank-deficient. | `X_pinv = np.linalg.pinv(X)`<br/>`beta = X_pinv @ y` |
| Normal equations | $\beta = (X^T X)^{-1} X^T y$ — Use `solve(X.T @ X, X.T @ y)` for stability. | `beta = np.linalg.solve(X.T @ X, X.T @ y)` |
| Ridge regression closed-form | $\beta = (X^T X + \lambda I)^{-1} X^T y$. | `lam = 1.0`<br/>`beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)` |
| Triangular solve | `from scipy.linalg import solve_triangular`<br/>`solve_triangular(A, b, lower=False, trans=0)` — Fast when `A` triangular. | `from scipy.linalg import solve_triangular`<br/>`x = solve_triangular(L, b, lower=True)` |

## Eigenvalues and eigenvectors

| Method | Description | Code example |
|---|---|---|
| `np.linalg.eig()` | `np.linalg.eig(a)` — Returns `(eigenvalues, eigenvectors)` (columns are eigenvectors). May be complex. | `vals, vecs = np.linalg.eig([[2, 0], [0, 3]])`<br/>`print(vals)  # [2., 3.]` |
| `np.linalg.eigh()` | `np.linalg.eigh(a, UPLO='L')` — For symmetric/Hermitian matrices. Faster, stable, always real. **Use for covariance matrices.** | `cov = X.T @ X / (X.shape[0] - 1)`<br/>`vals, vecs = np.linalg.eigh(cov)` |
| `np.linalg.eigvals()` / `eigvalsh()` | Eigenvalues only. Use `eigvalsh` for symmetric (faster). | `vals = np.linalg.eigvalsh(cov)` |
| Eigenvalue interpretation | $Av = \lambda v$ — `v` direction preserved, scaled by `λ`. | `# Verify: A @ vecs[:, 0] ≈ vals[0] * vecs[:, 0]` |
| Spectral decomposition | $A = V \Lambda V^{-1}$ (general) or $A = V \Lambda V^T$ (symmetric). | `# A == vecs @ np.diag(vals) @ vecs.T  (symmetric case)` |
| Sort eigenvalues descending | `idx = np.argsort(vals)[::-1]; vals = vals[idx]; vecs = vecs[:, idx]`. | `idx = np.argsort(vals)[::-1]`<br/>`vals = vals[idx]; vecs = vecs[:, idx]` |
| PSD test | All eigenvalues ≥ 0. Covariance matrices are always PSD. | `is_psd = np.all(np.linalg.eigvalsh(A) >= -1e-10)` |
| Positive-definite test | All eigenvalues > 0. | `is_pd = np.all(np.linalg.eigvalsh(A) > 0)` |

## Matrix decompositions

| Method | Description | Code example |
|---|---|---|
| SVD | `np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)` — $A = U \Sigma V^T$. Singular values in descending order. | `U, S, Vt = np.linalg.svd(A, full_matrices=False)`<br/>`# Reconstruct: U @ np.diag(S) @ Vt` |
| Truncated SVD (low-rank approx) | Keep top-k singular values; discard the rest. Foundation of PCA, recommender systems. | `k = 5`<br/>`A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]` |
| QR decomposition | `np.linalg.qr(a, mode='reduced')` — $A = QR$. `Q` orthogonal, `R` upper triangular. Stable for least-squares. | `Q, R = np.linalg.qr(A)` |
| Cholesky | `np.linalg.cholesky(a)` — $A = LL^T$ where `L` is lower triangular. Requires `A` symmetric PSD. Fastest solver for PSD. | `L = np.linalg.cholesky(cov)` |
| LU decomposition | `from scipy.linalg import lu`<br/>`lu(a, permute_l=False)` — $PA = LU$. Used internally by `solve()`. | `from scipy.linalg import lu`<br/>`P, L, U = lu(A)` |
| Schur | `from scipy.linalg import schur`<br/>`schur(A, output='real')` — $A = Q T Q^T$ where `T` is (quasi) triangular. | `from scipy.linalg import schur`<br/>`T, Z = schur(A)` |
| Polar | `from scipy.linalg import polar`<br/>`polar(a, side='right')` — $A = UP$ with `U` orthogonal, `P` symmetric PSD. | `from scipy.linalg import polar`<br/>`U, P = polar(A)` |

## Principal Component Analysis (PCA)

| Method | Description | Code example |
|---|---|---|
| PCA via SVD (preferred) | Center data, then SVD. Right singular vectors are principal components. | `X_c = X - X.mean(axis=0)`<br/>`U, S, Vt = np.linalg.svd(X_c, full_matrices=False)`<br/>`# PCs: rows of Vt; projections: X_c @ Vt.T  (or U * S)` |
| Explained variance | `S²/(n-1)` are eigenvalues of the covariance matrix. | `expl = (S ** 2) / (X.shape[0] - 1)`<br/>`ratio = expl / expl.sum()` |
| Cumulative explained variance | Pick `k` for ≥95% variance. | `cum = np.cumsum(ratio)`<br/>`k = np.argmax(cum >= 0.95) + 1` |
| Project to k dims | `X_c @ Vt[:k].T` — Reduces to k-dimensional representation. | `X_low = X_c @ Vt[:k].T  # shape (n, k)` |
| Reconstruct from k components | Lossy approximation; quality matches singular values kept. | `X_recon = X_low @ Vt[:k] + X.mean(axis=0)` |
| PCA via covariance + eigh | Equivalent but on covariance matrix. Use `eigh` for symmetry/stability. | `cov = (X_c.T @ X_c) / (X.shape[0] - 1)`<br/>`vals, vecs = np.linalg.eigh(cov)` |
| sklearn equivalent | `PCA(n_components=None, whiten=False, svd_solver='auto', random_state=None)`. | `from sklearn.decomposition import PCA`<br/>`pca = PCA(n_components=0.95)`<br/>`X_low = pca.fit_transform(X)` |

## Orthogonality and projections

| Method | Description | Code example |
|---|---|---|
| Orthogonal matrix | $Q^T Q = I$ — Columns orthonormal. Preserves lengths and angles. | `Q, _ = np.linalg.qr(np.random.rand(5, 3))`<br/>`np.allclose(Q.T @ Q, np.eye(3))  # True` |
| Gram-Schmidt | Build orthonormal basis. Use `np.linalg.qr` in practice (more stable). | `Q, R = np.linalg.qr(vectors)` |
| Projection matrix onto subspace | If `Q` columns are orthonormal basis of subspace: $P = QQ^T$. | `P = Q @ Q.T`<br/>`projected = P @ x` |
| Hat matrix (regression projection) | $H = X(X^T X)^{-1}X^T$ — Projects `y` onto column space of `X`. Diagonals = leverages. | `H = X @ np.linalg.solve(X.T @ X, X.T)`<br/>`leverage = np.diag(H)` |
| Orthogonal complement projection | $I - QQ^T$ projects onto orthogonal complement. | `Q_perp = np.eye(n) - Q @ Q.T` |
| Null space | `from scipy.linalg import null_space`<br/>`null_space(A, rcond=None)` — Returns matrix whose columns span `ker(A)`. | `from scipy.linalg import null_space`<br/>`ns = null_space(A)` |

## Geometric intuitions

| Concept | Description | Example |
|---|---|---|
| Determinant as volume | $|\det(A)|$ is the volume scaling factor of the linear map `A`. Negative = orientation reversed. | A scales the unit square's area by `|det(A)|`. |
| Eigenvectors as invariant directions | Directions that `A` only scales (no rotation). | `# Av = λv  →  v's direction preserved` |
| SVD as rotate-scale-rotate | $A = U \Sigma V^T$: rotate by $V^T$, scale by $\Sigma$, rotate by $U$. | Any linear map decomposes as rotation + non-uniform scaling + rotation. |
| Rank as image-space dimension | `rank(A)` is the dim of the subspace `Av` can reach. | `print(np.linalg.matrix_rank(A))` |
| Null space | Vectors `v` such that $Av = 0$. Dimension = `n - rank(A)`. | `from scipy.linalg import null_space`<br/>`ns = null_space(A)` |
| Column space (image) | Span of `A`'s columns. Equals image of the map. | `rank(A) == dim(column space)` |
| Trace = sum of eigenvalues | $\text{tr}(A) = \sum_i \lambda_i$. | `np.trace(A) == np.linalg.eigvals(A).sum()` |
| Determinant = product of eigenvalues | $\det(A) = \prod_i \lambda_i$. | `np.linalg.det(A) == np.linalg.eigvals(A).prod()` |
| Frobenius² = sum of σ² | $\|A\|_F^2 = \sum_i \sigma_i^2$. | `np.linalg.norm(A,"fro")**2 == (np.linalg.svd(A, compute_uv=False)**2).sum()` |

## Matrix calculus and gradients

| Method | Description | Code example |
|---|---|---|
| Gradient of linear | $\nabla_x (a^T x) = a$. | `# For loss = a.T @ x, gradient is a.` |
| Gradient of quadratic | $\nabla_x (x^T A x) = (A + A^T) x$. Symmetric `A` → $2Ax$. | `grad = 2 * A @ x  # symmetric A` |
| Gradient of squared L2 | $\nabla_x \|x\|_2^2 = 2x$. | `grad = 2 * x` |
| Gradient of L1 | $\nabla_x \|x\|_1 = \text{sign}(x)$ (element-wise, undefined at 0). | `grad = np.sign(x)` |
| Chain rule (matrix form) | For $y = Wx$ and loss $L$: $\partial L / \partial W = (\partial L / \partial y) x^T$, $\partial L / \partial x = W^T (\partial L / \partial y)$. | `dL_dy = ...`<br/>`dL_dW = np.outer(dL_dy, x)`<br/>`dL_dx = W.T @ dL_dy` |
| Hessian | Matrix of second partial derivatives. Symmetric for smooth functions. Used in Newton's method. | `# For f(x) = x.T @ A @ x: Hessian = A + A.T (= 2A if symmetric)` |
| Jacobian | For vector-valued `f`: matrix of partials. | `import torch`<br/>`from torch.autograd.functional import jacobian`<br/>`J = jacobian(lambda x: x ** 2, torch.tensor([1.0, 2.0]))` |
| Gradient via autodiff (PyTorch) | `tensor.backward()` populates `.grad` on leaf tensors. | `x = torch.tensor([1.0, 2.0], requires_grad=True)`<br/>`((x ** 2).sum()).backward()`<br/>`print(x.grad)  # [2., 4.]` |

## PyTorch linear algebra (`torch.linalg`)

| Method | Description | Code example |
|---|---|---|
| Matmul | `A @ B` or `torch.matmul(A, B)`. Same semantics as NumPy. | `A = torch.randn(3, 4); B = torch.randn(4, 5)`<br/>`print((A @ B).shape)  # (3, 5)` |
| Solve | `torch.linalg.solve(A, B)` — Differentiable. | `x = torch.linalg.solve(A, b)` |
| SVD | `torch.linalg.svd(A, full_matrices=True)` — Returns `(U, S, Vh)`. Differentiable. | `U, S, Vh = torch.linalg.svd(A, full_matrices=False)` |
| QR | `torch.linalg.qr(A, mode='reduced')`. | `Q, R = torch.linalg.qr(A)` |
| Cholesky | `torch.linalg.cholesky(A, upper=False)`. | `L = torch.linalg.cholesky(cov)` |
| Eigenvalues | `torch.linalg.eigvals(A)` / `torch.linalg.eigvalsh(A)` (symmetric, real). | `vals = torch.linalg.eigvalsh(cov)` |
| Norm | `torch.linalg.norm(x, ord=None, dim=None, keepdim=False)`. | `print(torch.linalg.norm(x, ord=2))` |
| Determinant | `torch.linalg.det(A)` / `torch.linalg.slogdet(A)` (stable for extreme dets). | `sign, log_det = torch.linalg.slogdet(A)` |
| Inverse | `torch.linalg.inv(A)`. Same caveat: prefer `solve()`. | `A_inv = torch.linalg.inv(A)` |
| Batched ops | All `torch.linalg` functions support leading batch dim. | `A = torch.randn(8, 3, 3)`<br/>`vals = torch.linalg.eigvalsh(A)  # (8, 3)` |

## Common patterns

| Pattern | Code |
|---|---|
| Cosine similarity between rows of two matrices | `def cos_sim_matrix(A, B):`<br/>`    A = A / np.linalg.norm(A, axis=1, keepdims=True)`<br/>`    B = B / np.linalg.norm(B, axis=1, keepdims=True)`<br/>`    return A @ B.T  # shape (m, n)` |
| Pairwise Euclidean distances | `def pdist(A, B):`<br/>`    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))` |
| OLS via QR (stable) | `Q, R = np.linalg.qr(X)`<br/>`beta = np.linalg.solve(R, Q.T @ y)` |
| Ridge closed-form | `beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)` |
| PCA top-k components | `X_c = X - X.mean(axis=0)`<br/>`U, S, Vt = np.linalg.svd(X_c, full_matrices=False)`<br/>`X_pca = X_c @ Vt[:k].T` |
| Mahalanobis distance | `def mahalanobis(x, mu, cov_inv):`<br/>`    diff = x - mu`<br/>`    return np.sqrt(diff @ cov_inv @ diff)` |
| Spectral clustering (graph Laplacian) | `L = D - W  # degree minus adjacency`<br/>`vals, vecs = np.linalg.eigh(L)`<br/>`# Cluster on first k smallest non-zero eigenvectors` |
| Low-rank approximation | `U, S, Vt = np.linalg.svd(M, full_matrices=False)`<br/>`M_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]` |
| Test if PSD | `is_psd = np.all(np.linalg.eigvalsh(A) >= -1e-10)` |
| Robust solver | `try:`<br/>`    x = np.linalg.solve(A, b)`<br/>`except np.linalg.LinAlgError:`<br/>`    x, *_ = np.linalg.lstsq(A, b, rcond=None)` |
| Xavier (Glorot) init | `std = np.sqrt(2.0 / (fan_in + fan_out))`<br/>`W = np.random.normal(0, std, (fan_in, fan_out))` |
| He (Kaiming) init | `std = np.sqrt(2.0 / fan_in)  # for ReLU`<br/>`W = np.random.normal(0, std, (fan_in, fan_out))` |
| Power iteration (top eigenvector) | `def power_iter(A, n_iter=100):`<br/>`    v = np.random.rand(A.shape[1])`<br/>`    for _ in range(n_iter):`<br/>`        v = A @ v`<br/>`        v = v / np.linalg.norm(v)`<br/>`    return v, (v @ A @ v)` |
| Whitening (ZCA) | `X_c = X - X.mean(axis=0)`<br/>`U, S, Vt = np.linalg.svd(X_c, full_matrices=False)`<br/>`X_white = U @ Vt  # ZCA: preserves orientation` |
