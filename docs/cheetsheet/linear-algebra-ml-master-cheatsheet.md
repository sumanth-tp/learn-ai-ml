---
title: Linear Algebra for ML Master Cheatsheet
sidebar_position: 23
---

# Linear Algebra for ML Master Cheatsheet

## Arrays, vectors, and matrices

| Method | Description | Code example |
|---|---|---|
| Vector | One-dimensional numeric array. | `x = np.array([1.0, 2.0, 3.0])` |
| Matrix | Two-dimensional numeric array. | `A = np.array([[1, 2], [3, 4]])` |
| Shape | Matrix dimensions. | `rows, cols = A.shape` |
| Transpose | Swaps rows and columns. | `A_T = A.T` |
| Reshape | Changes view/shape when compatible. | `X = np.arange(12).reshape(3, 4)` |
| Broadcasting | Expands compatible dimensions without copying. | `X_centered = X - X.mean(axis=0)` |

## Core operations

| Method | Description | Code example |
|---|---|---|
| Dot product | Similarity between two vectors. | `dot = np.dot(x, y)` |
| Matrix multiply | Composes linear transformations. | `C = A @ B` |
| Elementwise multiply | Multiplies aligned elements. | `Z = X * mask` |
| Norm | Vector length or matrix magnitude. | `l2 = np.linalg.norm(x)`<br/>`fro = np.linalg.norm(A, ord="fro")` |
| Distance | Norm of difference. | `dist = np.linalg.norm(x - y)` |
| Cosine similarity | Dot product normalized by vector lengths. | `cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))` |

## Solving systems and decompositions

| Method | Description | Code example |
|---|---|---|
| Inverse | Matrix inverse when it exists. Avoid explicit inverse for solving systems. | `A_inv = np.linalg.inv(A)` |
| Solve | Solves `A x = b` more stably than inverse. | `x = np.linalg.solve(A, b)` |
| Least squares | Solves overdetermined systems. | `coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)` |
| Eigenvalues | Values describing stretch directions for square matrices. | `values, vectors = np.linalg.eig(A)` |
| SVD | Factorizes matrix into orthogonal directions and singular values. | `U, S, Vt = np.linalg.svd(X, full_matrices=False)` |
| QR decomposition | Factorizes matrix into orthogonal and triangular matrices. | `Q, R = np.linalg.qr(X)` |

## ML applications

| Method | Description | Code example |
|---|---|---|
| Linear regression closed form | Normal equation with ridge-style stabilization. | `lam = 1e-3`<br/>`coef = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)` |
| PCA via SVD | Principal components from centered data. | `Xc = X - X.mean(axis=0)`<br/>`U, S, Vt = np.linalg.svd(Xc, full_matrices=False)`<br/>`components = Vt[:2]` |
| Projection | Project vector onto direction. | `u = direction / np.linalg.norm(direction)`<br/>`projection = np.dot(x, u) * u` |
| Embedding similarity | Rank vectors by cosine similarity. | `scores = embeddings @ query / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query))` |
| Low-rank approximation | Reconstruct matrix using top singular values. | `k = 10`<br/>`X_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]` |
| Gradient shape check | Keep matrix dimensions explicit. | `assert X.shape == (batch_size, n_features)`<br/>`assert W.shape == (n_features, n_outputs)` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Standardize design matrix | Center and scale before linear models. | `X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)` |
| Add bias column | Include intercept in matrix formula. | `X_bias = np.c_[np.ones(len(X)), X]` |
| Stable softmax | Subtract max before exponentiating. | `z = logits - logits.max(axis=1, keepdims=True)`<br/>`probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)` |
| Pairwise distances | Compute all query-to-database distances. | `dists = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)` |
| Nearest neighbor | Find most similar vector. | `idx = np.argmax(scores)`<br/>`nearest = documents[idx]` |
| Rank check | Detect collinearity. | `rank = np.linalg.matrix_rank(X)` |
| Condition number | Large values indicate numerical instability. | `cond = np.linalg.cond(X.T @ X)` |
| Symmetric matrix | Build covariance-like matrix. | `cov = X_centered.T @ X_centered / (len(X) - 1)` |

## Senior numerical stability

| Method | Description | Code example |
|---|---|---|
| Avoid explicit inverse | Solve linear systems directly for stability and speed. | `coef = np.linalg.solve(X.T @ X + lam * I, X.T @ y)` |
| Cholesky solve | Efficient solve for symmetric positive definite matrices. | `L = np.linalg.cholesky(A)`<br/>`z = scipy.linalg.solve_triangular(L, b, lower=True)`<br/>`x = scipy.linalg.solve_triangular(L.T, z)` |
| Pseudoinverse | Handles rank-deficient matrices via SVD. | `coef = np.linalg.pinv(X) @ y` |
| Whitening | Decorrelates features and scales variance. | `cov = np.cov(X, rowvar=False)`<br/>`U, S, _ = np.linalg.svd(cov)`<br/>`X_white = X @ U @ np.diag(1 / np.sqrt(S + 1e-6))` |
| Log-sum-exp | Stable computation of log of summed exponentials. | `m = x.max()`<br/>`lse = m + np.log(np.exp(x - m).sum())` |
| Orthogonality check | Verify numerical decompositions. | `err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))` |
| Low-rank memory | Store factorized matrix instead of dense matrix. | `approx = U_k @ np.diag(S_k) @ Vt_k` |
| Float precision | Use float64 for sensitive linear algebra, float32 for deep learning throughput. | `X64 = X.astype(np.float64)` |

## Deep learning matrix calculus

| Method | Description | Code example |
|---|---|---|
| Linear layer shape | Batch matrix multiply for dense layers. | `Y = X @ W + b`<br/>`assert X.shape[-1] == W.shape[0]` |
| Gradient of linear layer | Backprop shapes for dense layer. | `dW = X.T @ dY`<br/>`db = dY.sum(axis=0)`<br/>`dX = dY @ W.T` |
| Attention scores | Query-key dot products scaled by head dimension. | `scores = Q @ K.transpose(-1, -2) / np.sqrt(d_k)` |
| Softmax Jacobian intuition | Gradient couples probabilities within a row. | `# d softmax is not elementwise; use framework autograd in practice.` |
| Batch covariance | Compute covariance for feature analysis. | `Xc = X - X.mean(axis=0, keepdims=True)`<br/>`cov = Xc.T @ Xc / (X.shape[0] - 1)` |
| Spectral norm | Largest singular value, used in stability constraints. | `sigma_max = np.linalg.svd(W, compute_uv=False)[0]` |
| Frobenius regularization | L2 penalty on matrix weights. | `penalty = np.sum(W * W)` |
| Embedding centering | Remove common mean direction before similarity. | `emb = emb - emb.mean(axis=0, keepdims=True)` |
