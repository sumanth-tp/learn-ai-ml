---
title: Calculus and Optimization Master Cheatsheet
sidebar_position: 24
---

# Calculus and Optimization Master Cheatsheet

Calculus drives every training step — gradient descent, backpropagation, adaptive optimizers — and optimization theory frames the convergence guarantees behind them. Covers derivatives, gradients, common loss landscapes, convexity, gradient-based optimizers, and second-order methods. Code in NumPy, SciPy, and PyTorch.

## Derivatives and partial derivatives

| Method | Description | Code example |
|---|---|---|
| Derivative (1D) | $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ — Instantaneous rate of change. | `# d/dx (x²) = 2x`<br/>`f = lambda x: x ** 2`<br/>`fp = lambda x: 2 * x` |
| Common derivatives | $\frac{d}{dx} x^n = n x^{n-1}$, $\frac{d}{dx} e^x = e^x$, $\frac{d}{dx} \log x = 1/x$, $\frac{d}{dx} \sin x = \cos x$. | `# d/dx (x³ + 2x) = 3x² + 2` |
| Chain rule | $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$. Foundation of backpropagation. | `# d/dx (sin(x²)) = cos(x²) · 2x` |
| Product rule | $(fg)' = f'g + fg'$. | `# d/dx (x · sin(x)) = sin(x) + x·cos(x)` |
| Quotient rule | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$. | `# d/dx (x / (1+x²)) = (1+x² - x·2x) / (1+x²)²` |
| Partial derivative | $\frac{\partial f}{\partial x_i}$ — Derivative w.r.t. one variable, others held constant. | `# f(x, y) = x² + y²`<br/>`# ∂f/∂x = 2x;  ∂f/∂y = 2y` |
| Gradient | `np.gradient(f, *varargs, axis=None)` — Numerical first derivative; for analytic gradients use autodiff. | `f_values = np.array([1, 4, 9, 16, 25])`<br/>`print(np.gradient(f_values))` |
| Numerical derivative (finite difference) | Central difference: $f'(x) \approx (f(x+h) - f(x-h)) / (2h)$ with `h ≈ 1e-5`. | `def numerical_grad(f, x, h=1e-5):`<br/>`    return (f(x + h) - f(x - h)) / (2 * h)`<br/>`print(numerical_grad(lambda x: x**2, 3.0))  # ≈ 6.0` |
| Symbolic differentiation | `sympy.diff(expr, var)` — Exact symbolic derivatives. | `import sympy as sp`<br/>`x = sp.Symbol("x")`<br/>`print(sp.diff(sp.sin(x ** 2), x))  # 2*x*cos(x**2)` |

## Gradients (multivariate calculus)

| Method | Description | Code example |
|---|---|---|
| Gradient definition | $\nabla f(x) = (\partial f / \partial x_1, \ldots, \partial f / \partial x_n)$ — Vector of partial derivatives. Points in direction of steepest ascent. | `# f(x, y) = x² + 3y² → ∇f = (2x, 6y)` |
| Gradient of dot product | $\nabla_x (a^T x) = a$. | `# loss = a.T @ x  →  ∇ = a` |
| Gradient of quadratic | $\nabla_x (x^T A x) = (A + A^T) x$. Symmetric `A` → $2Ax$. | `grad = 2 * A @ x  # symmetric A` |
| Gradient of L2 norm squared | $\nabla_x \|x\|^2 = 2x$. | `grad = 2 * x` |
| Gradient of L2 norm (not squared) | $\nabla_x \|x\| = x / \|x\|$ for `x ≠ 0`. | `grad = x / np.linalg.norm(x)` |
| Gradient of L1 norm | $\nabla_x \|x\|_1 = \text{sign}(x)$ (subgradient at 0). | `grad = np.sign(x)` |
| Gradient of softmax + cross entropy | $\nabla_{\text{logits}} L = p - y$ (probabilities minus one-hot target). The reason it's the universal classification loss. | `dL_dlogits = softmax(logits) - one_hot(y)` |
| Hessian | $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ — Symmetric matrix of second partials. | `# For f(x) = x.T @ A @ x:  H = A + A.T` |
| Jacobian | `J_{ij} = \partial f_i / \partial x_j` — Matrix of partials for vector-valued `f`. | `# For f(x) = Wx:  J = W` |
| Directional derivative | $D_u f(x) = \nabla f(x) \cdot u$ — Rate of change in direction `u`. | `# For unit vector u: derivative along u = ∇f(x) @ u` |
| Total derivative chain rule | For $z = f(g(x))$: $\frac{dz}{dx} = \frac{df}{dg} \frac{dg}{dx}$ (matrix product). | `# Used in backprop: dL/dW1 = (dL/dy)(dy/dh)(dh/dW1)` |

## Autodiff with PyTorch

| Method | Description | Code example |
|---|---|---|
| `requires_grad=True` | Mark a tensor as a leaf parameter; autograd will track operations on it. | `import torch`<br/>`x = torch.tensor([1.0, 2.0], requires_grad=True)` |
| `.backward()` | `tensor.backward(gradient=None, retain_graph=False, create_graph=False)` — Compute gradients of a scalar w.r.t. all leaf tensors. | `loss = (x ** 2).sum()`<br/>`loss.backward()`<br/>`print(x.grad)  # [2., 4.]` |
| `.grad` | Attribute populated by `.backward()`. Accumulates — zero between optimizer steps. | `optimizer.zero_grad()` |
| `torch.autograd.grad()` | `torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False)` — Compute grads without populating `.grad`. | `grads = torch.autograd.grad(loss, params)` |
| `torch.no_grad()` | Context manager: `with torch.no_grad(): ...`. Disables grad tracking (for inference). | `with torch.no_grad():`<br/>`    out = model(x)` |
| Jacobian via `torch.autograd.functional` | `jacobian(func, inputs, create_graph=False, vectorize=False)`. | `from torch.autograd.functional import jacobian`<br/>`J = jacobian(lambda x: x ** 2, torch.tensor([1.0, 2.0]))` |
| Hessian via `torch.autograd.functional` | `hessian(func, inputs, create_graph=False)`. | `from torch.autograd.functional import hessian`<br/>`H = hessian(lambda x: (x ** 2).sum(), torch.tensor([1.0, 2.0]))` |
| Higher-order gradients | `create_graph=True` in `backward()` retains the graph, enabling gradients of gradients. | `loss.backward(create_graph=True)`<br/>`grad_of_grad = torch.autograd.grad(x.grad.sum(), x)` |
| Vectorized Jacobian (JAX-style) | `vmap` (in `torch.func`) for efficient batched derivatives. | `import torch.func as tf`<br/>`jac = tf.jacrev(lambda x: x.sin())(x)` |

## Convexity

| Concept | Description | Example |
|---|---|---|
| Convex function | $f(\alpha x + (1-\alpha) y) \le \alpha f(x) + (1-\alpha) f(y)$ for all $x, y$, $\alpha \in [0, 1]$. | Quadratic with PSD Hessian, log-loss, hinge, exp. |
| Strictly convex | Strict inequality unless $x = y$. Guarantees unique minimum. | $x^2$, $-\log x$, $e^x$. |
| Concave function | $-f$ is convex. Has a unique global maximum (not minimum). | $\log x$, $\sqrt{x}$. |
| Convex set | $\alpha x + (1-\alpha) y \in S$ for all $x, y \in S$, $\alpha \in [0, 1]$. | Half-spaces, balls, polyhedra. |
| First-order condition | `f` differentiable + convex iff $f(y) \ge f(x) + \nabla f(x)^T (y - x)$. | The tangent plane lies below the function. |
| Second-order condition | `f` twice differentiable + convex iff Hessian $\nabla^2 f(x)$ is PSD everywhere. | Linear regression has Hessian $2X^T X$ — always PSD. |
| Local = global minimum (convex) | For convex `f`, every local minimum is global. | Why convex losses (logistic, OLS) have clean optimization. |
| Common convex losses | Squared error, log-loss, hinge loss, Huber loss. Convex in their argument (logits or predictions). | Train logistic regression → convex → guaranteed global optimum. |
| Common non-convex losses | Neural networks (composition of non-linear activations + linear). 0-1 loss. | Deep nets get stuck in saddle points / local minima. |
| Convex combination of convex functions | A non-negative weighted sum of convex functions is convex. | $L_1 + \lambda L_2$ is convex if both are. |

## Loss landscapes and critical points

| Concept | Description | Example |
|---|---|---|
| Local minimum | $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*) \succeq 0$ (PSD Hessian). | Quadratic with positive eigenvalues. |
| Local maximum | $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*) \preceq 0$ (NSD Hessian). | Inverted quadratic. |
| Saddle point | $\nabla f(x^*) = 0$ but Hessian has both positive and negative eigenvalues. | $f(x, y) = x^2 - y^2$ at origin. Common in deep nets. |
| Plateau | Region where gradient norm is very small but not zero. Slows training. | Sigmoid saturation regions; flat regions of cross-entropy. |
| Sharp vs flat minima | Flat minima generalize better empirically (Keskar et al.). Hessian eigenvalues near zero. | SGD's noise pushes models toward flatter regions. |
| Loss surface visualization | `loss_landscape` library or 2D random projections. | Plot loss as a function of two random directions $\delta_1, \delta_2$: $L(W + \alpha \delta_1 + \beta \delta_2)$. |

## Gradient descent and variants

| Method | Description | Code example |
|---|---|---|
| Gradient descent step | $x_{t+1} = x_t - \eta \nabla f(x_t)$ where $\eta$ is the learning rate. | `x = x - lr * grad` |
| Batch gradient descent | Use full dataset to compute gradient each step. Slow on large data. | `# for epoch in epochs: grad = mean grad over ALL samples; update` |
| Stochastic gradient descent (SGD) | Update on one sample at a time. Noisier but cheap. | `for x_i, y_i in zip(X, y):`<br/>`    grad = compute_grad(x_i, y_i)`<br/>`    w -= lr * grad` |
| Mini-batch SGD | Compromise: use a batch of B samples per step. The default in deep learning. | `for batch in dataloader:`<br/>`    grad = compute_grad(batch)`<br/>`    w -= lr * grad` |
| Momentum (heavy ball) | $v_t = \beta v_{t-1} + \nabla f(x_t)$, $x_{t+1} = x_t - \eta v_t$. Smooths trajectory; escapes shallow minima. | `v = beta * v + grad`<br/>`x -= lr * v` |
| Nesterov accelerated gradient | Look-ahead gradient: $v_t = \beta v_{t-1} + \nabla f(x_t - \eta \beta v_{t-1})$. Slightly better than plain momentum. | `torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)` |
| Adagrad | Per-parameter LR: $x_{t+1} = x_t - \eta / \sqrt{G_t + \epsilon} \cdot g_t$ where $G_t = \sum g_i^2$. Decreasing effective LR. | `torch.optim.Adagrad(params, lr=0.01)` |
| RMSprop | Exponential moving average of $g^2$: $v_t = \rho v_{t-1} + (1-\rho) g_t^2$, $x_{t+1} = x_t - \eta g_t / \sqrt{v_t + \epsilon}$. | `torch.optim.RMSprop(params, lr=0.001, alpha=0.99)` |
| Adam | Combines momentum ($m$) + RMSprop ($v$) with bias correction: $\hat m, \hat v$. Default modern optimizer. | `torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)` |
| AdamW | Adam with decoupled weight decay (regularization). Standard for transformers. | `torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)` |
| Adafactor | Memory-efficient Adam (factored second moment). For very large models (e.g., T5). | `from transformers import Adafactor`<br/>`Adafactor(params, scale_parameter=True, relative_step=True)` |
| Lion | Sign-based update (no second moment). Memory-efficient alternative to Adam. | `from lion_pytorch import Lion`<br/>`Lion(params, lr=1e-4, weight_decay=1e-2)` |
| SGD with warmup | Linear LR ramp from 0 → max over `warmup_steps`, then decay. Stabilizes early training. | `if step < warmup: lr = max_lr * step / warmup`<br/>`else: lr = decay_schedule(step)` |

## Learning rate schedules

| Method | Description | Code example |
|---|---|---|
| Constant LR | Same LR throughout. Rarely optimal except for small fine-tunes. | `lr = 1e-3` |
| Step decay | Multiply LR by `gamma` every N epochs. | `from torch.optim.lr_scheduler import StepLR`<br/>`scheduler = StepLR(opt, step_size=10, gamma=0.5)` |
| Exponential decay | $\eta_t = \eta_0 \gamma^t$. | `from torch.optim.lr_scheduler import ExponentialLR`<br/>`scheduler = ExponentialLR(opt, gamma=0.95)` |
| Cosine annealing | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))$. Smooth, popular for transformers. | `from torch.optim.lr_scheduler import CosineAnnealingLR`<br/>`scheduler = CosineAnnealingLR(opt, T_max=total_steps)` |
| Cosine with restarts | Periodic cosine decay with warm restarts. | `from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts`<br/>`scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)` |
| OneCycle (Smith) | Triangular: rises then falls. Often achieves best result in fewest steps. | `from torch.optim.lr_scheduler import OneCycleLR`<br/>`scheduler = OneCycleLR(opt, max_lr=1e-3, total_steps=total)` |
| Reduce on plateau | Drop LR when validation metric stops improving. | `from torch.optim.lr_scheduler import ReduceLROnPlateau`<br/>`scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)`<br/>`scheduler.step(val_loss)` |
| Linear warmup + decay | Standard for BERT/transformer fine-tuning. Warmup over ~6%, linear decay to 0. | `from transformers import get_linear_schedule_with_warmup`<br/>`scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total)` |

## Second-order methods

| Method | Description | Code example |
|---|---|---|
| Newton's method | $x_{t+1} = x_t - [\nabla^2 f(x_t)]^{-1} \nabla f(x_t)$. Quadratic convergence near optimum. Expensive (Hessian inversion). | `# 1D: x = x - f'(x) / f''(x)` |
| Quasi-Newton (BFGS) | Approximate Hessian from gradient differences. Used by scipy's `optimize`. | `from scipy.optimize import minimize`<br/>`res = minimize(f, x0, method="BFGS", jac=grad_f)` |
| L-BFGS | Limited-memory BFGS. Stores only the last `m` updates. For high-dim problems. | `from scipy.optimize import minimize`<br/>`res = minimize(f, x0, method="L-BFGS-B", jac=grad_f)`<br/>`# PyTorch:`<br/>`opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20)` |
| Gauss-Newton | For least-squares: approximate $H \approx J^T J$ (no second derivatives needed). | Used in nonlinear least-squares solvers. |
| Hessian-vector products | `torch.autograd.grad(grad_loss @ v, x)` — Cheap to compute, no full Hessian needed. Used in conjugate gradient methods. | `g = torch.autograd.grad(loss, x, create_graph=True)[0]`<br/>`Hv = torch.autograd.grad(g @ v, x)[0]` |
| K-FAC / Shampoo | Block-diagonal approximations to natural gradient. Used at very large scale. | Implementations exist in `torchopt`, `optax`. |

## SciPy optimization

| Method | Description | Code example |
|---|---|---|
| Unconstrained minimization | `scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, bounds=None, constraints=(), tol=None, options=None)`. | `from scipy.optimize import minimize`<br/>`f = lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2`<br/>`res = minimize(f, x0=[0, 0])`<br/>`print(res.x)  # [1., 2.]` |
| Methods | `'BFGS'`, `'L-BFGS-B'`, `'Nelder-Mead'`, `'CG'`, `'Newton-CG'`, `'trust-constr'`, `'COBYLA'`, `'SLSQP'`. | `minimize(f, x0, method="L-BFGS-B", bounds=[(0, 10), (0, 10)])` |
| Provide gradient (faster) | Pass `jac=grad_fn` if you can compute the gradient. | `minimize(f, x0, method="BFGS", jac=lambda x: 2 * (x - [1, 2]))` |
| Constrained optimization | `constraints=[{'type': 'eq', 'fun': ...}, {'type': 'ineq', 'fun': ...}]`. | `cons = [{"type": "ineq", "fun": lambda x: 10 - x[0] - x[1]}]`<br/>`minimize(f, x0, method="SLSQP", constraints=cons)` |
| Global optimization | `scipy.optimize.differential_evolution(func, bounds, ...)` — Genetic-algorithm style for non-convex problems. | `from scipy.optimize import differential_evolution`<br/>`res = differential_evolution(f, bounds=[(-10, 10), (-10, 10)])` |
| Least squares | `scipy.optimize.least_squares(fun, x0, jac='2-point', bounds=(-inf, inf), method='trf')` — Non-linear least squares (Levenberg-Marquardt). | `from scipy.optimize import least_squares`<br/>`residuals = lambda x: y - model(x, t)`<br/>`res = least_squares(residuals, x0)` |
| Linear programming | `scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs')`. | `from scipy.optimize import linprog`<br/>`# minimize c^T x s.t. A_ub @ x <= b_ub`<br/>`res = linprog(c=[-1, -2], A_ub=[[1, 1]], b_ub=[10])` |
| Root finding | `scipy.optimize.root(fun, x0, method='hybr', jac=None)` — Find `x` such that `f(x) = 0`. | `from scipy.optimize import root`<br/>`root(lambda x: x ** 3 - 2 * x - 5, x0=2)` |

## Regularization (optimization perspective)

| Method | Description | Code example |
|---|---|---|
| L2 regularization (weight decay) | Add $\lambda \|w\|^2$ to loss. Shrinks weights toward zero. Hessian becomes more PSD. | `# In PyTorch:`<br/>`optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)` |
| L1 regularization | Add $\lambda \|w\|_1$ to loss. Drives some weights to exactly zero (sparsity). | `loss = base_loss + lam * sum(p.abs().sum() for p in params)` |
| Elastic net | $\lambda_1 \|w\|_1 + \lambda_2 \|w\|^2$. Combines sparsity and smoothness. | `from sklearn.linear_model import ElasticNet`<br/>`model = ElasticNet(alpha=0.1, l1_ratio=0.5)` |
| Gradient clipping (by norm) | If $\|\text{grad}\| > c$, scale to length `c`. Prevents exploding gradients in RNNs/transformers. | `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` |
| Gradient clipping (by value) | Clamp each component to `[-c, c]`. Coarser but simple. | `torch.nn.utils.clip_grad_value_(params, clip_value=0.5)` |
| Dropout | At each training step, zero out fractions of activations. Implicit regularization. | `self.dropout = nn.Dropout(p=0.5)` |
| Early stopping | Stop training when validation loss stops improving. The most universal regularizer. | `# See validation loop in any training script.` |
| Data augmentation | Random transforms of inputs (rotation, masking, mixup). Implicit smoothness regularization. | `# Vision: torchvision.transforms; NLP: random word drop, back-translation` |

## Constrained and proximal methods

| Method | Description | Code example |
|---|---|---|
| Projected gradient descent | After gradient step, project onto feasible set: $x_{t+1} = P_C(x_t - \eta \nabla f)$. | `# Project onto simplex (probabilities)`<br/>`x = np.maximum(x - eta * grad, 0)`<br/>`x /= x.sum()` |
| Proximal gradient | For $f + g$ (with `g` non-smooth, e.g., L1): $x_{t+1} = \text{prox}_{\eta g}(x_t - \eta \nabla f)$. | `# Soft-thresholding for L1:`<br/>`def soft_threshold(z, lam):`<br/>`    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)` |
| Lagrangian | For constrained $\min f(x)$ s.t. $g(x) = 0$: $\mathcal{L}(x, \lambda) = f(x) + \lambda^T g(x)$. KKT conditions for optimality. | `# Lagrange multipliers — used in SVM dual, constrained least squares.` |
| ADMM | Alternating direction method of multipliers. For separable problems with linear constraints. | Used in distributed optimization, robust PCA. |
| Frank-Wolfe | Conditional gradient: solve a linear subproblem each step. For optimizing over polytopes. | Useful for structured prediction. |

## Common patterns

| Pattern | Code |
|---|---|
| Compute analytic gradient (small problem) | `def loss(w): return (X @ w - y) @ (X @ w - y)`<br/>`def grad(w): return 2 * X.T @ (X @ w - y)`<br/>`w = w - 0.01 * grad(w)  # one GD step` |
| Verify analytic gradient with finite differences | `def check_grad(f, grad_f, x, h=1e-5):`<br/>`    g_num = np.zeros_like(x)`<br/>`    for i in range(len(x)):`<br/>`        e = np.zeros_like(x); e[i] = h`<br/>`        g_num[i] = (f(x + e) - f(x - e)) / (2 * h)`<br/>`    return np.linalg.norm(g_num - grad_f(x)) / np.linalg.norm(g_num)` |
| Closed-form OLS | `w = np.linalg.solve(X.T @ X, X.T @ y)` |
| GD training loop | `for step in range(n_steps):`<br/>`    grad = compute_grad(w, X, y)`<br/>`    w -= lr * grad`<br/>`    if step % 100 == 0:`<br/>`        print(loss(w))` |
| Adam from scratch (intuition) | `m = beta1 * m + (1 - beta1) * g`<br/>`v = beta2 * v + (1 - beta2) * g ** 2`<br/>`m_hat = m / (1 - beta1 ** t)`<br/>`v_hat = v / (1 - beta2 ** t)`<br/>`w -= lr * m_hat / (np.sqrt(v_hat) + eps)` |
| Newton's method (1D) | `for _ in range(20):`<br/>`    x = x - f_prime(x) / f_double_prime(x)` |
| Use scipy for blackbox optimization | `from scipy.optimize import minimize`<br/>`res = minimize(lambda p: -log_likelihood(p, data), x0=initial)`<br/>`print(res.x, res.fun)` |
| Hessian-vector product (PyTorch) | `def hvp(loss_fn, params, v):`<br/>`    g = torch.autograd.grad(loss_fn(), params, create_graph=True)`<br/>`    flat_g = torch.cat([gi.flatten() for gi in g])`<br/>`    return torch.autograd.grad(flat_g @ v, params)` |
| Optim + scheduler step pattern | `for batch in loader:`<br/>`    optimizer.zero_grad()`<br/>`    loss = criterion(model(batch.x), batch.y)`<br/>`    loss.backward()`<br/>`    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`<br/>`    optimizer.step()`<br/>`    scheduler.step()` |
| Gradient checkpointing (trade compute for memory) | `from torch.utils.checkpoint import checkpoint`<br/>`out = checkpoint(self.expensive_block, x, use_reentrant=False)` |
| Find good LR (LR range test) | `lrs = np.logspace(-7, 0, 100)`<br/>`losses = []`<br/>`for lr in lrs:`<br/>`    set_lr(opt, lr)`<br/>`    loss = train_one_step(...)`<br/>`    losses.append(loss)`<br/>`# Plot losses vs lr; pick where loss drops fastest` |
| Soft-thresholding for L1 prox | `def soft_thresh(z, lam):`<br/>`    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)` |
| Convex check via Hessian | `vals = np.linalg.eigvalsh(H)`<br/>`is_convex = np.all(vals >= -1e-8)` |
| Approximate Hessian (Gauss-Newton) | `# For least squares with residuals r(x):`<br/>`J = jacobian_of_r(x)  # shape (n_samples, n_params)`<br/>`H_approx = J.T @ J` |
