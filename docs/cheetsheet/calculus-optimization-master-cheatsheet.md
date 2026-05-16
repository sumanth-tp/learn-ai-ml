---
title: Calculus and Optimization Master Cheatsheet
sidebar_position: 24
---

# Calculus and Optimization Master Cheatsheet

## Derivatives and gradients

| Method | Description | Code example |
|---|---|---|
| Derivative | Instantaneous rate of change for one variable. | `def numerical_derivative(f, x, eps=1e-5):`<br/>`    return (f(x + eps) - f(x - eps)) / (2 * eps)` |
| Partial derivative | Derivative with respect to one variable while others stay fixed. | `x = torch.tensor([1.0, 2.0], requires_grad=True)`<br/>`loss = (x[0] ** 2) + (3 * x[1])`<br/>`loss.backward()`<br/>`print(x.grad)` |
| Gradient | Vector of partial derivatives. Points steepest uphill. | `w = torch.randn(3, requires_grad=True)`<br/>`loss = (w ** 2).sum()`<br/>`loss.backward()`<br/>`grad = w.grad` |
| Chain rule | Backpropagation repeatedly applies the chain rule. | `x = torch.tensor(2.0, requires_grad=True)`<br/>`y = (x * 3) ** 2`<br/>`y.backward()`<br/>`print(x.grad)` |
| Jacobian | Matrix of first derivatives for vector-valued functions. | `J = torch.autograd.functional.jacobian(lambda z: z ** 2, x)` |
| Hessian | Matrix of second derivatives. Useful for curvature analysis. | `H = torch.autograd.functional.hessian(lambda z: (z ** 2).sum(), x)` |

## Losses and landscapes

| Method | Description | Code example |
|---|---|---|
| MSE loss | Regression loss: average squared error. | `loss = ((y_pred - y_true) ** 2).mean()` |
| Cross entropy | Classification loss over logits and class indices. | `loss = torch.nn.functional.cross_entropy(logits, labels)` |
| Negative log likelihood | Maximizes probability of observed data. | `loss = -dist.log_prob(observed).mean()` |
| L1 regularization | Encourages sparsity. | `loss = data_loss + lam * sum(p.abs().sum() for p in model.parameters())` |
| L2 regularization | Penalizes large weights. | `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)` |
| Convexity check | Convex functions have no bad local minima. | `# If Hessian is positive semidefinite, function is locally convex.` |

## Optimizers

| Method | Description | Code example |
|---|---|---|
| Gradient descent | Updates parameters opposite the gradient. | `w = w - lr * grad` |
| Momentum | Smooths updates using velocity. | `v = beta * v + (1 - beta) * grad`<br/>`w = w - lr * v` |
| Adam | Adaptive optimizer using first and second gradient moments. | `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)` |
| AdamW | Adam with decoupled weight decay. Common default for transformers. | `optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)` |
| Learning rate schedule | Changes LR over training. | `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)` |
| Gradient clipping | Prevents exploding gradients. | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |

## Training mechanics

| Method | Description | Code example |
|---|---|---|
| Zero gradients | Clears previous accumulated gradients. | `optimizer.zero_grad(set_to_none=True)` |
| Backward pass | Computes gradients from scalar loss. | `loss.backward()` |
| Optimizer step | Updates parameters. | `optimizer.step()` |
| No grad | Disables autograd for evaluation. | `with torch.no_grad():`<br/>`    pred = model(x)` |
| Accumulation | Simulates larger batch size across microbatches. | `loss = loss / accumulation_steps`<br/>`loss.backward()`<br/>`if step % accumulation_steps == 0:`<br/>`    optimizer.step()` |
| Mixed precision | Faster training on modern GPUs. | `with torch.autocast(device_type="cuda", dtype=torch.float16):`<br/>`    loss = model(batch).loss` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Full training step | Standard PyTorch optimization loop. | `optimizer.zero_grad(set_to_none=True)`<br/>`loss = model(batch).loss`<br/>`loss.backward()`<br/>`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`<br/>`optimizer.step()` |
| Check gradients | Debug dead layers or exploding gradients. | `for name, p in model.named_parameters():`<br/>`    if p.grad is not None:`<br/>`        print(name, p.grad.norm().item())` |
| Finite difference check | Validate custom gradient. | `approx = (f(x + eps) - f(x - eps)) / (2 * eps)` |
| Early stopping | Stop when validation stops improving. | `if val_loss < best_loss:`<br/>`    best_loss = val_loss`<br/>`    patience = 0`<br/>`else:`<br/>`    patience += 1` |
| Warmup schedule | Ramp LR up before decay. | `lr = base_lr * min(1.0, step / warmup_steps)` |
| Detect NaNs | Fail fast on numerical issues. | `if torch.isnan(loss):`<br/>`    raise FloatingPointError("loss became NaN")` |
| Freeze layers | Optimize only selected parameters. | `for p in encoder.parameters():`<br/>`    p.requires_grad = False` |
| Optimize in log space | Keep positive parameters positive. | `log_sigma = torch.nn.Parameter(torch.zeros(()))`<br/>`sigma = torch.exp(log_sigma)` |

## Senior optimization practice

| Method | Description | Code example |
|---|---|---|
| Parameter groups | Use different learning rates or weight decay for different layers. | `optimizer = AdamW([{"params": backbone.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)` |
| No decay group | Do not apply weight decay to bias and normalization parameters. | `decay, no_decay = [], []`<br/>`for name, p in model.named_parameters():`<br/>`    (no_decay if name.endswith("bias") or "norm" in name else decay).append(p)` |
| Gradient accumulation | Match large-batch behavior under memory limits. | `loss = loss / accum_steps`<br/>`loss.backward()`<br/>`if (step + 1) % accum_steps == 0:`<br/>`    optimizer.step()` |
| Learning rate finder | Sweep LR to identify stable range. | `for lr in np.logspace(-6, -1, 100):`<br/>`    set_lr(optimizer, lr)`<br/>`    train_one_batch()` |
| Warmup plus cosine | Common transformer schedule. | `if step < warmup:`<br/>`    lr = base_lr * step / warmup`<br/>`else:`<br/>`    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(progress * math.pi))` |
| EMA weights | Smooth weights for evaluation. | `for ema_p, p in zip(ema_model.parameters(), model.parameters()):`<br/>`    ema_p.data.mul_(0.999).add_(p.data, alpha=0.001)` |
| Loss scaling | Prevent mixed-precision underflow. | `scaler.scale(loss).backward()`<br/>`scaler.step(optimizer)`<br/>`scaler.update()` |
| Hessian-vector product | Estimate curvature without forming Hessian. | `grad = torch.autograd.grad(loss, params, create_graph=True)`<br/>`hvp = torch.autograd.grad(dot(grad, vector), params)` |

## Debugging convergence

| Method | Description | Code example |
|---|---|---|
| Overfit one batch | Sanity check model can learn at all. | `batch = next(iter(loader))`<br/>`for _ in range(500):`<br/>`    train_step(batch)` |
| Compare train/eval modes | Dropout and batchnorm can explain metric shifts. | `model.train()`<br/>`train_out = model(x)`<br/>`model.eval()`<br/>`eval_out = model(x)` |
| Gradient clipping diagnostics | Log pre-clip norm and clip frequency. | `norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`<br/>`wandb.log({"grad_norm": norm.item()})` |
| Activation stats | Detect saturation or dead layers. | `def hook(module, inp, out):`<br/>`    print(out.mean().item(), out.std().item())` |
| Label sanity | Verify labels align with inputs. | `for image, label in sample_batch:`<br/>`    show(image, title=str(label))` |
| Numerical anomaly mode | Find operation causing NaNs. | `with torch.autograd.set_detect_anomaly(True):`<br/>`    loss.backward()` |
| Optimizer state reset | Reset optimizer when unfreezing large modules. | `for p in encoder.parameters():`<br/>`    p.requires_grad = True`<br/>`optimizer = AdamW(model.parameters(), lr=lr)` |
| Reproducible run | Pin seed and deterministic flags for debugging. | `torch.manual_seed(seed)`<br/>`torch.backends.cudnn.deterministic = True` |
