 **5 tough, hands-on problems** on tensors, broadcasting, autograd, and manual training loops (no `nn.Module`, no `torch.optim`). Each one is fully specified so you can code from scratch.

---

## 1) Broadcasted Standardization + Cosine Similarity (no loops)

**Goal:** Given a dataset (X \in \mathbb{R}^{N\times D}), standardize features and compute all-pairs cosine similarity.

**Data:** Generate `X` with `torch.randn(N, D)`; use e.g. `N=4000, D=256` (GPU if available).

**Tasks:**

1. Standardize each column: (X_{\text{std}} = (X - \mu) / \sigma) using **broadcasting only** (no loops).
2. Compute the **cosine similarity matrix** (S = \frac{X_{\text{std}} X_{\text{std}}^\top}{|X_i||X_j|}) efficiently (avoid `for`).
3. Define a scalar loss: the **mean** of the upper-triangular entries of (S) (excluding diagonal). Call `.backward()` and confirm that `X.grad` has shape `(N,D)` and is **not** all zeros.

**Constraints:** no loops for standardization or similarity.
**Report:** time taken (CPU vs GPU), `X.grad` stats (min/max/mean).

---

## 2) Ridge Linear Regression (manual GD, with gradient accumulation)

**Goal:** Fit ( \hat{y} = Xw + b ) with **L2 penalty** ( \lambda |w|^2 ) using your own training loop.

**Data:** Create `X ∈ R^{N×D}` (`N=10_000, D=50`), true `w*`, `b*`, and targets `y = Xw* + b* + noise`.

**Loss:**
[
\mathcal{L}(w,b)=\frac{1}{N}|Xw + b - y|^2 + \lambda |w|^2
]
with (\lambda=1e{-2}).

**Tasks:**

1. Initialize `w (D,1)` and `b (1,)` with `requires_grad=True`.
2. Train with **mini-batches** of size 512 using your own dataloader logic (tensor slicing).
3. Implement **gradient accumulation**: simulate batch size 2048 by accumulating four 512-sized steps before updating once. Verify it matches (approximately) a true 2048 step.
4. Use `torch.no_grad()` for updates and `.grad.zero_()` correctly.

**Constraints:** no `nn.Module`, no `torch.optim`.
**Report:** training loss curve, final (|w-w^*|), and whether accumulated vs true larger batch give similar updates.

---

## 3) Logistic Regression (binary) with Numerical Stability

**Goal:** Build binary classifier ( \hat{y}=\sigma(Xw+b) ) from scratch.

**Data:** Make two Gaussian blobs in 2D (`N=3000`, `D=2`) with partial overlap; labels in `{0,1}`.

**Loss:** **Binary cross-entropy**. Implement **numerically stable** sigmoid + BCE: clamp logits or use `logsumexp` trick so you avoid `nan`.

**Tasks:**

1. Implement forward pass to compute logits and stable BCE loss (mean).
2. Train with manual GD; try `lr=0.1`, then `lr=1.0` to see instability.
3. Track **accuracy** each epoch.
4. Add **L2 regularization** and observe decision boundary smoothing (qualitative: compute accuracy + loss only).

**Constraints:** no `nn.Module`, no `torch.optim`, avoid `torch.sigmoid` if you want extra challenge—write a stable version yourself.
**Report:** final accuracy, effect of lr and L2 on convergence.

---

## 4) Softmax Classifier (multiclass) + Gradient Check

**Goal:** Implement a linear softmax classifier for `K` classes and **verify gradients** with finite differences.

**Data:** `X ∈ R^{N×D}` with `N=1200, D=10`; create `K=4` centroids and sample around them; labels in `[0..3]`.

**Loss:** **Cross-entropy** with **stable** softmax (subtract max per row before `exp`).

**Tasks:**

1. Parameters: `W ∈ R^{D×K}`, `b ∈ R^{K}`. Forward → logits → softmax → CE loss.
2. Train with mini-batch GD; show loss decreasing.
3. **Gradient check:** pick a tiny batch (e.g., `N=5`), freeze a copy of `W,b`, compute numerical gradients via finite differences (\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}), compare with autograd `W.grad`, `b.grad`. Target relative error < `1e-3`.

**Constraints:** no `nn.Module`, no `torch.optim`.
**Report:** max relative gradient error, training accuracy.

---

## 5) Two-Layer MLP (manual parameters) + Regularization & Clipping

**Goal:** Build a 1-hidden-layer network **by hand**:
[
h=\text{ReLU}(XW_1+b_1),\quad \hat{y}=hW_2+b_2
]
Use MSE regression on a **nonlinear target**.

**Data:** `X ∈ R^{N×2}` with `N=5000`, targets `y = sin(2x_1) + 0.5 cos(3x_2) + noise` (shape `(N,1)`).

**Tasks:**

1. Initialize `W1 (2×64)`, `b1 (64)`, `W2 (64×1)`, `b2 (1)` with `requires_grad=True`.
2. Train with mini-batches; track train/val MSE.
3. Add **weight decay** (L2 on `W1,W2`), and **gradient clipping** to max norm (e.g., 1.0) before the update.
4. Show that without clipping + with big `lr` the loss can blow up, and with clipping it stabilizes.

**Constraints:** no `nn.Module`, no `torch.optim`.
**Report:** final train/val MSE, effect of clipping and weight decay.

---

### How to work

* Prefer **GPU tensors** when size is large: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
* Always: updates inside `torch.no_grad()`, then `.grad.zero_()`.
* Log key stats (loss, norms of grads/weights) to “feel” training.

Want to start with one together? **Pick a problem (1–5)** and tell me which you’ll attempt first—I’ll give you a tiny starter scaffold (just variable names/shapes, not the solution).
