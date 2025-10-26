

# 🧠 PyTorch Practice Problems — Manual ML from Scratch

These 5 problems test your understanding of **tensors, autograd, broadcasting, and manual gradient descent** in PyTorch.
No `nn.Module` or `torch.optim` — do everything from scratch.

---

## **1️⃣ Broadcasted Standardization + Cosine Similarity**

**Goal:**
Given a dataset X ∈ ℝ^(N×D), standardize its columns and compute all-pairs cosine similarity.

**Data:**

```python
X = torch.randn(N, D)  # e.g., N=4000, D=256
```

**Tasks:**

1. Standardize each column using broadcasting:

   ```
   X_std = (X - mean) / std
   ```
2. Compute cosine similarity matrix:

   ```
   S = (X_std @ X_std.T) / (||X_i|| * ||X_j||)
   ```

   Use vectorized operations, no loops.
3. Define a scalar loss = mean of all **upper-triangular** entries of S (excluding diagonal).
   Call `.backward()` and verify:

   * `X.grad` exists
   * Shape is (N, D)
   * Not all zeros

**Report:**

* CPU vs GPU time
* `X.grad` min, max, mean

---

## **2️⃣ Ridge Linear Regression (Manual Gradient Descent)**

**Goal:** Fit

```
ŷ = Xw + b
```

with L2 regularization

```
Loss = (1/N) * ||Xw + b - y||² + λ * ||w||²
```

**Data:**

```python
N, D = 10_000, 50
X = torch.randn(N, D)
true_w, true_b = torch.randn(D, 1), torch.randn(1)
y = X @ true_w + true_b + 0.1 * torch.randn(N, 1)
```

**Tasks:**

1. Train with mini-batches (size = 512).
2. Implement **gradient accumulation**: simulate batch size 2048 by accumulating 4 × 512 steps before updating.
3. Update rule:

   ```python
   with torch.no_grad():
       w -= lr * w.grad
       b -= lr * b.grad
   w.grad.zero_()
   b.grad.zero_()
   ```

**Report:**

* Loss curve
* ||w - w*|| at end
* Compare accumulated vs full-batch training

---

## **3️⃣ Logistic Regression (Binary) with Stability**

**Goal:** Build binary classifier

```
ŷ = σ(Xw + b)
σ(z) = 1 / (1 + exp(-z))
```

**Data:**

* Create two 2D Gaussian blobs (N=3000).
* Labels = {0, 1}.

**Loss (Binary Cross-Entropy):**

```
L = -(1/N) * Σ [ y*log(ŷ) + (1 - y)*log(1 - ŷ) ]
```

**Tasks:**

1. Implement a **numerically stable sigmoid** (avoid overflow).
2. Compute BCE loss manually (no `torch.nn`).
3. Train with manual gradient descent (try lr=0.1, then lr=1.0).
4. Track accuracy each epoch.
5. Add L2 regularization, compare decision boundaries.

**Report:**

* Final accuracy
* How learning rate and L2 affect convergence

---

## **4️⃣ Softmax Classifier + Gradient Check**

**Goal:** Implement a linear softmax classifier with cross-entropy loss and verify gradients manually.

**Model:**

```
logits = XW + b
ŷ = softmax(logits)
```

**Loss (Cross-Entropy):**

```
L = -(1/N) * Σ log(ŷ[i, y_i])
```

**Stability Tip:** Subtract the max logit before exponentiating:

```python
logits -= logits.max(dim=1, keepdim=True).values
```

**Tasks:**

1. Implement softmax forward and CE loss manually.
2. Train with mini-batch gradient descent.
3. Gradient check:

   * Compute autograd gradients (`.backward()`).
   * Compute numerical gradients:

     ```
     (L(θ+ε) - L(θ-ε)) / (2ε)
     ```
   * Compare relative error < 1e−3.

**Report:**

* Max relative gradient error
* Training accuracy

---

## **5️⃣ Two-Layer MLP (Manual Params) + Regularization & Clipping**

**Goal:** Build 1-hidden-layer MLP for regression:

```
h = ReLU(X @ W1 + b1)
ŷ = h @ W2 + b2
```

**Data:**

```
y = sin(2x1) + 0.5*cos(3x2) + noise
```

with `X ∈ R^(N×2)`, `y ∈ R^(N×1)` and `N=5000`.

**Parameters:**

```
W1: (2, 64)
b1: (64,)
W2: (64, 1)
b2: (1,)
```

All with `requires_grad=True`.

**Loss (MSE + weight decay):**

```
L = MSE(ŷ, y) + λ*(||W1||² + ||W2||²)
```

**Tasks:**

1. Implement forward + backward manually.
2. Add:

   * Weight decay (L2)
   * Gradient clipping: clip grad norm ≤ 1.0
3. Compare:

   * With vs without clipping
   * With vs without weight decay
4. Track train/val MSE.

**Report:**

* Final MSE
* Effect of clipping and regularization

---

### 🧩 General Tips

* Always:

  ```python
  with torch.no_grad():
      param -= lr * param.grad
      param.grad.zero_()
  ```
* Use GPU if available:

  ```python
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  ```
* Log loss, grad norms, and weight norms each epoch.

---

