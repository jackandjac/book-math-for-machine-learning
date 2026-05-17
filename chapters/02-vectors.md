# Chapter 2: Linear Algebra I — Vectors

> *"A vector is just an array with geometry attached."*

---

## 2.1 The Concept: What Is a Vector?

In programming, an array is just a container of numbers:

```python
temperatures = [72.1, 68.4, 75.0, 71.3]
```

A **vector** is the same thing, but with an important extra property: the *position* of each number has a geometric meaning. When you have a vector `[3.0, 4.0]`, you're not just storing two numbers — you're describing a **point in 2D space**, or an **arrow pointing from the origin to that point**.

This geometric interpretation is what makes vectors powerful for ML. Every data point in your training set is a vector. Every image, sentence embedding, and user feature profile is a vector. The math we do on vectors — dot products, norms, projections — directly translates to ML operations.

### Vectors as Arrows

Think of the vector $\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ as an arrow:

```
  y
  5 |        ●  ← tip at (3, 4)
  4 |       /
  3 |      /
  2 |     /  ← the arrow
  1 |    /
  0 |___/_____________ x
    0  1  2  3  4  5
    ↑
  origin (tail)
```

The vector starts at the origin (0, 0) and points to (3, 4). The length of this arrow is something we'll compute shortly — it's called the **norm**.

### Notation

We write vectors in bold lowercase: $\mathbf{v}$, $\mathbf{x}$, $\mathbf{w}$.

A vector in $n$-dimensional space:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

- $v_1, v_2, \ldots, v_n$ are the **components** (or elements)
- $\mathbb{R}^n$ means "a list of $n$ real numbers"
- In code: `v = np.array([v1, v2, ..., vn])`

**Column vs row vectors:**  
By convention, vectors are **columns** (tall) unless stated otherwise. A row vector is written $\mathbf{v}^T$ (transposed). This distinction matters for matrix multiplication in Chapter 3.

```python
import numpy as np

# Column vector (conceptually)
v = np.array([3.0, 4.0])   # shape: (2,)

# Row vector (transpose)
v_row = v.reshape(1, -1)   # shape: (1, 2)
v_col = v.reshape(-1, 1)   # shape: (2, 1)
```

---

## 2.2 Vector Operations

### 2.2.1 Addition

Adding two vectors: add component by component.

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$$

**Worked example:**

$$\begin{bmatrix} 1 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 2 \end{bmatrix} = \begin{bmatrix} 1+4 \\ 3+2 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$$

**Geometric meaning:** Vector addition is like following two arrows end-to-end. Go 1 right and 3 up, then go 4 right and 2 up — you end up at (5, 5).

```python
u = np.array([1.0, 3.0])
v = np.array([4.0, 2.0])
result = u + v
print(result)  # [5. 5.]
```

**Rule:** Vectors must have the same dimension to be added. Adding a 3D vector to a 2D vector is a type error — both in math and NumPy.

### 2.2.2 Scalar Multiplication

A **scalar** is a single number (as opposed to a vector). Multiplying a vector by a scalar stretches or shrinks it:

$$c \cdot \mathbf{v} = c \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \end{bmatrix}$$

**Worked example:**

$$3 \cdot \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 6 \\ 3 \end{bmatrix}$$

If $c > 1$: stretches the vector (makes it longer).  
If $0 < c < 1$: shrinks it (makes it shorter).  
If $c < 0$: flips the direction AND scales it.

```python
v = np.array([2.0, 1.0])
print(3 * v)    # [6. 3.]
print(-1 * v)   # [-2. -1.]  ← flipped direction
print(0.5 * v)  # [1. 0.5]   ← shrunk
```

### 2.2.3 Element-wise Multiplication (Hadamard Product)

Not to be confused with the dot product. The Hadamard product multiplies corresponding elements:

$$\mathbf{u} \odot \mathbf{v} = \begin{bmatrix} u_1 \cdot v_1 \\ u_2 \cdot v_2 \end{bmatrix}$$

```python
u = np.array([2.0, 3.0])
v = np.array([4.0, 5.0])
print(u * v)   # [8. 15.]  ← element-wise (Hadamard)
```

This is used in neural network attention mechanisms and gating. The dot product (next section) is completely different.

---

## 2.3 The Dot Product

The dot product is the single most important vector operation in ML. It appears in:
- Every linear regression prediction
- Every neuron in a neural network
- Cosine similarity for embeddings
- The attention mechanism in Transformers

### Definition

Given two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, their dot product is:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

The result is a **scalar** (a single number), not a vector.

**Worked example — 3D vectors:**

$$\mathbf{u} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$$

$$\mathbf{u} \cdot \mathbf{v} = (1)(4) + (2)(5) + (3)(6) = 4 + 10 + 18 = 32$$

```python
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])

# Three equivalent ways to compute the dot product:
print(np.dot(u, v))   # 32.0
print(u @ v)          # 32.0  ← preferred in modern ML code
print(sum(u * v))     # 32.0  ← explicit version
```

### The Geometric Meaning

Here's the key insight the formula hides:

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where $\theta$ is the angle between the two vectors.

> **Trust this result:** The algebraic proof that $\sum u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ requires the Law of Cosines from trigonometry. We won't prove it here, but you can verify it numerically: for any two vectors, compute both sides and confirm they match. The important thing is *what this formula tells you*: the dot product is zero when $\theta = 90°$, positive when the angle is acute, and negative when obtuse.

This means:
- If $\mathbf{u}$ and $\mathbf{v}$ point in the **same direction**: $\cos(0°) = 1$, so the dot product is maximized (positive)
- If they're **perpendicular**: $\cos(90°) = 0$, so the dot product is zero
- If they point in **opposite directions**: $\cos(180°) = -1$, so the dot product is negative

**Why does this matter for ML?**

In a neural network, a neuron computes:

$$\text{output} = \mathbf{w} \cdot \mathbf{x} = \sum_{i} w_i x_i$$

where $\mathbf{w}$ is the weight vector and $\mathbf{x}$ is the input feature vector. The dot product measures how much $\mathbf{x}$ "aligns with" $\mathbf{w}$. High alignment = high activation. This is literally what it means for a neuron to "fire" strongly.

**Worked example — geometric interpretation:**

Let $\mathbf{u} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (pointing right) and $\mathbf{v} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (pointing up).

$$\mathbf{u} \cdot \mathbf{v} = (1)(0) + (0)(1) = 0$$

Zero — they're perpendicular. In ML terms: these two features are completely uncorrelated in the direction they "point."

```python
u = np.array([1.0, 0.0])
v = np.array([0.0, 1.0])
print(u @ v)  # 0.0 — perpendicular vectors

# Same direction:
a = np.array([1.0, 0.0])
b = np.array([3.0, 0.0])
print(a @ b)  # 3.0 — positive, aligned

# Opposite directions:
c = np.array([1.0, 0.0])
d = np.array([-1.0, 0.0])
print(c @ d)  # -1.0 — negative, opposed
```

---

## 2.4 Vector Norms (Length)

The **norm** of a vector is its length — the distance from the origin to its tip.

### L2 Norm (Euclidean Norm)

The most common norm. For vector $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$:

$$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^n v_i^2}$$

This is just the Pythagorean theorem generalized to $n$ dimensions.

**Worked example:**

$$\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$

$$\|\mathbf{v}\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

This is the famous 3-4-5 right triangle. The vector has length 5.

```python
v = np.array([3.0, 4.0])
print(np.linalg.norm(v))     # 5.0
print(np.linalg.norm(v, 2))  # 5.0  ← explicit L2
print(np.sqrt(v @ v))        # 5.0  ← computed from dot product
```

Note: $\|\mathbf{v}\|^2 = \mathbf{v} \cdot \mathbf{v}$ — the squared norm equals the dot product with itself. This identity appears constantly in ML loss functions.

### L1 Norm (Manhattan Norm)

Sum of absolute values:

$$\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n| = \sum_{i=1}^n |v_i|$$

**Worked example:**

$$\mathbf{v} = \begin{bmatrix} 3 \\ -4 \end{bmatrix}, \quad \|\mathbf{v}\|_1 = |3| + |-4| = 3 + 4 = 7$$

The L1 norm doesn't use the Pythagorean theorem — it's the distance you'd travel if you could only move along grid lines (like city blocks). This is why it's called the "Manhattan" norm.

```python
v = np.array([3.0, -4.0])
print(np.linalg.norm(v, 1))   # 7.0
print(np.sum(np.abs(v)))      # 7.0
```

**When each norm is used in ML:**

| Norm | Symbol | Use case |
|------|--------|----------|
| L2 | $\|\mathbf{w}\|_2$ | L2 regularization (Ridge); distance metrics; default norm |
| L1 | $\|\mathbf{w}\|_1$ | L1 regularization (Lasso); promotes sparsity (many weights → 0) |
| L∞ | $\|\mathbf{w}\|_\infty$ | Max absolute element; used in adversarial robustness (e.g., FGSM attacks) |

### L∞ Norm

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

```python
v = np.array([3.0, -4.0, 1.0])
print(np.linalg.norm(v, np.inf))  # 4.0 — largest absolute value
```

---

## 2.5 Unit Vectors and Normalization

A **unit vector** has norm exactly 1. You create one by dividing a vector by its norm:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

(The hat notation $\hat{}$ means "unit vector in direction of.")

**Worked example:**

$$\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \quad \|\mathbf{v}\| = 5$$

$$\hat{\mathbf{v}} = \frac{1}{5}\begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}$$

Verify: $\|\hat{\mathbf{v}}\| = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1.0} = 1$ ✓

```python
v = np.array([3.0, 4.0])
v_hat = v / np.linalg.norm(v)
print(v_hat)                      # [0.6 0.8]
print(np.linalg.norm(v_hat))      # 1.0
```

**Why normalize?** In ML, unnormalized features cause problems. If feature A is in dollars (0–100,000) and feature B is in years (0–100), the dollar feature dominates the dot product by sheer magnitude. Normalization puts all features on the same scale so the model can compare them fairly.

---

## 2.6 Cosine Similarity

Cosine similarity measures how similar two vectors are *in direction*, regardless of magnitude. It's the backbone of recommendation systems, search, and NLP embeddings.

$$\text{cosine similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

This is exactly $\cos\theta$ from the geometric dot product formula. Output range: $[-1, 1]$.

| Value | Meaning |
|-------|---------|
| 1.0 | Identical direction (most similar) |
| 0.0 | Perpendicular (no similarity) |
| -1.0 | Opposite direction (most dissimilar) |

**Worked example — word embeddings:**

Imagine words are represented as 3D vectors (in reality, 768 or more dimensions):

$$\text{"king"} = \begin{bmatrix} 0.8 \\ 0.3 \\ 0.1 \end{bmatrix}, \quad \text{"queen"} = \begin{bmatrix} 0.7 \\ 0.4 \\ 0.2 \end{bmatrix}, \quad \text{"apple"} = \begin{bmatrix} 0.1 \\ 0.9 \\ 0.8 \end{bmatrix}$$

Step 1 — compute dot products:
$$\text{"king"} \cdot \text{"queen"} = (0.8)(0.7) + (0.3)(0.4) + (0.1)(0.2) = 0.56 + 0.12 + 0.02 = 0.70$$
$$\text{"king"} \cdot \text{"apple"} = (0.8)(0.1) + (0.3)(0.9) + (0.1)(0.8) = 0.08 + 0.27 + 0.08 = 0.43$$

Step 2 — compute norms:
$$\|\text{"king"}\| = \sqrt{0.64 + 0.09 + 0.01} = \sqrt{0.74} \approx 0.860$$
$$\|\text{"queen"}\| = \sqrt{0.49 + 0.16 + 0.04} = \sqrt{0.69} \approx 0.831$$
$$\|\text{"apple"}\| = \sqrt{0.01 + 0.81 + 0.64} = \sqrt{1.46} \approx 1.208$$

Step 3 — compute cosine similarity:
$$\text{sim}(\text{"king"}, \text{"queen"}) = \frac{0.70}{0.860 \times 0.831} \approx \frac{0.70}{0.715} \approx 0.980$$
$$\text{sim}(\text{"king"}, \text{"apple"}) = \frac{0.43}{0.860 \times 1.208} \approx \frac{0.43}{1.039} \approx 0.414$$

"King" and "queen" have similarity 0.979 (nearly identical direction) while "king" and "apple" have 0.414 — far less similar. The model has learned that kings and queens are semantically close.

```python
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

king  = np.array([0.8, 0.3, 0.1])
queen = np.array([0.7, 0.4, 0.2])
apple = np.array([0.1, 0.9, 0.8])

print(f"king  vs queen: {cosine_similarity(king, queen):.4f}")  # ~0.9796
print(f"king  vs apple: {cosine_similarity(king, apple):.4f}")  # ~0.4137
```

---

## 2.7 Linear Combinations and Span

A **linear combination** of vectors is the result of scaling and adding them:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k$$

where $c_1, c_2, \ldots, c_k$ are scalars.

**Example:**

$$3 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 2 \end{bmatrix} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$$

The **span** of a set of vectors is all possible linear combinations — the set of all points you can reach by choosing any scalars $c_1, c_2, \ldots$

**When do vectors span all of $\mathbb{R}^2$?** Only when they're *linearly independent* (§2.8). Here's the contrast:

- $\mathbf{v}_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $\mathbf{v}_2 = \begin{bmatrix}0\\1\end{bmatrix}$ → span all of $\mathbb{R}^2$. Any point $(a, b)$ = $a\mathbf{v}_1 + b\mathbf{v}_2$. ✓

- $\mathbf{v}_1 = \begin{bmatrix}1\\2\end{bmatrix}$ and $\mathbf{v}_2 = \begin{bmatrix}2\\4\end{bmatrix}$ → span only a *line* through the origin. Because $\mathbf{v}_2 = 2\mathbf{v}_1$, any combination $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = (c_1 + 2c_2)\mathbf{v}_1$ — you never escape the line $y = 2x$. ✗

The zero vector $\mathbf{0} = \begin{bmatrix}0\\0\end{bmatrix}$ is always in the span of any set (set all scalars to 0). It plays an important role in ML: zero-initialized weights, zero gradients at convergence, and the null space of a matrix.

**Why this matters for ML:** Every prediction a linear model makes is a linear combination of the input features:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = \mathbf{w} \cdot \mathbf{x}$$

This is literally the dot product — and the entire expressiveness of a linear model is limited to linear combinations of its input features.

---

## 2.8 Linear Independence

Two vectors are **linearly independent** if neither is a scalar multiple of the other — they point in genuinely different directions and neither can be built from the other.

**Example — dependent:**

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix} = 2\mathbf{v}_1$$

$\mathbf{v}_2$ is just $\mathbf{v}_1$ scaled by 2. They're on the same line — linearly **dependent**.

**Example — independent:**

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Neither can be built from the other. They're linearly **independent**.

**Why this matters for ML:**

- **Redundant features are linearly dependent.** If feature A is always twice feature B, one of them carries no new information.
- **Matrix rank** (Chapter 3) and **PCA** (Chapter 4) are fundamentally about linear independence.
- **Multicollinearity** in regression is the case where features are nearly linearly dependent — it destabilizes model training.

---

## 2.9 Engineer's Angle: Vectors in ML Code

Let's connect everything to real ML workflows.

### Features as Vectors

Every row in your training data is a feature vector:

```python
# A dataset of 4 users with 3 features each: [age, income, years_experience]
X = np.array([
    [25.0, 50000.0, 2.0],   # user 0: x^(0)
    [34.0, 85000.0, 8.0],   # user 1: x^(1)
    [28.0, 62000.0, 4.0],   # user 2: x^(2)
    [45.0, 120000.0, 20.0], # user 3: x^(3)
])
# Shape: (4, 3) — 4 examples, 3 features

# Each row is one feature vector
user_0 = X[0]  # np.array([25., 50000., 2.])

# The feature "income" across all users is a column vector
incomes = X[:, 1]  # np.array([50000., 85000., 62000., 120000.])
```

### The Problem with Different Scales

Look at the data above: age is ~25–45, but income is 50,000–120,000. If we compute dot products directly, income will dominate by a factor of thousands.

```python
# Without normalization
weights = np.array([1.0, 1.0, 1.0])  # equal weights (naive)
pred_user0 = weights @ X[0]
print(pred_user0)  # 50027.0 — income dominates completely

# With normalization (zero mean, unit std)
X_mean = X.mean(axis=0)        # per-feature mean
X_std  = X.std(axis=0, ddof=0) # per-feature std (ddof=0 = population std)
# Note: ddof=0 is NumPy's default and matches sklearn's StandardScaler.
# Use ddof=1 for sample std (e.g., in pandas .std()), but ddof=0 is standard
# for feature normalization in ML.
X_norm = (X - X_mean) / X_std

print(X_norm)
# Now all features are on comparable scales
```

This is **feature normalization** — it converts each feature to a unit vector space. The math is straightforward: subtract the mean (shift to center), divide by std (scale to unit variance).

### Distance Between Examples

How "similar" are two users? Use the L2 distance (Euclidean distance):

$$d(\mathbf{x}^{(0)}, \mathbf{x}^{(1)}) = \|\mathbf{x}^{(0)} - \mathbf{x}^{(1)}\|_2$$

```python
# After normalization, distance is meaningful
user0_norm = X_norm[0]
user2_norm = X_norm[2]

diff = user0_norm - user2_norm
distance = np.linalg.norm(diff)
print(f"Distance user0-user2: {distance:.4f}")
# Smaller = more similar users
```

This is exactly what **k-NN (k-Nearest Neighbors)** does — finds training examples closest in feature space.

---

## 2.10 Full Code Example

```python
import numpy as np

# ─── Setup ───────────────────────────────────────────────
np.set_printoptions(precision=4, suppress=True)

# ─── Basic operations ────────────────────────────────────
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])

print("=== Basic Vector Operations ===")
print("u + v =", u + v)          # [5. 7. 9.]
print("3 * u =", 3 * u)          # [3. 6. 9.]
print("u · v =", u @ v)          # 32.0

# ─── Norms ───────────────────────────────────────────────
print("\n=== Norms ===")
print("||u||_2 =", np.linalg.norm(u))            # 3.7417
print("||u||_1 =", np.linalg.norm(u, 1))         # 6.0
print("||u||_inf =", np.linalg.norm(u, np.inf))  # 3.0

# ─── Normalization ───────────────────────────────────────
print("\n=== Unit Vector ===")
u_hat = u / np.linalg.norm(u)
print("û =", u_hat)                           # [0.2673 0.5345 0.8018]
print("||û|| =", np.linalg.norm(u_hat))       # 1.0

# ─── Cosine Similarity ───────────────────────────────────
print("\n=== Cosine Similarity ===")
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

king  = np.array([0.8, 0.3, 0.1])
queen = np.array([0.7, 0.4, 0.2])
apple = np.array([0.1, 0.9, 0.8])

print(f"king vs queen: {cosine_sim(king, queen):.4f}")  # ~0.9796
print(f"king vs apple: {cosine_sim(king, apple):.4f}")  # ~0.4137

# ─── Linear Combination ──────────────────────────────────
print("\n=== Linear Combination ===")
e1 = np.array([1.0, 0.0])
e2 = np.array([0.0, 1.0])
result = 3 * e1 + 2 * e2
print("3*e1 + 2*e2 =", result)  # [3. 2.]

# ─── Neural network neuron (dot product in action) ───────
print("\n=== Single Neuron Computation ===")
# Input features: [age_normalized, income_normalized, experience_normalized]
x = np.array([0.5, 1.2, -0.3])
# Learned weights
w = np.array([0.8, 0.1, 0.6])
# Bias term
b = 0.2

# Neuron output (before activation)
z = w @ x + b
print(f"z = w·x + b = {z:.4f}")  # 0.5 + 0.12 - 0.18 + 0.2 = 0.64

# Apply ReLU activation: max(0, z)
# (Activation functions are covered in depth in Chapter 10)
activation = max(0, z)
print(f"ReLU(z) = {activation:.4f}")  # 0.64 (positive, so unchanged)
```

---

## 2.11 Chapter Summary

| Concept | Formula | Code |
|---------|---------|------|
| Vector | $\mathbf{v} \in \mathbb{R}^n$ | `np.array([...])` |
| Addition | $\mathbf{u} + \mathbf{v}$ | `u + v` |
| Scalar multiply | $c\mathbf{v}$ | `c * v` |
| Dot product | $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$ | `u @ v` |
| L2 norm | $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$ | `np.linalg.norm(v)` |
| L1 norm | $\|\mathbf{v}\|_1 = \sum |v_i|$ | `np.linalg.norm(v, 1)` |
| Unit vector | $\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|$ | `v / np.linalg.norm(v)` |
| Hadamard product | $\mathbf{u} \odot \mathbf{v}$ | `u * v` (element-wise) |
| Cosine similarity | $\frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | see §2.6 |
| Zero vector | $\mathbf{0} = \begin{bmatrix}0\\\vdots\\0\end{bmatrix}$ | `np.zeros(n)` |

**Key insight:** The dot product measures alignment between vectors. Every linear model, every neural network layer, every embedding comparison uses this one operation.

---

## Exercises

**2.1** Compute $\mathbf{u} \cdot \mathbf{v}$ by hand, then in Python:

$$\mathbf{u} = \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} 1 \\ 4 \\ 2 \end{bmatrix}$$

*Solution:* $(2)(1) + (-1)(4) + (3)(2) = 2 - 4 + 6 = 4$

```python
u = np.array([2, -1, 3])
v = np.array([1, 4, 2])
print(u @ v)  # 4
```

**2.2** Find the L2 norm of $\mathbf{w} = \begin{bmatrix} 5 \\ 12 \end{bmatrix}$.

*Solution:* $\|\mathbf{w}\| = \sqrt{25 + 144} = \sqrt{169} = 13$

```python
w = np.array([5, 12])
print(np.linalg.norm(w))  # 13.0
```

**2.3** A user's feature vector is $\mathbf{x} = \begin{bmatrix} 0.3 \\ 0.7 \\ 0.1 \end{bmatrix}$ and a model's weights are $\mathbf{w} = \begin{bmatrix} 2.0 \\ -1.0 \\ 0.5 \end{bmatrix}$ with bias $b = 0.4$. Compute the neuron's output $z = \mathbf{w} \cdot \mathbf{x} + b$.

*Solution:* $(2.0)(0.3) + (-1.0)(0.7) + (0.5)(0.1) + 0.4 = 0.6 - 0.7 + 0.05 + 0.4 = 0.35$

**2.4** Are these two vectors linearly independent? $\mathbf{a} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$, $\mathbf{b} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

*Solution:* No. $\mathbf{a} = 3\mathbf{b}$. They're linearly **dependent** — both point in the same direction (along the line $y = 2x$). As features, one carries no new information.

**2.5** (Challenge) Show algebraically why $\|\mathbf{v}\|^2 = \mathbf{v} \cdot \mathbf{v}$.

*Solution:*
$$\mathbf{v} \cdot \mathbf{v} = \sum_{i=1}^n v_i \cdot v_i = \sum_{i=1}^n v_i^2$$
$$\|\mathbf{v}\|^2 = \left(\sqrt{\sum_{i=1}^n v_i^2}\right)^2 = \sum_{i=1}^n v_i^2$$

They're equal. $\square$

---

*Next: Chapter 3 — Matrices. Vectors in groups, and the operations that transform them.*
