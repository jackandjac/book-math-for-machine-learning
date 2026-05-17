# Chapter 4: Linear Algebra III — Decompositions

> *"Every matrix hides a story. Decomposition tells you that story."*

---

## 4.1 The Concept: Why Decompose a Matrix?

A matrix looks like a grid of numbers. Decomposition reveals the *structure* underneath those numbers: the directions of maximum variance, the rank, the meaningful components.

Three decompositions matter most for ML:

| Decomposition | What it reveals | ML use case |
|--------------|----------------|-------------|
| **Eigendecomposition** | "Natural" axes of the transformation | PCA, graph algorithms |
| **SVD** | Full structure for any matrix | PCA, recommendation systems, text analysis |
| **PCA** | Directions of maximum variance in data | Dimensionality reduction, visualization |

We'll build them in order, each one resting on the previous.

---

## 4.2 Eigenvectors and Eigenvalues

### The Core Idea

Most vectors, when multiplied by a matrix, both rotate *and* scale. But some special vectors only **scale** — they don't rotate. These are called **eigenvectors**.

Formally, for a square matrix $A$:

$$A\mathbf{v} = \lambda \mathbf{v}$$

- $\mathbf{v}$ is an **eigenvector** (a non-zero vector that doesn't rotate when $A$ is applied)
- $\lambda$ (lambda) is the corresponding **eigenvalue** (the scalar it gets stretched by)

### A Visual Intuition

Imagine a rubber sheet stretched by matrix $A$. Most points on the sheet move to new locations — they rotate and scale. But there are special directions along which points only slide forward or backward (scale), never rotating. Those directions are the eigenvectors.

```
Before A:                  After A:
   ↑                          ↑↑   ← eigenvalue = 2 (stretched)
   v₁                         v₁
   
   → v₂                 →→→→→ v₂    ← eigenvalue = 3 (stretched more)
```

### Finding Eigenvalues

The condition $A\mathbf{v} = \lambda\mathbf{v}$ can be rewritten as:

$$A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For this to have a non-trivial solution (non-zero $\mathbf{v}$), the matrix $(A - \lambda I)$ must be singular:

$$\det(A - \lambda I) = 0$$

This is the **characteristic equation**. Solving it gives the eigenvalues.

**Worked example — finding eigenvalues of a 2×2 matrix:**

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

$$A - \lambda I = \begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix}$$

$$\det(A - \lambda I) = (4-\lambda)(3-\lambda) - (1)(2) = 0$$
$$= 12 - 4\lambda - 3\lambda + \lambda^2 - 2 = 0$$
$$= \lambda^2 - 7\lambda + 10 = 0$$
$$= (\lambda - 5)(\lambda - 2) = 0$$

So $\lambda_1 = 5$ and $\lambda_2 = 2$.

### Finding Eigenvectors

For each eigenvalue, solve $(A - \lambda I)\mathbf{v} = \mathbf{0}$.

**For $\lambda_1 = 5$:**

$$A - 5I = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}$$

Both rows say the same thing: $-v_1 + v_2 = 0$, so $v_1 = v_2$.

Eigenvector: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ (or any scalar multiple)

**Verify:** $A\mathbf{v}_1 = \begin{bmatrix}4&1\\2&3\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}5\\5\end{bmatrix} = 5\begin{bmatrix}1\\1\end{bmatrix}$ ✓

**For $\lambda_2 = 2$:**

$$A - 2I = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}$$

Both rows say: $2v_1 + v_2 = 0$, so $v_2 = -2v_1$.

Eigenvector: $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$

**Verify:** $A\mathbf{v}_2 = \begin{bmatrix}4&1\\2&3\end{bmatrix}\begin{bmatrix}1\\-2\end{bmatrix} = \begin{bmatrix}2\\-4\end{bmatrix} = 2\begin{bmatrix}1\\-2\end{bmatrix}$ ✓

```python
import numpy as np

A = np.array([[4.0, 1.0], [2.0, 3.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
# [5. 2.]  ← matches our hand calculation

print("Eigenvectors (columns):\n", eigenvectors)
# Each COLUMN is one eigenvector (NumPy convention)
# Column 0 corresponds to eigenvalue 5
# Column 1 corresponds to eigenvalue 2

# Verify: A @ v = λ * v
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"λ={lam}: A@v = {A@v}, λ*v = {lam*v}")
    print(f"  Match: {np.allclose(A @ v, lam * v)}")
```

### What Eigenvalues Tell You

- **Large eigenvalue:** that eigenvector direction is strongly amplified by $A$
- **Small eigenvalue:** that direction is shrunk
- **Zero eigenvalue:** that direction is completely collapsed — the matrix is singular
- **Negative eigenvalue:** that direction is flipped and scaled

In ML, large eigenvalues = directions of large variance in your data. This is the foundation of PCA.

---

## 4.3 Eigendecomposition

If a matrix $A \in \mathbb{R}^{n \times n}$ has $n$ linearly independent eigenvectors, we can write:

> **Edge case:** Not all matrices are diagonalizable. If a matrix has repeated eigenvalues, it may not have $n$ independent eigenvectors (e.g., $\begin{bmatrix}1&1\\0&1\end{bmatrix}$ has eigenvalue 1 with multiplicity 2 but only one independent eigenvector). In ML practice, covariance matrices (which are symmetric positive semi-definite) are always diagonalizable, so this edge case rarely bites you.

$$A = Q \Lambda Q^{-1}$$

where:
- $Q$ = matrix whose **columns are eigenvectors** $[\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n]$
- $\Lambda$ = **diagonal matrix** with eigenvalues $\text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$
- $Q^{-1}$ = inverse of $Q$

For **symmetric** matrices (covariance matrices are always symmetric), $Q^{-1} = Q^T$ (the eigenvectors form an orthogonal basis), so:

$$A = Q \Lambda Q^T$$

This is called the **spectral decomposition**. It decomposes the matrix into its natural "components."

**Worked example:**

Using $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$, $\lambda_1=5$, $\lambda_2=2$, $\mathbf{v}_1=\begin{bmatrix}1\\1\end{bmatrix}$, $\mathbf{v}_2=\begin{bmatrix}1\\-2\end{bmatrix}$:

$$Q = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$$

```python
# Reconstruct A from eigendecomposition
eigenvalues, Q = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)
A_reconstructed = Q @ Lambda @ np.linalg.inv(Q)
print("Reconstructed A:\n", A_reconstructed.round(10))
# [[4. 1.]
#  [2. 3.]]  ← original A recovered
```

---

## 4.4 Singular Value Decomposition (SVD)

Eigendecomposition only works for square matrices. **SVD works for any matrix**, which is why it's the more general and practically useful tool.

### The Decomposition

For any matrix $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$ — left singular vectors (orthogonal matrix: $U^TU = I$)
- $\Sigma \in \mathbb{R}^{m \times n}$ — singular values on the diagonal (non-negative, sorted descending)
- $V \in \mathbb{R}^{n \times n}$ — right singular vectors (orthogonal matrix: $V^TV = I$)

The **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$ are the key. They measure how much "action" the matrix has in each direction.

### Geometric Interpretation

Any matrix $A$ can be understood as three operations in sequence:
1. **$V^T$**: Rotate the input space
2. **$\Sigma$**: Scale each axis (by the singular values)
3. **$U$**: Rotate the output space

```
Input space  →(V^T)→  Rotated  →(Σ)→  Scaled  →(U)→  Output space
```

### Worked Example

$$A = \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix}$$

```python
A = np.array([[3.0, 2.0],
              [2.0, 3.0],
              [2.0, -2.0]])

U, sigma, Vt = np.linalg.svd(A)
print("U shape:", U.shape)       # (3, 3)
print("sigma:", sigma)           # singular values (descending)
print("Vt shape:", Vt.shape)     # (2, 2)

# Reconstruct A
Sigma = np.zeros_like(A)
Sigma[:len(sigma), :len(sigma)] = np.diag(sigma)
A_reconstructed = U @ Sigma @ Vt
print("Reconstructed:\n", A_reconstructed.round(10))
```

### Low-Rank Approximation: The Key Application

Here's the most important property of SVD: **you can approximate a matrix by keeping only the top-$k$ singular values** and discarding the rest. This gives you the best possible rank-$k$ approximation.

$$A \approx A_k = U_k \Sigma_k V_k^T$$

where we keep only the first $k$ columns of $U$, the first $k$ singular values of $\Sigma$, and the first $k$ rows of $V^T$.

**Why this works:** Large singular values carry most of the "energy" (information) of the matrix. Small singular values contribute noise or redundancy. By keeping only the large ones, you get a compressed representation.

**Worked example — image compression:**

```python
# Simulate a small "image" (matrix of pixel values)
image = np.array([
    [10, 20, 20, 10],
    [20, 40, 40, 20],
    [20, 40, 40, 20],
    [10, 20, 20, 10],
], dtype=float)

U, sigma, Vt = np.linalg.svd(image)
print("Singular values:", sigma.round(2))
# [100.  0.   0.   0.]  ← rank 1! All info in first singular value

# Rank-1 approximation (k=1)
k = 1
image_approx = sigma[0] * np.outer(U[:, 0], Vt[0, :])
print("Original:\n", image)
print("Rank-1 approx:\n", image_approx.round(2))
# Identical! This image has perfect rank-1 structure.

# Compute compression ratio
original_values = image.size                    # 16 numbers
compressed_values = k * (U.shape[0] + 1 + Vt.shape[1])  # k * (m + 1 + n)
print(f"Compression: {original_values} → {compressed_values} values")
```

**Real-world impact:** The Netflix Prize-winning algorithms (2009) used **matrix factorization** — a technique closely related to SVD — to decompose the user-movie rating matrix into low-rank latent factors. The factors represented latent "taste profiles." True SVD requires all entries to be present; matrix factorization extends the idea to sparse, incomplete matrices with missing ratings.

---

## 4.5 Principal Component Analysis (PCA)

PCA is the most commonly used dimensionality reduction technique in ML. It finds the directions of **maximum variance** in your data and projects onto them.

### The Problem PCA Solves

Suppose your data has 100 features (dimensions). Visualizing or training on 100D data is hard. PCA finds the 2 or 3 directions that capture the most information, letting you project to 2D or 3D for visualization or downstream modeling.

**Key insight:** The directions of maximum variance are the eigenvectors of the **covariance matrix** of your data.

### The Covariance Matrix

For a dataset $X \in \mathbb{R}^{n \times d}$ (n examples, d features), after centering (subtracting the mean):

$$\Sigma = \frac{1}{n-1} X^T X \in \mathbb{R}^{d \times d}$$

Each entry $\Sigma_{ij}$ measures how much feature $i$ and feature $j$ vary together.

- $\Sigma_{ii}$ = variance of feature $i$
- $\Sigma_{ij} = \Sigma_{ji}$ = covariance of features $i$ and $j$ (always symmetric!)

### PCA Step by Step

**Step 1: Center the data** (subtract mean of each feature)

$$\tilde{X} = X - \bar{X}$$

**Step 2: Compute the covariance matrix** (using the *centered* $\tilde{X}$ from Step 1)

$$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$$

Note: this formula only works on the already-centered data $\tilde{X}$. Using the raw $X$ here would give the wrong result (it would measure distance from the origin, not from the mean).

**Step 3: Compute eigendecomposition of $\Sigma$**

$$\Sigma = Q \Lambda Q^T$$

The eigenvectors (columns of $Q$) are the **principal components** — the new axes.

**Step 4: Sort by eigenvalue** (largest first)

The eigenvector with the largest eigenvalue is the direction of maximum variance (1st principal component). Second largest is perpendicular to it with second-most variance, etc.

**Step 5: Project data onto top-$k$ components**

$$X_{\text{reduced}} = \tilde{X} \cdot Q_k$$

where $Q_k$ contains the top-$k$ eigenvectors.

### Worked Example — PCA from Scratch

```python
import numpy as np

# ─── Toy dataset: 5 points in 2D ─────────────────────────
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9],
])
n, d = X.shape  # 10 examples, 2 features

# ─── Step 1: Center ──────────────────────────────────────
X_mean = X.mean(axis=0)
X_centered = X - X_mean
print("Feature means:", X_mean.round(4))

# ─── Step 2: Covariance matrix ───────────────────────────
Cov = (X_centered.T @ X_centered) / (n - 1)
print("\nCovariance matrix:\n", Cov.round(4))
# Should be 2×2 and symmetric

# ─── Step 3: Eigendecomposition ──────────────────────────
eigenvalues, eigenvectors = np.linalg.eigh(Cov)
# eigh (not eig!) for symmetric matrices:
# - Guarantees real eigenvalues (symmetric matrices always have real eigenvalues)
# - More numerically stable and faster than eig for symmetric inputs
# - Returns eigenvalues sorted ascending (we flip to descending in Step 4)

# ─── Step 4: Sort descending ─────────────────────────────
idx = np.argsort(eigenvalues)[::-1]  # sort indices descending
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues (variance explained):", eigenvalues.round(4))
print("Principal components (columns):\n", eigenvectors.round(4))

# ─── Step 5: Project to 1D (keep top-1 component) ────────
k = 1
PC = eigenvectors[:, :k]           # top-k eigenvectors
X_projected = X_centered @ PC      # shape: (10, 1)
print("\nProjected data (1D):\n", X_projected.round(4))

# ─── Variance explained ──────────────────────────────────
variance_explained = eigenvalues / eigenvalues.sum()
print("\nVariance explained per component:", (variance_explained * 100).round(1), "%")
# The first PC should explain ~96% of the variance for this dataset
```

### PCA with sklearn (Production Use)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
# Should match our manual calculation
```

### Why Variance = Information

Variance measures how "spread out" data is along a direction. Low variance means the data points are nearly the same along that axis — there's little information. High variance means points vary a lot — there's rich information.

PCA finds the directions where your data is most "interesting."

### Reading a Scree Plot

A **scree plot** shows eigenvalues (variance explained) per component. You look for the "elbow" — where adding more components gives diminishing returns.

```
Variance
100% |■
 80% |■
 60% |  ■
 40% |     ■
 20% |        ■  ■  ■  ■
  0% |_________________________
      PC1 PC2 PC3 PC4 PC5 ...
           ↑
        elbow here → use 3 PCs
```

---

## 4.6 SVD vs PCA: What's the Connection?

PCA can be computed via SVD of the centered data matrix directly — without explicitly computing the covariance matrix:

If $\tilde{X} = U\Sigma V^T$, then:
- The principal components are the columns of $V$ (right singular vectors)
- The eigenvalues of the covariance matrix are $\sigma_i^2 / (n-1)$

```python
# PCA via SVD (numerically more stable for high-dimensional data)
X_centered = X - X.mean(axis=0)
U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

# IMPORTANT: np.linalg.svd returns V-TRANSPOSE (Vt), not V itself.
# The formula is A = U @ diag(sigma) @ Vt
# Principal components are the COLUMNS of V = ROWS of Vt
# So we take Vt.T to get the matrix whose columns are PCs
PC_via_svd = Vt.T   # shape: (n_features, n_features); columns = principal components
print("Principal components via SVD:\n", PC_via_svd.round(4))

# Variance explained
variance_via_svd = (sigma**2) / (n - 1)
print("Eigenvalues via SVD:", variance_via_svd.round(4))
```

**Why sklearn uses SVD internally:** For high-dimensional data (thousands of features), computing an explicit $d \times d$ covariance matrix is expensive. SVD on the data matrix is more efficient and numerically stable.

---

## 4.7 Engineer's Angle: When to Use Each Technique

| Technique | Use when | Don't use when |
|-----------|----------|----------------|
| PCA | Visualizing data, removing correlated features, speeding up downstream models | Data is not linearly structured; interpretability required (PCA components are mixtures) |
| SVD | Matrix factorization, recommendation systems, text analysis (LSA) | You need interpretable components |
| Eigendecomposition | Symmetric matrices only; graph algorithms (PageRank); understanding covariance | Non-square matrices |

### LoRA: Low-Rank Adaptation in LLMs

One of the most important recent applications of SVD principles is **LoRA** (Low-Rank Adaptation), which enables efficient fine-tuning of large language models.

The idea: instead of updating the full weight matrix $W$ (billions of parameters), decompose the update into a low-rank matrix:

$$W_{\text{updated}} = W_{\text{pretrained}} + \Delta W, \quad \text{where} \quad \Delta W = BA$$

with $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ and $r \ll d$.

Instead of updating $d \times d$ parameters, you update $2 \times d \times r$ — a massive reduction for small $r$ (e.g., $r=8$ vs $d=4096$).

This works because: weight updates tend to be low-rank — the "meaningful" directions of change concentrate in a subspace, just like SVD shows that matrices often have low effective rank.

---

## 4.8 Full Code Example

```python
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ─── Eigendecomposition ──────────────────────────────────
print("=== Eigendecomposition ===")
A = np.array([[4.0, 1.0], [2.0, 3.0]])
vals, vecs = np.linalg.eig(A)
print("Eigenvalues:", vals)       # [5. 2.]
print("Eigenvectors:\n", vecs)

# Verify Av = λv for each eigenpair
for i in range(len(vals)):
    v = vecs[:, i]
    print(f"λ={vals[i]:.0f}: Av={A@v}, λv={vals[i]*v}, match={np.allclose(A@v, vals[i]*v)}")

# ─── SVD ─────────────────────────────────────────────────
print("\n=== SVD ===")
M = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
U, sigma, Vt = np.linalg.svd(M)
print("Singular values:", sigma)

# Reconstruct
Sigma = np.zeros_like(M)
Sigma[:len(sigma), :len(sigma)] = np.diag(sigma)
M_reconstructed = U @ Sigma @ Vt
print("Reconstruction error:", np.linalg.norm(M - M_reconstructed))  # ~0

# Low-rank approximation (k=1)
k = 1
M_approx = sigma[0] * np.outer(U[:, 0], Vt[0, :])
print("Rank-1 approximation:\n", M_approx)
print("Approximation error:", np.linalg.norm(M - M_approx).round(4))

# ─── PCA from scratch ────────────────────────────────────
print("\n=== PCA ===")
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9],
              [1.9, 2.2], [3.1, 3.0], [2.3, 2.7],
              [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])

n = len(X)
X_c = X - X.mean(axis=0)
Cov = X_c.T @ X_c / (n - 1)
vals, vecs = np.linalg.eigh(Cov)

# Sort descending
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]

print("Eigenvalues:", vals)
var_explained = vals / vals.sum() * 100
print("Variance explained:", var_explained.round(1), "%")
print("PC1:", vecs[:, 0].round(4))
print("PC2:", vecs[:, 1].round(4))

# Project to 1D
X_1d = X_c @ vecs[:, :1]
print("Projected shape:", X_1d.shape)  # (10, 1)
```

---

## 4.9 Chapter Summary

| Concept | Formula | Key Property |
|---------|---------|-------------|
| Eigenvector/value | $A\mathbf{v} = \lambda\mathbf{v}$ | Direction unchanged by $A$; scaled by $\lambda$ |
| Eigendecomposition | $A = Q\Lambda Q^{-1}$ | Square matrices with $n$ independent eigenvectors |
| Symmetric decomp | $A = Q\Lambda Q^T$ | $Q^{-1} = Q^T$ for symmetric matrices |
| SVD | $A = U\Sigma V^T$ | Any matrix; singular values sorted descending |
| PCA | Project onto top-$k$ eigenvectors | Directions of max variance |
| Low-rank approx | $A \approx U_k\Sigma_k V_k^T$ | Best rank-$k$ approximation (Eckart-Young theorem) |

**Key insights:**
- Eigenvectors are the "natural axes" of a matrix transformation
- SVD is the universal version: any matrix, any shape
- PCA = SVD applied to centered data
- Large singular/eigenvalues = important directions; small = noise/redundancy
- LoRA in LLMs exploits the low-rank structure of weight updates

---

## Exercises

**4.1** Find the eigenvalues of $A = \begin{bmatrix} 2 & 0 \\ 0 & 5 \end{bmatrix}$ without computation. Explain.

*Solution:* The matrix is diagonal, so the eigenvalues are just the diagonal entries: $\lambda_1 = 2$, $\lambda_2 = 5$. The eigenvectors are $\begin{bmatrix}1\\0\end{bmatrix}$ and $\begin{bmatrix}0\\1\end{bmatrix}$ — the coordinate axes. Diagonal matrices don't rotate anything, they only scale.

**4.2** A covariance matrix has eigenvalues $[25.3, 4.1, 0.8, 0.2]$. How many principal components should you keep to explain at least 95% of the variance?

*Solution:*
Total variance = $25.3 + 4.1 + 0.8 + 0.2 = 30.4$

- 1 PC: $25.3/30.4 = 83.2\%$ — not enough
- 2 PCs: $(25.3+4.1)/30.4 = 96.7\%$ ✓

Keep **2 principal components**.

**4.3** In the SVD $A = U\Sigma V^T$, what does it mean if the smallest singular value is 0?

*Solution:* A singular value of 0 means the matrix maps some non-zero input direction to zero — the matrix has a **null space**. The matrix is rank-deficient (not full rank) and therefore singular (no inverse exists). In ML, zero singular values in a feature matrix indicate perfectly linearly dependent features.

**4.4** (Challenge) Why does PCA on the centered data $\tilde{X}$ via SVD give the same principal components as the eigendecomposition of the covariance matrix $\Sigma = \frac{1}{n-1}\tilde{X}^T\tilde{X}$?

*Solution:* If $\tilde{X} = U\Sigma V^T$, then:
$$\tilde{X}^T\tilde{X} = (U\Sigma V^T)^T(U\Sigma V^T) = V\Sigma^T U^T U \Sigma V^T = V\Sigma^T\Sigma V^T = V(\Sigma^2)V^T$$

This is exactly the eigendecomposition of $\tilde{X}^T\tilde{X}$: eigenvectors are the columns of $V$ (right singular vectors of $\tilde{X}$) and eigenvalues are $\sigma_i^2$. Dividing by $(n-1)$ gives the covariance matrix eigenvalues $\sigma_i^2/(n-1)$. The eigenvectors are identical. $\square$

---

*Next: Chapter 5 — Calculus. We've described data; now we learn how models improve.*
