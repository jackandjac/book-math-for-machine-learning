# Chapter 3: Linear Algebra II — Matrices

> *"A matrix is a function in disguise — it transforms one vector into another."*

---

## 3.1 The Concept: What Is a Matrix?

In Chapter 2, a vector was a 1D array of numbers. A **matrix** is a 2D array — a grid of numbers organized into rows and columns.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
# Shape: (2, 3) — 2 rows, 3 columns
```

But here's the deeper insight: **a matrix is a transformation**. When you multiply a matrix by a vector, you're not just doing arithmetic — you're moving that vector to a new position in space. This is how neural network layers work, how images are transformed, and how data is projected from high dimensions to low ones.

Think of a matrix as a function: it takes a vector as input and produces a new vector as output.

$$A: \mathbb{R}^n \rightarrow \mathbb{R}^m$$

---

## 3.2 Notation and Anatomy

An $m \times n$ matrix has $m$ rows and $n$ columns:

$$A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

- $a_{ij}$ means row $i$, column $j$ — like `A[i][j]` (1-indexed in math, 0-indexed in Python)
- Matrices are written in **bold uppercase**: $A$, $B$, $W$
- $A \in \mathbb{R}^{m \times n}$ means "$A$ is a matrix with $m$ rows and $n$ columns of real numbers"

**Concrete example:**

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

- $a_{12} = 2$ (row 1, col 2)
- $a_{21} = 4$ (row 2, col 1)
- In Python: `A[0, 1] = 2`, `A[1, 0] = 4`

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A.shape)    # (2, 3)
print(A[0, 1])    # 2    ← a_12 (0-indexed)
print(A[1, 0])    # 4    ← a_21 (0-indexed)
```

### Special Matrix Shapes

| Name | Shape | Description |
|------|-------|-------------|
| Square | $n \times n$ | Same rows and columns |
| Row vector | $1 \times n$ | Single row |
| Column vector | $m \times 1$ | Single column |
| Scalar | $1 \times 1$ | A single number as a matrix |

---

## 3.3 Matrix Operations

### 3.3.1 Transpose

The **transpose** $A^T$ flips a matrix across its diagonal — rows become columns, columns become rows:

$$\text{If } A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \text{ then } A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

If $A \in \mathbb{R}^{m \times n}$, then $A^T \in \mathbb{R}^{n \times m}$.

Formally: $(A^T)_{ij} = A_{ji}$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]
print(A.T.shape)  # (3, 2)
```

**Why it matters in ML:** The relationship between row and column vectors uses transpose. When computing $\mathbf{u} \cdot \mathbf{v}$ as a matrix product, it's written $\mathbf{u}^T \mathbf{v}$ (row vector times column vector = scalar).

### 3.3.2 Addition and Subtraction

Add element-by-element. Both matrices must have **identical** shape.

$$A + B = \begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{bmatrix}$$

**Worked example:**

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A + B)
# [[ 6  8]
#  [10 12]]
```

### 3.3.3 Scalar Multiplication

Multiply every element by the scalar:

$$3 \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 9 & 12 \end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
print(3 * A)
# [[ 3  6]
#  [ 9 12]]
```

---

## 3.4 Matrix Multiplication

This is the most important operation in all of ML. **Every neural network forward pass is a sequence of matrix multiplications.**

### The Rule

To multiply $A \in \mathbb{R}^{m \times k}$ by $B \in \mathbb{R}^{k \times n}$, the **inner dimensions must match**. The result is $C \in \mathbb{R}^{m \times n}$.

$$A_{m \times k} \times B_{k \times n} = C_{m \times n}$$

**Memory trick:**
```
(m × k) @ (k × n) = (m × n)
         ↑↑
      must match
```

Each element $c_{ij}$ of $C$ is the dot product of row $i$ of $A$ with column $j$ of $B$:

$$c_{ij} = \sum_{l=1}^{k} a_{il} \cdot b_{lj}$$

### Step-by-Step Worked Example

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

Both are $2 \times 2$, so $C = AB$ is also $2 \times 2$:

$$c_{11} = \text{row 1 of } A \cdot \text{col 1 of } B = (1)(5) + (2)(7) = 5 + 14 = 19$$
$$c_{12} = \text{row 1 of } A \cdot \text{col 2 of } B = (1)(6) + (2)(8) = 6 + 16 = 22$$
$$c_{21} = \text{row 2 of } A \cdot \text{col 1 of } B = (3)(5) + (4)(7) = 15 + 28 = 43$$
$$c_{22} = \text{row 2 of } A \cdot \text{col 2 of } B = (3)(6) + (4)(8) = 18 + 32 = 50$$

$$C = AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
print(C)
# [[19 22]
#  [43 50]]
```

### Matrix × Vector (The Core ML Operation)

When a matrix multiplies a vector, the output is a new vector:

$$A\mathbf{x} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 1x_1 + 2x_2 \\ 3x_1 + 4x_2 \\ 5x_1 + 6x_2 \end{bmatrix}$$

Each output element is a dot product of one row with $\mathbf{x}$.

**Worked example:**

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$$

$$A\mathbf{x} = \begin{bmatrix} (1)(5)+(2)(6) \\ (3)(5)+(4)(6) \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
x = np.array([5.0, 6.0])
print(A @ x)  # [17. 39.]
```

**Neural network connection:** A fully connected layer computes:

$$\mathbf{h} = W\mathbf{x} + \mathbf{b}$$

where:
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$ is the weight matrix
- $\mathbf{x} \in \mathbb{R}^{d_{in}}$ is the input vector
- $\mathbf{b} \in \mathbb{R}^{d_{out}}$ is the bias vector
- $\mathbf{h} \in \mathbb{R}^{d_{out}}$ is the output (hidden layer activations)

A 3-layer network is just: $\hat{y} = W_3(\text{activation}(W_2(\text{activation}(W_1\mathbf{x} + \mathbf{b}_1)) + \mathbf{b}_2)) + \mathbf{b}_3$

### Critical Property: Matrix Multiplication is NOT Commutative

For scalars: $3 \times 5 = 5 \times 3$. For matrices: **$AB \neq BA$ in general.**

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}$$

$$\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}$$

These are different!

**Real ML bug this causes:** Suppose you build a two-layer network and accidentally swap the layer order:

```python
# Correct: W2 applied after W1
h = W2 @ (W1 @ x)   # applies W1 first, then W2

# Wrong: layers swapped
h = W1 @ (W2 @ x)   # applies W2 first — different transformation entirely
```

Both lines run without errors. Both produce output of the same shape. But you've built a completely different model. The non-commutativity of matrix multiplication means **layer order is semantically significant**, not just an implementation detail. This is one of the most common silent bugs in custom neural network implementations.

---

## 3.5 Special Matrices

### Identity Matrix $I$

The identity matrix has 1s on the diagonal and 0s everywhere else:

$$I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

Property: $AI = IA = A$ for any compatible matrix $A$. It's the matrix equivalent of multiplying by 1.

```python
I = np.eye(3)  # 3×3 identity
A = np.array([[1, 2], [3, 4]])
I2 = np.eye(2)
print(A @ I2)  # [[1. 2.] [3. 4.]] — unchanged
```

### Diagonal Matrix

Only the diagonal elements are non-zero:

$$D = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}$$

Multiplying by a diagonal matrix scales each dimension independently.

```python
D = np.diag([2.0, 3.0, 5.0])
x = np.array([1.0, 1.0, 1.0])
print(D @ x)  # [2. 3. 5.] — each dim scaled separately
```

### Symmetric Matrix

$A = A^T$ — the matrix equals its own transpose.

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 4 \\ 3 & 4 & 6 \end{bmatrix}$$

**Why it matters:** Covariance matrices (used in PCA, Gaussian distributions) are always symmetric. This is a critical structural property that simplifies decomposition (Chapter 4).

```python
A = np.array([[1, 2, 3], [2, 5, 4], [3, 4, 6]])
print(np.allclose(A, A.T))  # True — it's symmetric
```

### Orthogonal Matrix

A square matrix $Q$ where $Q^T Q = I$ — the columns are all unit vectors and are mutually perpendicular.

**Property:** $Q^{-1} = Q^T$ (the inverse is free — just transpose!).

This property is used extensively in eigendecomposition and SVD (Chapter 4).

---

## 3.6 The Matrix Inverse

For a square matrix $A$, its inverse $A^{-1}$ satisfies:

$$AA^{-1} = A^{-1}A = I$$

Think of it like the reciprocal: $3 \times \frac{1}{3} = 1$. The inverse "undoes" the transformation.

**Worked example — 2×2 inverse:**

For $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:

$$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

The quantity $ad - bc$ is called the **determinant**.

> **Trust this result:** This formula is derived by solving $AA^{-1} = I$ as a system of four equations. For $3\times3$ and larger matrices, the formula becomes far more complex (Cramer's rule, cofactor expansion). In practice, you never compute inverses by hand for matrices larger than $2\times2$ — that's what `np.linalg.inv` is for. **Do not try to generalize the 2×2 formula to 3×3** — it doesn't extend directly.

$$A = \begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix}$$

Step 1 — determinant: $\det(A) = (2)(3) - (1)(5) = 6 - 5 = 1$

Step 2 — inverse:
$$A^{-1} = \frac{1}{1} \begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix} = \begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix}$$

Step 3 — verify: $AA^{-1}$ should equal $I$:

$$\begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix} \begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix} = \begin{bmatrix} (6-5) & (-2+2) \\ (15-15) & (-5+6) \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I \checkmark$$

```python
A = np.array([[2.0, 1.0], [5.0, 3.0]])
A_inv = np.linalg.inv(A)
print(A_inv)
# [[ 3. -1.]
#  [-5.  2.]]
print(A @ A_inv)
# [[1. 0.]
#  [0. 1.]]  ← identity (rounding errors possible; use np.allclose)
print(np.allclose(A @ A_inv, np.eye(2)))  # True
```

### When Does the Inverse Not Exist?

When the determinant is zero: $\det(A) = 0$. This means the matrix "squashes" space — it collapses a dimension, so you can't undo it.

$$B = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}, \quad \det(B) = (1)(4) - (2)(2) = 4 - 4 = 0$$

$B$ has no inverse. This matrix maps all of $\mathbb{R}^2$ onto a single line, losing all information in one direction.

```python
B = np.array([[1.0, 2.0], [2.0, 4.0]])
print(np.linalg.det(B))  # ~0.0
# np.linalg.inv(B) would raise LinAlgError: Singular matrix
```

In ML, a **singular** (non-invertible) matrix is often a sign of **multicollinearity** — your features are linearly dependent.

---

## 3.7 The Determinant

The **determinant** of a square matrix is a single number that encodes whether the matrix is invertible and how it scales areas/volumes.

$$\det\left(\begin{bmatrix} a & b \\ c & d \end{bmatrix}\right) = ad - bc$$

**Geometric interpretation:** The determinant is the scale factor of the area transformation.

- $|\det(A)| = 2$: the matrix doubles areas
- $|\det(A)| = 0.5$: halves areas
- $\det(A) = 0$: squashes to zero area (singular matrix)
- $\det(A) < 0$: the transformation flips orientation (mirror)

```python
A = np.array([[3.0, 0.0], [0.0, 2.0]])  # scale x by 3, y by 2
print(np.linalg.det(A))  # 6.0 — areas scaled by 6x
```

**In ML:** The determinant appears in the formula for multivariate Gaussian distributions (Chapter 8) and in understanding if a system of equations has a unique solution.

---

## 3.8 Matrix Rank

The **rank** of a matrix is the number of linearly independent rows (or columns) — equivalently, the "true" dimensionality of the information the matrix contains.

For $A \in \mathbb{R}^{m \times n}$:
- Maximum possible rank: $\min(m, n)$
- **Full rank:** rank = $\min(m, n)$ — no redundant information
- **Rank-deficient:** rank < $\min(m, n)$ — some rows/columns are redundant

**Example:**

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 5 & 7 & 9 \end{bmatrix}$$

Row 3 = Row 1 + Row 2: $[5, 7, 9] = [1,2,3] + [4,5,6]$. So rank = 2, not 3.

```python
A = np.array([[1, 2, 3], [4, 5, 6], [5, 7, 9]], dtype=float)
print(np.linalg.matrix_rank(A))  # 2
```

**In ML:** Low-rank matrices appear everywhere:
- **Compressed representations:** images often have low-rank structure
- **Overfitting detection:** if your weight matrix becomes low rank during training, it's a sign of redundancy
- **LoRA** (Low-Rank Adaptation): the technique that makes fine-tuning LLMs cheap, by constraining weight updates to low-rank matrices

---

## 3.9 Systems of Linear Equations

A system of linear equations can always be written as $A\mathbf{x} = \mathbf{b}$:

**System:**
$$2x + y = 5$$
$$5x + 3y = 13$$

**Matrix form:**
$$\underbrace{\begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} x \\ y \end{bmatrix}}_{\mathbf{x}} = \underbrace{\begin{bmatrix} 5 \\ 13 \end{bmatrix}}_{\mathbf{b}}$$

**Solution:** If $A$ is invertible: $\mathbf{x} = A^{-1}\mathbf{b}$

$$\mathbf{x} = \begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix} \begin{bmatrix} 5 \\ 13 \end{bmatrix} = \begin{bmatrix} (15-13) \\ (-25+26) \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

Verify: $x=2, y=1$: $2(2)+1=5$ ✓ and $5(2)+3(1)=13$ ✓

```python
A = np.array([[2.0, 1.0], [5.0, 3.0]])
b = np.array([5.0, 13.0])

# Method 1: inverse (not recommended for large systems — numerically unstable)
x = np.linalg.inv(A) @ b
print(x)  # [2. 1.]

# Method 2: np.linalg.solve (preferred — uses numerically stable algorithms)
x = np.linalg.solve(A, b)
print(x)  # [2. 1.]
```

> **Practical note:** For large or ill-conditioned systems, prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b`. The solver uses LU decomposition — faster and more numerically stable. For small, well-conditioned systems (e.g., 3×3), the difference is negligible.

**Connection to ML:** Linear regression solves exactly this — find weights $\mathbf{w}$ such that $X\mathbf{w} \approx \mathbf{y}$. The closed-form solution (the Normal Equation) is:

$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

> **Warning:** This formula requires computing a matrix inverse, which is $O(n^3)$ and numerically unstable when features are nearly linearly dependent (making $X^TX$ nearly singular). In practice, scikit-learn and other libraries use gradient descent or QR decomposition instead. We'll discuss when the Normal Equation is and isn't appropriate in Chapter 9.

---

## 3.10 Broadcasting in NumPy

One NumPy feature every ML engineer needs to understand is **broadcasting** — it's NumPy's way of automatically expanding shapes to make operations work.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)
b = np.array([10, 20, 30]) # shape (3,)

# Adding a (3,) vector to a (2, 3) matrix:
# NumPy broadcasts b to [[10,20,30],[10,20,30]] automatically
result = A + b
print(result)
# [[11 22 33]
#  [14 25 36]]
```

**The broadcasting rule** — NumPy aligns shapes from the **right**:

1. If shapes have different number of dimensions, prepend 1s to the shorter shape
2. For each dimension, sizes must either be equal OR one of them must be 1
3. A dimension of size 1 gets "stretched" to match the other

```
A: (2, 3)
b:    (3,)   → treated as (1, 3) → stretched to (2, 3)  ✓ compatible

A: (2, 3)
c:    (2,)   → treated as (1, 2) → can't match (2, 3)   ✗ ERROR
```

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = np.array([10, 20, 30])             # (3,) → broadcasts to (2, 3)
c = np.array([10, 20])                 # (2,) → shapes (2,3) and (2,) → ERROR

print(A + b)   # works: [[11,22,33],[14,25,36]]
# print(A + c) # ValueError: operands could not be broadcast
# Fix: reshape c to a column vector
c_col = c.reshape(-1, 1)  # shape (2, 1) → broadcasts to (2, 3)
print(A + c_col)  # [[11,12,13],[24,25,26]]
```

This is used constantly for adding bias vectors in neural networks:

```python
# Adding bias b to all examples in a batch
W = np.random.randn(4, 3)   # weight matrix (4 output neurons, 3 inputs)
x_batch = np.random.randn(100, 3)  # 100 training examples, 3 features each
b = np.zeros(4)              # bias vector

# Without broadcasting, you'd need a loop.
# With broadcasting:
h = x_batch @ W.T + b       # shape: (100, 4)
# Each of the 100 examples gets the bias added automatically
```

---

## 3.11 Full Code Example

```python
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ─── Matrix basics ───────────────────────────────────────
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

print("=== Matrix Multiplication ===")
print("A @ B =\n", A @ B)
# [[19. 22.]
#  [43. 50.]]

print("\n=== Transpose ===")
print("A.T =\n", A.T)
# [[1. 3.]
#  [2. 4.]]

print("\n=== Determinant ===")
print("det(A) =", np.linalg.det(A))  # -2.0

print("\n=== Inverse ===")
A_inv = np.linalg.inv(A)
print("A_inv =\n", A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]
print("A @ A_inv =\n", A @ A_inv)  # identity

# ─── Rank ────────────────────────────────────────────────
print("\n=== Rank ===")
C_fullrank = np.array([[1, 0], [0, 1]])
C_degenerate = np.array([[1, 2], [2, 4]])
print("rank(I₂) =", np.linalg.matrix_rank(C_fullrank))       # 2
print("rank(degenerate) =", np.linalg.matrix_rank(C_degenerate))  # 1

# ─── Solving linear system ───────────────────────────────
print("\n=== Linear System Ax = b ===")
A_sys = np.array([[2.0, 1.0], [5.0, 3.0]])
b = np.array([5.0, 13.0])
x = np.linalg.solve(A_sys, b)
print("Solution x =", x)          # [2. 1.]
print("Verify Ax =", A_sys @ x)   # [5. 13.]

# ─── Neural network layer ────────────────────────────────
print("\n=== Neural Network Layer ===")
W = np.array([[0.5, -0.3,  0.8],   # weight matrix: 2 output neurons, 3 inputs
              [0.1,  0.9, -0.2]])
b_nn = np.array([0.1, -0.1])        # bias

x_input = np.array([1.0, 0.5, -1.0])  # input feature vector
h = W @ x_input + b_nn
print("h = W @ x + b =", h)
# h[0] = 0.5*1 + (-0.3)*0.5 + 0.8*(-1) + 0.1 = 0.5 - 0.15 - 0.8 + 0.1 = -0.35
# h[1] = 0.1*1 + 0.9*0.5 + (-0.2)*(-1) + (-0.1) = 0.1 + 0.45 + 0.2 - 0.1 = 0.65
```

Let's verify those neuron outputs manually:

```python
# Manual verification
h0 = 0.5*1.0 + (-0.3)*0.5 + 0.8*(-1.0) + 0.1
h1 = 0.1*1.0 + 0.9*0.5 + (-0.2)*(-1.0) + (-0.1)
print(f"h[0] = {h0:.4f}")  # -0.3500
print(f"h[1] = {h1:.4f}")  # 0.6500
```

---

## 3.12 Chapter Summary

| Concept | Definition | Code |
|---------|-----------|------|
| Matrix | $A \in \mathbb{R}^{m \times n}$ | `np.array([[...], [...]])` |
| Transpose | $(A^T)_{ij} = A_{ji}$ | `A.T` |
| Matrix multiply | $c_{ij} = \sum_l a_{il}b_{lj}$ | `A @ B` |
| Identity | $AI = A$ | `np.eye(n)` |
| Inverse | $AA^{-1} = I$ | `np.linalg.inv(A)` |
| Determinant | Scalar; 0 = singular | `np.linalg.det(A)` |
| Rank | # independent rows/cols | `np.linalg.matrix_rank(A)` |
| Solve $A\mathbf{x}=\mathbf{b}$ | $\mathbf{x} = A^{-1}\mathbf{b}$ | `np.linalg.solve(A, b)` |

**Key insights:**
1. Matrix multiplication = applying transformations in sequence
2. $AB \neq BA$ — order matters
3. A singular matrix ($\det = 0$) loses information irreversibly
4. Every neural network layer = one matrix multiplication + bias

---

## Exercises

**3.1** Compute $AB$ by hand:

$$A = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}, \quad B = \begin{bmatrix} 1 & 4 \\ 2 & 1 \end{bmatrix}$$

*Solution:*
$$c_{11} = 2(1)+0(2)=2, \quad c_{12}=2(4)+0(1)=8$$
$$c_{21} = 1(1)+3(2)=7, \quad c_{22}=1(4)+3(1)=7$$

$$AB = \begin{bmatrix} 2 & 8 \\ 7 & 7 \end{bmatrix}$$

```python
A = np.array([[2, 0], [1, 3]])
B = np.array([[1, 4], [2, 1]])
print(A @ B)  # [[ 2  8] [ 7  7]]
```

**3.2** Is $AB = BA$ for the matrices in 3.1? Compute $BA$ and compare.

*Solution:*
$$b_{11}=1(2)+4(1)=6, \quad b_{12}=1(0)+4(3)=12$$
$$b_{21}=2(2)+1(1)=5, \quad b_{22}=2(0)+1(3)=3$$

$$BA = \begin{bmatrix} 6 & 12 \\ 5 & 3 \end{bmatrix} \neq AB = \begin{bmatrix} 2 & 8 \\ 7 & 7 \end{bmatrix}$$

Matrix multiplication is **not** commutative.

**3.3** A neural network layer has weight matrix $W = \begin{bmatrix} 1 & -1 & 2 \\ 0 & 3 & -2 \end{bmatrix}$ and bias $\mathbf{b} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$. Compute the output for input $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$.

*Solution:*
$$W\mathbf{x} = \begin{bmatrix} 1(1)+(-1)(2)+2(3) \\ 0(1)+3(2)+(-2)(3) \end{bmatrix} = \begin{bmatrix} 1-2+6 \\ 0+6-6 \end{bmatrix} = \begin{bmatrix} 5 \\ 0 \end{bmatrix}$$

$$W\mathbf{x} + \mathbf{b} = \begin{bmatrix} 5.5 \\ -0.5 \end{bmatrix}$$

**3.4** Find the inverse of $A = \begin{bmatrix} 3 & 1 \\ 2 & 1 \end{bmatrix}$ by hand using the 2×2 formula.

*Solution:* $\det(A) = 3(1) - 1(2) = 1$

$$A^{-1} = \frac{1}{1}\begin{bmatrix} 1 & -1 \\ -2 & 3 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -2 & 3 \end{bmatrix}$$

Verify: $\begin{bmatrix}3&1\\2&1\end{bmatrix}\begin{bmatrix}1&-1\\-2&3\end{bmatrix} = \begin{bmatrix}3-2&-3+3\\2-2&-2+3\end{bmatrix} = \begin{bmatrix}1&0\\0&1\end{bmatrix}$ ✓

**3.5** (Challenge) Why is `np.linalg.solve(A, b)` preferred over `np.linalg.inv(A) @ b`?

*Answer:* `solve` uses LU decomposition — a numerically stable algorithm that avoids the amplification of floating-point errors that can occur when computing the explicit inverse. For large matrices, it's also significantly faster since computing the full inverse is $O(n^3)$ operations while solving is also $O(n^3)$ but with a smaller constant and without the intermediate storage.

---

*Next: Chapter 4 — Decompositions. We'll break matrices apart to reveal their hidden structure — and discover PCA.*
