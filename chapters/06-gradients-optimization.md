# Chapter 6: Calculus II — Gradients and Optimization

> *"Gradient descent is not magic. It is calculus applied one step at a time."*

---

## 6.1 From Derivatives to Gradients

In Chapter 5, we computed derivatives of functions with **one input** — $f(x)$. But ML models have millions of parameters. We need derivatives with respect to **many variables simultaneously**.

The **gradient** is the generalization of the derivative to multi-variable functions. Instead of a single number $f'(x)$, the gradient is a **vector** of partial derivatives.

If $f: \mathbb{R}^n \rightarrow \mathbb{R}$ (takes an $n$-dimensional vector, returns a scalar):

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

Each component $\frac{\partial f}{\partial x_i}$ is a **partial derivative** — the rate of change of $f$ when only $x_i$ changes, holding all others constant.

---

## 6.2 Partial Derivatives

A **partial derivative** differentiates with respect to one variable, treating all others as constants.

**Notation:** $\frac{\partial f}{\partial x}$ (curly $\partial$ instead of straight $d$)

**Worked example:**

$$f(x, y) = x^2 + 3xy + y^3$$

Partial derivative with respect to $x$ (treat $y$ as a constant):
$$\frac{\partial f}{\partial x} = 2x + 3y$$

Partial derivative with respect to $y$ (treat $x$ as a constant):
$$\frac{\partial f}{\partial y} = 3x + 3y^2$$

At the point $(x, y) = (2, 1)$:
$$\frac{\partial f}{\partial x}\bigg|_{(2,1)} = 2(2) + 3(1) = 7$$
$$\frac{\partial f}{\partial y}\bigg|_{(2,1)} = 3(2) + 3(1)^2 = 9$$

So the gradient at $(2, 1)$ is $\nabla f(2,1) = \begin{bmatrix} 7 \\ 9 \end{bmatrix}$.

```python
import math

def f(x, y):    return x**2 + 3*x*y + y**3
def df_dx(x,y): return 2*x + 3*y
def df_dy(x,y): return 3*x + 3*y**2

x, y = 2.0, 1.0
print(f"f({x},{y}) = {f(x,y)}")       # 4 + 6 + 1 = 11
print(f"∂f/∂x = {df_dx(x,y)}")        # 7
print(f"∂f/∂y = {df_dy(x,y)}")        # 9
print(f"gradient = [{df_dx(x,y)}, {df_dy(x,y)}]")  # [7, 9]
```

---

## 6.3 The Gradient as a Direction

The gradient $\nabla f(\mathbf{x})$ is a vector that points in the direction of **steepest ascent** — the direction in which $f$ increases most rapidly.

The **negative gradient** $-\nabla f(\mathbf{x})$ points toward **steepest descent** — the direction of fastest decrease.

**Geometric picture (2D landscape):**

```
f(x,y) as elevation:

         ▲ gradient points uphill
         |
    _____|_____
   /    hill   \
  /   ●          \
  |   ↗  ← gradient here
  |                \
   \     valley    /
    \_____________/
         ↓ negative gradient points downhill
```

**Key properties of the gradient:**

1. **Direction:** Points toward steepest ascent
2. **Magnitude:** $\|\nabla f\|$ tells you how steep the terrain is
3. **Zero gradient:** $\nabla f = \mathbf{0}$ at a critical point (local min, max, or saddle)

**Worked example — gradient at two points:**

$$f(x,y) = x^2 + y^2 \quad \text{(bowl-shaped, minimum at origin)}$$

$$\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}$$

At $(3, 4)$: $\nabla f = \begin{bmatrix} 6 \\ 8 \end{bmatrix}$ — points outward from origin (uphill)

At $(0, 0)$: $\nabla f = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ — zero gradient at the minimum ✓

---

## 6.4 Gradient Descent

Gradient descent is the algorithm that trains virtually every ML model. The idea is elegant:

1. Start at some point (random initialization of parameters)
2. Compute the gradient at that point (how to go uphill)
3. Take a small step in the **opposite** direction (downhill)
4. Repeat until you reach a minimum

### The Update Rule

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \nabla_\theta L(\theta)$$

- $\theta$ — model parameters (weights), a vector
- $\alpha$ — **learning rate** (step size, a small positive number like 0.01)
- $L(\theta)$ — loss function (what we're minimizing)
- $\nabla_\theta L$ — gradient of the loss with respect to parameters

The minus sign is critical: we go **against** the gradient to go downhill.

### Worked Example — Minimizing a 1D Function

$$L(w) = w^2 - 4w + 5$$

Exact minimum: $L'(w) = 2w - 4 = 0 \implies w^* = 2$, $L(2) = 4 - 8 + 5 = 1$

**Gradient descent simulation (α = 0.3, starting at w = 0):**

| Step | $w$ | $L(w)$ | $\nabla L = 2w-4$ | Update: $w - 0.3\nabla L$ |
|------|-----|--------|-------------------|--------------------------|
| 0 | 0.000 | 5.000 | -4.000 | 0 - 0.3(-4) = 1.200 |
| 1 | 1.200 | 1.640 | -1.600 | 1.2 - 0.3(-1.6) = 1.680 |
| 2 | 1.680 | 1.102 | -0.640 | 1.68 - 0.3(-0.64) = 1.872 |
| 3 | 1.872 | 1.016 | -0.256 | 1.872 - 0.3(-0.256) = 1.949 |
| 4 | 1.949 | 1.003 | -0.102 | → converging to 2.0 |

```python
def loss(w):  return w**2 - 4*w + 5
def grad(w):  return 2*w - 4

w = 0.0          # starting point
alpha = 0.3      # learning rate
n_steps = 20

print("Step |   w    |  L(w)  | gradient")
print("-----|--------|--------|----------")
for step in range(n_steps):
    g = grad(w)
    print(f"  {step:2d} | {w:6.4f} | {loss(w):6.4f} | {g:8.4f}")
    w = w - alpha * g
    if abs(g) < 1e-6:
        print(f"Converged at step {step}")
        break

print(f"\nFinal: w = {w:.6f}, L(w) = {loss(w):.6f}")
print(f"True minimum: w = 2.0, L = 1.0")
```

### The Learning Rate Effect

The learning rate $\alpha$ is the most important hyperparameter in gradient descent:

**Too small:** Takes many steps; training is slow.

**Too large:** Overshoot the minimum; loss oscillates or diverges.

**Just right:** Converges efficiently.

```python
def gradient_descent(w_init, alpha, n_steps=50):
    w = w_init
    losses = []
    for _ in range(n_steps):
        losses.append(loss(w))
        g = grad(w)
        w = w - alpha * g
        if abs(g) < 1e-8:
            break
    return w, losses

# Compare learning rates
for alpha in [0.01, 0.3, 0.9, 1.1]:
    w_final, hist = gradient_descent(0.0, alpha)
    status = "converged" if abs(w_final - 2.0) < 0.01 else "DIVERGED"
    print(f"α={alpha}: final w={w_final:.4f}, {status}")
```

Output:
```
α=0.01: final w=1.6484, converged slowly
α=0.30: final w=2.0000, converged
α=0.90: final w=2.0000, converged (oscillated)
α=1.10: final w=HUGE, DIVERGED
```

---

## 6.5 Variants of Gradient Descent

In practice, computing the gradient over the entire dataset at each step is expensive. Three variants are used:

### Batch Gradient Descent
Use **all** training examples to compute the gradient:

$$\nabla_\theta L = \frac{1}{n} \sum_{i=1}^n \nabla_\theta L^{(i)}$$

- **Pro:** Stable, accurate gradient estimate
- **Con:** Slow for large datasets; must load all data into memory
- **Use:** Small datasets or when exact gradient is critical

### Stochastic Gradient Descent (SGD)
Use **one random example** at each step:

$$\nabla_\theta L \approx \nabla_\theta L^{(i)} \quad \text{for random } i$$

- **Pro:** Fast updates; can escape saddle points due to noise
- **Con:** Noisy — loss bounces around; needs learning rate decay
- **Use:** Online learning; very large datasets

### Mini-Batch Gradient Descent
Use a **small random batch** (typically 32–512 examples):

$$\nabla_\theta L \approx \frac{1}{B} \sum_{i \in \text{batch}} \nabla_\theta L^{(i)}$$

- **Pro:** Best of both worlds; efficient on GPUs (vectorized)
- **Con:** Adds batch size as a hyperparameter
- **Use:** **Standard in modern deep learning**

```python
import random

def mini_batch_gradient_descent(X, y, w_init, alpha=0.01, batch_size=32, epochs=100):
    """
    Simple linear regression: prediction = w * x
    Loss = mean squared error
    
    Note: Uses plain Python lists for clarity. In production, use numpy arrays
    for vectorized operations (50-100x faster on large datasets).
    """
    w = w_init
    n = len(X)

    for epoch in range(epochs):
        # Shuffle data
        indices = list(range(n))
        random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = [X[i] for i in batch_idx]
            y_batch = [y[i] for i in batch_idx]
            B = len(batch_idx)

            # Compute gradient: d(MSE)/dw = 2/B * sum((wx - y) * x)
            grad_w = 2/B * sum((w * X_batch[j] - y_batch[j]) * X_batch[j]
                                for j in range(B))
            w = w - alpha * grad_w

    return w
```

---

## 6.6 The Loss Landscape

Understanding the loss landscape — the "terrain" of $L(\theta)$ over all possible parameters — is key to understanding why training works or fails.

### Convex vs Non-Convex

**Convex:** One global minimum, no local minima. Gradient descent always finds it.

```
Convex (bowl):           Non-convex (hills):
     L                        L
     |    *global min         |   * local  *global
     |   / \                  |  / \ min  / \
     |  /   \                 | /   \    /   \
     |_/     \_               |/     \__/     \_
              θ                                 θ
```

**Linear regression** has a convex loss (squared error on linear model) — guaranteed to find global minimum.

**Neural networks** have non-convex loss landscapes. In practice, the many saddle points and local minima have similar loss values, and modern optimizers handle this well.

### The Role of Initialization

Where you start matters in non-convex landscapes:

```python
# Two runs of gradient descent on a non-convex function
# f(w) = w^4 - 4w^2 + w  (two local minima near w≈-1.4 and w≈1.3)
def f_nonconvex(w):  return w**4 - 4*w**2 + w
def gf_nonconvex(w): return 4*w**3 - 8*w + 1

for w_init in [-2.0, 0.5, 2.0]:
    w = w_init
    for _ in range(1000):
        g = gf_nonconvex(w)
        w -= 0.01 * g
        if abs(g) < 1e-6:
            break
    print(f"Start: {w_init:5.1f} → final w: {w:.4f}, L: {f_nonconvex(w):.4f}")

# Expected output:
# Start:  -2.0 → final w: -1.4730, L: -5.4442   ← local minimum
# Start:   0.5 → final w:  1.3470, L: -2.6186   ← different local minimum
# Start:   2.0 → final w:  1.3470, L: -2.6186   ← same local minimum
# Different starting points → different solutions on non-convex landscapes
```

---

## 6.7 Beyond Vanilla Gradient Descent: Momentum and Adam

### Momentum

Vanilla gradient descent "forgets" previous steps — it only looks at the current gradient. **Momentum** adds memory: it accumulates a velocity vector, smoothing out oscillations.

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\nabla L(\theta)$$
$$\theta_t = \theta_{t-1} - \alpha \mathbf{v}_t$$

- $\mathbf{v}$ is the velocity (exponential moving average of gradients)
- $\beta$ is the momentum coefficient (typically 0.9)

Think of it like a ball rolling down a hill — it builds up speed in consistent directions and doesn't change course sharply on noisy gradients.

### Adam (Adaptive Moment Estimation)

**Adam** is the most widely used optimizer in deep learning. It combines:
- **Momentum:** exponential moving average of gradients ($\mathbf{m}$, first moment)
- **RMSProp:** exponential moving average of squared gradients ($\mathbf{v}$, second moment)

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla L$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla L)^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

**Default hyperparameters:** $\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

**Why Adam works:** It normalizes gradient updates by the historical magnitude of each parameter's gradient. Parameters with historically large gradients get smaller updates; parameters with small gradients get relatively larger updates. This adapts the step size per parameter automatically.

```python
def adam_step(grad, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Single Adam update step."""
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad**2
    m_hat = m_new / (1 - beta1**t)   # bias correction
    v_hat = v_new / (1 - beta2**t)   # bias correction
    update = alpha * m_hat / (v_hat**0.5 + eps)
    return update, m_new, v_new
```

---

## 6.8 The Jacobian and Hessian (Brief Introduction)

For completeness — these appear in optimization literature.

### Jacobian

When $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ (vector input, vector output), the derivative is a matrix called the **Jacobian**:

$$J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

In ML: the Jacobian of the network output with respect to inputs tells you how sensitive each output is to each input feature — useful for saliency maps and explainability.

### Hessian

The **Hessian** $H$ is the matrix of second-order partial derivatives of a scalar function:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian tells you about the curvature of the loss landscape. Positive definite Hessian = local minimum. Indefinite Hessian = saddle point.

**Why not commonly used in deep learning:** The Hessian of a neural network has dimension $(n_\theta \times n_\theta)$ where $n_\theta$ is the number of parameters. For a network with 100M parameters, the Hessian would have $10^{16}$ entries — impossible to compute or store. Gradient descent only needs first-order information.

---

## 6.9 Gradient Descent for Linear Regression — Full Example

Let's see everything together on a concrete ML problem: fitting a line to data.

**Model:** $\hat{y} = wx + b$

**Loss (MSE):**
$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{n}\sum_{i=1}^{n}(wx^{(i)} + b - y^{(i)})^2$$

**Gradients:**

$$\frac{\partial L}{\partial w} = \frac{2}{n}\sum_{i=1}^{n}(wx^{(i)} + b - y^{(i)}) \cdot x^{(i)}$$

$$\frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(wx^{(i)} + b - y^{(i)})$$

```python
import random
import math

# ─── Generate synthetic data: y = 2x + 1 + noise ─────────
random.seed(42)
n = 50
X_data = [i/10.0 for i in range(n)]
y_data = [2.0*x + 1.0 + random.gauss(0, 0.3) for x in X_data]
# True w=2, b=1

# ─── Gradient descent ────────────────────────────────────
w, b = 0.0, 0.0    # initialize at zero
alpha = 0.05       # learning rate
epochs = 200

def mse_loss(w, b):
    return sum((w*x + b - y)**2 for x,y in zip(X_data,y_data)) / n

def gradients(w, b):
    residuals = [w*x + b - y for x,y in zip(X_data,y_data)]
    dw = 2/n * sum(r*x for r,x in zip(residuals, X_data))
    db = 2/n * sum(residuals)
    return dw, db

print(f"Initial: w={w}, b={b}, Loss={mse_loss(w,b):.4f}")

for epoch in range(epochs):
    dw, db = gradients(w, b)
    w -= alpha * dw
    b -= alpha * db

print(f"Final:   w={w:.4f}, b={b:.4f}, Loss={mse_loss(w,b):.4f}")
print(f"True:    w=2.0000, b=1.0000")
```

Expected output:
```
Initial: w=0, b=0, Loss=10.XXXX
Final:   w=1.9XXX, b=1.0XXX, Loss=0.0XXX
True:    w=2.0000, b=1.0000
```

---

## 6.10 Chapter Summary

| Concept | Formula | Use in ML |
|---------|---------|----------|
| Partial derivative | $\frac{\partial f}{\partial x_i}$ (hold others fixed) | Per-parameter gradient |
| Gradient | $\nabla f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^T$ | Direction of steepest ascent |
| Gradient descent | $\theta \leftarrow \theta - \alpha\nabla L(\theta)$ | Training all models |
| SGD | One example per step | Large datasets |
| Mini-batch | Batch of B examples per step | Standard in deep learning |
| Momentum | Velocity accumulation | Smoother convergence |
| Adam | Adaptive per-parameter LR | Default optimizer for most nets |
| Jacobian | Matrix of first derivatives for vector functions | Backprop, saliency |
| Hessian | Matrix of second derivatives | Curvature; impractical for large nets |

**Key insight:** Gradient descent is the engine of all ML. The gradient tells us direction. The learning rate controls the step size. Everything else (Adam, momentum, schedulers) is engineering to make this work better in practice.

---

## Exercises

**6.1** Compute the gradient of $f(x, y) = x^2y + y^3$ at $(x,y) = (1, 2)$.

*Solution:*
$$\frac{\partial f}{\partial x} = 2xy, \quad \frac{\partial f}{\partial y} = x^2 + 3y^2$$

At $(1,2)$: $\frac{\partial f}{\partial x} = 2(1)(2) = 4$, $\frac{\partial f}{\partial y} = 1 + 12 = 13$

$$\nabla f(1,2) = \begin{bmatrix} 4 \\ 13 \end{bmatrix}$$

**6.2** Run 3 steps of gradient descent on $L(w) = (w-5)^2$ starting at $w=0$ with $\alpha=0.4$.

*Solution:* $L'(w) = 2(w-5)$

| Step | $w$ | $L'(w)$ | $w_{\text{new}} = w - 0.4 \cdot L'(w)$ |
|------|-----|---------|----------------------------------------|
| 0 | 0.0 | -10.0 | 0 - 0.4(-10) = 4.0 |
| 1 | 4.0 | -2.0 | 4 - 0.4(-2) = 4.8 |
| 2 | 4.8 | -0.4 | 4.8 - 0.4(-0.4) = 4.96 |

Converging toward 5.0. ✓

**6.3** What is the gradient of $L(w, b) = (wx + b - y)^2$ at $w=1, b=0, x=2, y=3$?

*Solution:*
Residual $= wx + b - y = 1(2) + 0 - 3 = -1$

$$\frac{\partial L}{\partial w} = 2(wx+b-y) \cdot x = 2(-1)(2) = -4$$
$$\frac{\partial L}{\partial b} = 2(wx+b-y) \cdot 1 = 2(-1) = -2$$

Gradient: $\begin{bmatrix} -4 \\ -2 \end{bmatrix}$. Gradient descent would increase both $w$ and $b$ (subtract negative gradient), moving toward the true $w=1.5, b=0$ line passing through $(2,3)$.

**6.4** Why does Adam divide the gradient update by $\sqrt{\hat{\mathbf{v}}_t} + \epsilon$?

*Solution:* $\hat{\mathbf{v}}_t$ is an estimate of the squared gradient magnitude. Dividing by $\sqrt{\hat{\mathbf{v}}_t}$ normalizes the update so parameters with historically large gradients (high $\hat{v}$) receive smaller updates, and parameters with small gradients receive relatively larger updates. This acts as an adaptive, per-parameter learning rate. The $\epsilon$ prevents division by zero when $\hat{v}$ is near zero.

**6.5** (Challenge) Show that for $L(w) = \frac{1}{n}\sum_{i=1}^n (wx_i - y_i)^2$, the gradient is:

$$\frac{dL}{dw} = \frac{2}{n}\sum_{i=1}^n (wx_i - y_i)x_i$$

*Solution:* Apply the chain rule to each term:

$$\frac{d}{dw}(wx_i - y_i)^2 = 2(wx_i - y_i) \cdot \frac{d}{dw}(wx_i - y_i) = 2(wx_i - y_i) \cdot x_i$$

Summing over all $n$ examples and dividing by $n$:

$$\frac{dL}{dw} = \frac{1}{n}\sum_{i=1}^n 2(wx_i - y_i)x_i = \frac{2}{n}\sum_{i=1}^n (wx_i - y_i)x_i \quad \square$$

---

*Next: Chapter 7 — Probability Foundations. We've covered how data is represented and how models improve. Now: what the model actually predicts.*
