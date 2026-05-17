# Chapter 5: Calculus I — Derivatives

> *"The derivative tells you which way is downhill. Gradient descent is just repeatedly following that direction."*

---

## 5.1 The Concept: What Is a Derivative?

Imagine you're driving on a hilly road. At any point, you can ask: "Am I currently going uphill or downhill, and how steep is it?" The answer is the **derivative**.

More precisely, the derivative of a function $f$ at a point $x$ is the **instantaneous rate of change** of $f$ at $x$ — the slope of the function at that exact point.

This is the foundation of how models learn. The loss function tells you how wrong your model is. The derivative tells you which way to adjust the parameters to make it less wrong.

### From Average Rate of Change to Instantaneous

The **average rate of change** of $f$ between $x$ and $x + h$ is:

$$\frac{f(x + h) - f(x)}{h}$$

This is the slope of the line connecting two points. As we make $h$ smaller and smaller, this approaches the **instantaneous rate of change** — the derivative:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

The symbol $\lim_{h \to 0}$ means "as $h$ approaches zero" — we're making the gap infinitesimally small.

**Visual intuition:**

```
f(x)
   |         ●  ← f(x+h)
   |        /
   |       /  ← slope = [f(x+h)-f(x)]/h
   |      ●  ← f(x)
   |___________ x
            x  x+h

As h→0, the secant line becomes a tangent line — that's the derivative.
```

---

## 5.2 Notation

There are several equivalent ways to write "derivative of $f$":

| Notation | Reads as | Notes |
|----------|----------|-------|
| $f'(x)$ | "f prime of x" | Lagrange notation — common for single-variable |
| $\frac{df}{dx}$ | "df by dx" | Leibniz notation — shows what's varying |
| $\frac{d}{dx}f(x)$ | "d by dx of f(x)" | Leibniz, explicit |
| $\dot{f}$ | "f dot" | Newton notation — common in physics (time derivatives) |

In ML, you'll most often see $\frac{df}{dx}$ or $\frac{\partial f}{\partial x}$ (the partial derivative — Chapter 6).

---

## 5.3 Basic Derivative Rules

You don't need to compute limits every time. These rules handle 95% of ML calculus.

### Rule 1: Power Rule

$$\frac{d}{dx} x^n = n \cdot x^{n-1}$$

**Examples:**

| Function | Derivative | Shown |
|----------|-----------|-------|
| $x^2$ | $2x$ | $n=2$: $2 \cdot x^{2-1} = 2x$ |
| $x^3$ | $3x^2$ | $n=3$: $3 \cdot x^{3-1} = 3x^2$ |
| $x^1 = x$ | $1$ | $n=1$: $1 \cdot x^{0} = 1$ |
| $x^0 = 1$ | $0$ | $n=0$: $0 \cdot x^{-1} = 0$ |
| $x^{-1} = 1/x$ | $-x^{-2} = -1/x^2$ | $n=-1$ |
| $x^{1/2} = \sqrt{x}$ | $\frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$ | $n=1/2$ |

**Worked example:**

$$f(x) = x^4$$
$$f'(x) = 4x^3$$

At $x=2$: $f'(2) = 4(2)^3 = 32$. The function is rising at a rate of 32 units per unit of $x$ at this point.

### Rule 2: Constant Rule

$$\frac{d}{dx} c = 0 \quad \text{(c is any constant)}$$

A constant never changes, so its rate of change is zero.

### Rule 3: Constant Multiple Rule

$$\frac{d}{dx} [c \cdot f(x)] = c \cdot f'(x)$$

**Example:** $\frac{d}{dx}[5x^2] = 5 \cdot 2x = 10x$

### Rule 4: Sum/Difference Rule

$$\frac{d}{dx}[f(x) \pm g(x)] = f'(x) \pm g'(x)$$

**Example:**
$$\frac{d}{dx}[3x^3 - 2x^2 + 7x - 5] = 9x^2 - 4x + 7$$

(Differentiate term by term; the constant $-5$ disappears.)

### Rule 5: Product Rule

$$\frac{d}{dx}[f(x) \cdot g(x)] = f'(x)g(x) + f(x)g'(x)$$

**Memory trick:** "First times derivative of second, plus second times derivative of first"

**Worked example:**

$$h(x) = x^2 \cdot \sin(x)$$
$$h'(x) = 2x \cdot \sin(x) + x^2 \cdot \cos(x)$$

**ML use:** The product rule appears in attention mechanisms. The scaled dot-product attention score $\text{score}(q, k) = q^T k / \sqrt{d}$ involves a product of two learned vectors; when computing gradients through both $q$ and $k$, the product rule applies.

### Rule 6: Chain Rule (CRITICAL for ML)

The chain rule computes the derivative of a **composed** function.

If $h(x) = f(g(x))$, then:

$$h'(x) = f'(g(x)) \cdot g'(x)$$

**In words:** "derivative of outer function (evaluated at inner function) times derivative of inner function"

**Leibniz notation** (clearer for ML):

$$\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Worked example:**

$$h(x) = (3x + 1)^5$$

Here $f(u) = u^5$ and $g(x) = 3x + 1$:
- $f'(u) = 5u^4$
- $g'(x) = 3$

$$h'(x) = 5(3x+1)^4 \cdot 3 = 15(3x+1)^4$$

**Another example with nesting:**

$$h(x) = \sin(x^2)$$

- Outer: $f(u) = \sin(u)$, $f'(u) = \cos(u)$
- Inner: $g(x) = x^2$, $g'(x) = 2x$

$$h'(x) = \cos(x^2) \cdot 2x$$

**Why the chain rule dominates ML:** A neural network is a composition of functions: $\hat{y} = f_L(f_{L-1}(\cdots f_1(\mathbf{x})))$. Backpropagation — the algorithm that trains neural networks — is just the chain rule applied repeatedly from output to input.

---

## 5.4 Derivatives of Common Functions

| Function | Derivative | Why it matters in ML |
|----------|-----------|---------------------|
| $\sin(x)$ | $\cos(x)$ | Fourier features, positional encodings |
| $\cos(x)$ | $-\sin(x)$ | Same |
| $e^x$ | $e^x$ | **Self-derivative!** Used in softmax, exponential loss |
| $e^{ax}$ | $ae^{ax}$ | Scaled exponential |
| $\ln(x)$, $x > 0$ | $1/x$ | Cross-entropy loss contains $\ln$ |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid activation, logistic regression |

### The Sigmoid Derivative — Worked

This appears in every logistic regression and sigmoid-activated network.

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

Let's compute $\sigma'(x)$ using the quotient rule and chain rule:

$$\sigma(x) = (1 + e^{-x})^{-1}$$

Using chain rule with $u = 1 + e^{-x}$:

$$\frac{d\sigma}{dx} = -1 \cdot (1+e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}$$

Now we simplify using a trick. Note that $\sigma(x) = \frac{1}{1+e^{-x}}$ so $1 - \sigma(x) = \frac{e^{-x}}{1+e^{-x}}$:

$$\frac{d\sigma}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$$

The sigmoid derivative is expressible entirely in terms of $\sigma(x)$ itself! This is computationally efficient in backpropagation — you already computed $\sigma(x)$ in the forward pass, so the backward pass needs no new function calls. (Compare to $e^x$, which is the only function that literally equals its own derivative; sigmoid is different but similarly convenient.)

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Test at several points
for x in [-2, 0, 1, 2]:
    s = sigmoid(x)
    ds = sigmoid_deriv(x)
    print(f"x={x:3d}: σ(x)={s:.4f}, σ'(x)={ds:.4f}")
```

Output:
```
x= -2: σ(x)=0.1192, σ'(x)=0.1050
x=  0: σ(x)=0.5000, σ'(x)=0.2500  ← maximum gradient at x=0
x=  1: σ(x)=0.7311, σ'(x)=0.1966
x=  2: σ(x)=0.8808, σ'(x)=0.1050
```

Note: the gradient is maximum (0.25) at $x=0$ and decreases as $|x|$ grows. This is the root cause of the **vanishing gradient problem** — for very negative or positive pre-activations, the sigmoid gradient is nearly zero, and gradients stop flowing through the network. Chapter 10 covers why ReLU largely replaced sigmoid to address this issue.

---

## 5.5 Critical Points: Minima, Maxima, Saddle Points

Training an ML model means **minimizing** a loss function. Calculus tells us where minima live.

**Critical points** occur where $f'(x) = 0$ — the function is momentarily flat.

### Types of Critical Points

```
Local maximum:        Local minimum:        Saddle point:
    ●                                         
   / \                    ●                   ●  ←f'=0, but not min/max
  /   \         f'=0 → / \                  / \
          f'=0 →     /   \                /   \
                    ●     (min)
```

For a differentiable function:
- $f'(x) = 0$ AND $f''(x) > 0$: **local minimum** (curve is concave up)
- $f'(x) = 0$ AND $f''(x) < 0$: **local maximum** (curve is concave down)
- $f'(x) = 0$ AND $f''(x) = 0$: **inconclusive** — the second derivative test fails; the point could be a local min, local max, or saddle point. Higher-order analysis is required.

### The Second Derivative

The second derivative $f''(x)$ is the derivative of the derivative — it measures how the slope is changing (curvature):

- $f''(x) > 0$: slope is increasing → concave up → like a valley
- $f''(x) < 0$: slope is decreasing → concave down → like a hill

### Worked Example

$$f(x) = x^3 - 3x + 2$$

**Step 1:** Find critical points — set $f'(x) = 0$:
$$f'(x) = 3x^2 - 3 = 0$$
$$3x^2 = 3 \implies x^2 = 1 \implies x = \pm 1$$

**Step 2:** Classify using second derivative:
$$f''(x) = 6x$$

- At $x = 1$: $f''(1) = 6 > 0$ → **local minimum**. $f(1) = 1 - 3 + 2 = 0$
- At $x = -1$: $f''(-1) = -6 < 0$ → **local maximum**. $f(-1) = -1 + 3 + 2 = 4$

```python
import math

def f(x):    return x**3 - 3*x + 2
def fp(x):   return 3*x**2 - 3
def fpp(x):  return 6*x

# Critical points
for x_crit in [-1, 1]:
    print(f"x={x_crit}: f={f(x_crit)}, f'={fp(x_crit)}, f''={fpp(x_crit)}")
    typ = "minimum" if fpp(x_crit) > 0 else "maximum"
    print(f"  → {typ}")
```

### Why Saddle Points Matter in ML

In high dimensions (millions of weights), true local minima are rare. Most critical points are **saddle points** — they're local minima in some directions but local maxima in others. 

Research shows that for large neural networks, most saddle points have similar loss values to the global minimum — so gradient descent naturally finds good solutions even without reaching a true minimum.

---

## 5.6 The Chain Rule in Detail: Backpropagation Preview

Let's trace the chain rule through a tiny 2-layer network to preview how backpropagation works.

**Network setup:**

```
Input x → [Layer 1: z₁ = wx + b] → [Activation: a₁ = σ(z₁)] → [Layer 2: z₂ = v·a₁] → Loss: L = (y - z₂)²
```

We want $\frac{dL}{dw}$ — how the loss changes with respect to weight $w$ in Layer 1. (We omit bias $b$ from the computation below for clarity, but note that $\frac{\partial z_1}{\partial b} = 1$, so the gradient of $L$ with respect to $b$ is the same chain product without the final $\frac{dz_1}{dw} = x$ factor — just $dL/dz_2 \cdot dz_2/da_1 \cdot da_1/dz_1 \cdot 1$.)

**Forward pass (compute intermediate values):**
- $z_1 = wx$ (linear transformation, bias omitted for clarity)
- $a_1 = \sigma(z_1)$ (sigmoid activation)
- $z_2 = v \cdot a_1$ (second layer)
- $L = (y - z_2)^2$ (loss)

**Backward pass (apply chain rule repeatedly):**

$$\frac{dL}{dw} = \frac{dL}{dz_2} \cdot \frac{dz_2}{da_1} \cdot \frac{da_1}{dz_1} \cdot \frac{dz_1}{dw}$$

Compute each factor:
1. $\frac{dL}{dz_2} = -2(y - z_2)$
2. $\frac{dz_2}{da_1} = v$
3. $\frac{da_1}{dz_1} = \sigma(z_1)(1 - \sigma(z_1))$
4. $\frac{dz_1}{dw} = x$

$$\frac{dL}{dw} = -2(y - z_2) \cdot v \cdot \sigma(z_1)(1-\sigma(z_1)) \cdot x$$

**Worked numerical example:**

Let $x = 1.0$, $w = 0.5$, $v = 2.0$, $y = 3.0$:

```python
# Forward pass
x, w, v, y_true = 1.0, 0.5, 2.0, 3.0

z1 = w * x                      # 0.5
a1 = sigmoid(z1)                 # σ(0.5) ≈ 0.6225
z2 = v * a1                      # 2 × 0.6225 ≈ 1.2450
L  = (y_true - z2) ** 2         # (3 - 1.245)² ≈ 3.0806

print(f"z1={z1:.4f}, a1={a1:.4f}, z2={z2:.4f}, L={L:.4f}")

# Backward pass: chain rule
dL_dz2  = -2 * (y_true - z2)                    # ≈ 3.5100
dz2_da1 = v                                       # 2.0
da1_dz1 = sigmoid_deriv(z1)                       # σ(0.5)(1-σ(0.5)) ≈ 0.2350
dz1_dw  = x                                       # 1.0

dL_dw = dL_dz2 * dz2_da1 * da1_dz1 * dz1_dw
print(f"dL/dw = {dL_dw:.4f}")  # ≈ -1.6498
```

This number $\frac{dL}{dw} \approx -1.65$ tells us: increase $w$ by a tiny amount, and the loss *decreases* by ~1.65 (negative gradient = loss goes down with larger $w$). To **reduce** the loss, move $w$ in the direction of the negative gradient — in this case, *increase* $w$. That's gradient descent.

---

## 5.7 Numerical Differentiation: Verifying Derivatives

When you're unsure about a derivative, you can approximate it numerically using the definition:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

(Using the symmetric "central difference" formula — more accurate than the one-sided version.)

```python
def numerical_gradient(f, x, h=1e-5):
    """Approximate the derivative of f at x using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test on f(x) = x^3 - 3x + 2
f = lambda x: x**3 - 3*x + 2
fp_exact = lambda x: 3*x**2 - 3

for x in [-2, 0, 1, 2]:
    numerical = numerical_gradient(f, x)
    exact     = fp_exact(x)
    print(f"x={x}: numerical={numerical:.6f}, exact={exact:.6f}, diff={abs(numerical-exact):.2e}")
```

Output:
```
x=-2: numerical=9.000000, exact=9.000000, diff=2.84e-11
x= 0: numerical=-3.000000, exact=-3.000000, diff=1.18e-11
x= 1: numerical=0.000000, exact=0.000000, diff=4.44e-16
x= 2: numerical=9.000000, exact=9.000000, diff=4.26e-11
```

This technique — called **gradient checking** — was commonly used to verify backpropagation implementations before automatic differentiation (autograd) became standard.

---

## 5.8 Full Code Example

```python
import math

# ─── Basic derivatives (numerical verification) ──────────
def num_grad(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

print("=== Power Rule Verification ===")
for n in [2, 3, 4]:
    f  = lambda x, n=n: x**n
    fp = lambda x, n=n: n * x**(n-1)
    x = 2.5
    print(f"f(x)=x^{n}: exact f'(x)={fp(x):.4f}, numerical={num_grad(f, x):.4f}")

print("\n=== Chain Rule: h(x) = (3x+1)^5 ===")
h    = lambda x: (3*x + 1)**5
hp   = lambda x: 15 * (3*x + 1)**4
x = 0.5
print(f"Exact: {hp(x):.4f}, Numerical: {num_grad(h, x):.4f}")

print("\n=== Sigmoid and its Derivative ===")
def sigmoid(x): return 1 / (1 + math.exp(-x))
def sig_d(x):
    s = sigmoid(x)
    return s * (1 - s)

sig_fn = sigmoid
for x in [-2, -1, 0, 1, 2]:
    s  = sigmoid(x)
    sd = sig_d(x)
    nd = num_grad(sig_fn, x)
    print(f"x={x:2d}: σ={s:.4f}, σ'={sd:.4f}, numerical={nd:.4f}")

print("\n=== Finding minimum of f(x) = x² + 4x + 5 ===")
# Analytical: f'(x) = 2x + 4 = 0 → x = -2
f  = lambda x: x**2 + 4*x + 5
fp = lambda x: 2*x + 4
x_min = -4 / 2  # = -2
print(f"Minimum at x={x_min}, f(x_min)={f(x_min)}")
print(f"f'(x_min) = {fp(x_min)} (should be 0)")

print("\n=== Mini Backprop Demo ===")
x_in, w, v, y_true = 1.0, 0.5, 2.0, 3.0

z1 = w * x_in
a1 = sigmoid(z1)
z2 = v * a1
L  = (y_true - z2)**2

dL_dz2  = -2 * (y_true - z2)
dz2_da1 = v
da1_dz1 = sig_d(z1)
dz1_dw  = x_in
dL_dw   = dL_dz2 * dz2_da1 * da1_dz1 * dz1_dw

print(f"Forward: z1={z1:.4f}, a1={a1:.4f}, z2={z2:.4f}, L={L:.4f}")
print(f"dL/dw = {dL_dw:.6f}")

# Verify numerically
L_fn = lambda w_val: (y_true - v * sigmoid(w_val * x_in))**2
grad_check = num_grad(L_fn, w)
print(f"Gradient check: {grad_check:.6f}")
print(f"Match: {abs(dL_dw - grad_check) < 1e-4}")
```

---

## 5.9 Chapter Summary

| Concept | Formula | Use in ML |
|---------|---------|----------|
| Derivative definition | $f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$ | Foundation of all optimization |
| Power rule | $\frac{d}{dx}x^n = nx^{n-1}$ | Polynomial loss functions |
| Chain rule | $\frac{dh}{dx} = \frac{df}{dg}\frac{dg}{dx}$ | Backpropagation |
| $e^x$ derivative | $\frac{d}{dx}e^x = e^x$ | Softmax, exp loss |
| $\ln(x)$ derivative | $\frac{d}{dx}\ln(x) = 1/x$ | Cross-entropy gradient |
| Sigmoid derivative | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ | Logistic regression, RNNs |
| Critical point | $f'(x) = 0$ | Where minima/maxima live |
| Numerical gradient | $\frac{f(x+h)-f(x-h)}{2h}$ | Gradient checking |

**Key insight:** Backpropagation = chain rule applied backward through the network. Every "gradient" in ML is a derivative computed by the chain rule.

---

## Exercises

**5.1** Compute the derivative of $f(x) = 4x^3 - 2x^2 + 5x - 7$.

*Solution:*
$$f'(x) = 12x^2 - 4x + 5$$

(Power rule + sum rule + constant disappears)

**5.2** Find all critical points of $f(x) = x^2 - 6x + 9$ and classify them.

*Solution:*
$$f'(x) = 2x - 6 = 0 \implies x = 3$$
$$f''(x) = 2 > 0 \implies \text{local minimum at } x=3$$
$$f(3) = 9 - 18 + 9 = 0$$

(This is $(x-3)^2$, a perfect square — minimum value is 0 at $x=3$.)

**5.3** Use the chain rule to differentiate $h(x) = \ln(x^2 + 1)$.

*Solution:*
- Outer: $f(u) = \ln(u)$, $f'(u) = 1/u$
- Inner: $g(x) = x^2 + 1$, $g'(x) = 2x$

$$h'(x) = \frac{1}{x^2+1} \cdot 2x = \frac{2x}{x^2+1}$$

```python
h  = lambda x: math.log(x**2 + 1)
hp = lambda x: 2*x / (x**2 + 1)
x = 3.0
print(f"Exact: {hp(x):.6f}")
print(f"Numerical: {num_grad(h, x):.6f}")
```

**5.4** The ReLU activation function is $\text{ReLU}(x) = \max(0, x)$. What is its derivative?

*Solution:*

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, the gradient at $x=0$ is defined to be 0 (or sometimes 1, implementation-dependent). ReLU's derivative is called the **Heaviside step function**. The simplicity of this derivative is a key reason ReLU replaced sigmoid as the default activation — no $\sigma(1-\sigma)$ computation needed, just 0 or 1.

```python
def relu(x):    return max(0.0, x)
def relu_d(x):  return 1.0 if x > 0 else 0.0

for x in [-2, -0.1, 0, 0.1, 2]:
    print(f"x={x}: ReLU={relu(x)}, ReLU'={relu_d(x)}")
```

**5.5** (Challenge) Show that $\frac{d}{dx}\ln(\sigma(x)) = 1 - \sigma(x)$.

*Solution:*
Using chain rule: $\frac{d}{dx}\ln(\sigma(x)) = \frac{1}{\sigma(x)} \cdot \sigma'(x) = \frac{1}{\sigma(x)} \cdot \sigma(x)(1-\sigma(x)) = 1 - \sigma(x)$ $\square$

This identity appears in the gradient of the binary cross-entropy loss (Chapter 9).

---

*Next: Chapter 6 — Gradients and Optimization. We extend derivatives to multiple dimensions and show how gradient descent trains models.*
