# Chapter 9: Statistics for ML

> *"A model that trains well is promising. A model you can trust is useful."*

---

Chapters 7 and 8 gave you the language of probability: how to describe uncertainty, how distributions work, and what cross-entropy and KL divergence mean. This chapter uses that language to answer a harder question: **when can you trust a result?**

Every time you train a model you are making implicit statistical choices. Choosing cross-entropy loss instead of MSE is a statistical argument. Choosing L2 regularization over L1 is a statistical argument. Reporting "95% accuracy" without a confidence interval is a statistical mistake. This chapter makes those arguments explicit and gives you the tools to evaluate model results with rigor.

We connect everything to what came before:

- **Chapters 2–4** (linear algebra) underpin the Normal Equation and ridge regression.
- **Chapters 5–6** (calculus) underpin gradient-based training and the bias-variance tradeoff.
- **Chapters 7–8** (probability) underpin MLE, regularization as a prior, and hypothesis testing.

---

## 9.1 Point Estimation — What Does It Mean to Fit a Model?

### Plain English First

When you train a model, you are computing a **point estimate**: a single number (or vector of numbers) that summarizes what the data is telling you. For example, after seeing data $\{x_1, x_2, \ldots, x_n\}$, you might produce:

- A mean $\hat{\mu}$ estimating the center of the data.
- A probability $\hat{p}$ estimating the fraction of positive examples.
- A weight vector $\hat{\mathbf{w}}$ estimating the parameters of a linear model.

The hat notation $\hat{\theta}$ universally means "estimate of $\theta$ computed from data." It is distinct from the true but unknown value $\theta^*$.

There are infinitely many ways to produce an estimate. You could guess at random. You could always output zero. You could average the data. What makes one estimator better than another? We need a principled criterion. The most important one in machine learning is **Maximum Likelihood Estimation**.

### Formal Notation

An **estimator** is a function $\hat{\theta}(\mathbf{x})$ that maps a dataset $\mathbf{x} = (x_1, \ldots, x_n)$ to a parameter value. It is itself a random variable — different random datasets produce different estimates.

Two properties we want an estimator to have:

**Unbiasedness:** $\mathbb{E}[\hat{\theta}] = \theta^*$. On average, the estimator is correct.

**Consistency:** As $n \to \infty$, $\hat{\theta} \to \theta^*$ in probability. More data means a better estimate.

**Trust this result:** The sample mean $\hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i$ is both unbiased and consistent for the population mean. These are provable from Chapter 7's linearity of expectation and the Law of Large Numbers.

**The engineer's perspective:** Point estimation is what happens every time you call `.fit()`. The optimizer finds the parameter vector $\hat{\theta}$ that minimizes the training loss. But which loss? That choice is not arbitrary — it is determined by a statistical argument called Maximum Likelihood.

---

## 9.2 Maximum Likelihood Estimation (MLE)

### Plain English First

Maximum Likelihood Estimation answers a fundamental question: given that I observed this data, which parameter value made the data most probable?

Imagine you flip a coin five times and see heads, heads, tails, heads, tails. The coin has some unknown bias $p$ (probability of heads). If $p = 0.5$, this sequence has some probability. If $p = 0.6$, it has a slightly different probability. If $p = 0.9$, it has a much lower probability (three tails would be very surprising). MLE asks: what value of $p$ maximizes the probability of the data you actually observed?

That value — the one that makes the data "most likely" — is the MLE estimate $\hat{p}$.

This is not just a heuristic. It is the mathematical justification for why neural networks minimize cross-entropy loss, why linear regression minimizes squared error, and why these are the right choices under specific probabilistic assumptions.

### Formal Notation

Let $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ be $n$ independent, identically distributed (i.i.d.) observations from a distribution with parameter $\theta$.

**Likelihood function:** The probability of observing the data as a function of $\theta$:

$$\mathcal{L}(\theta) = p(\mathbf{x} \mid \theta) = \prod_{i=1}^{n} p(x_i \mid \theta)$$

The product appears because the observations are independent (as in Chapter 7 — joint probability of independent events is the product of individual probabilities).

**MLE estimate:**

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \mathcal{L}(\theta)$$

### 9.2.1 The Log-Likelihood Trick: Products Become Sums

The likelihood is a product of $n$ terms, each in $[0, 1]$. For large $n$, this product is astronomically small and numerically unstable. The solution: maximize the **log-likelihood** instead.

$$\ell(\theta) = \log \mathcal{L}(\theta) = \log \prod_{i=1}^{n} p(x_i \mid \theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta)$$

**Here's why** this is valid: logarithm is a strictly increasing function, so $\arg\max \mathcal{L}(\theta) = \arg\max \log \mathcal{L}(\theta)$. The maximizer does not change.

**Here's why** this matters: the product of 10,000 probabilities underflows to zero in floating point. The sum of 10,000 log-probabilities stays numerically well-behaved. This is why training loops accumulate log-probabilities (or cross-entropy losses), not raw probabilities.

**The connection to Chapter 8:** For a one-hot true distribution $P$ and a model distribution $Q$, the cross-entropy is:

$$H(P, Q) = -\sum_k P(k) \log Q(k) = -\log Q(\text{true class})$$

This is exactly the negative log-likelihood of the true class under the model's distribution. Training by minimizing cross-entropy is equivalent to maximizing the log-likelihood. This is not a coincidence — it is the mathematical foundation of why cross-entropy loss is correct.

### 9.2.2 Deriving MLE for the Bernoulli Parameter

**Setup:** We have $n$ independent coin flips, each following $\text{Bernoulli}(p)$. Observed outcomes $x_i \in \{0, 1\}$ with $k = \sum_{i=1}^{n} x_i$ heads total. Find $\hat{p}_{\text{MLE}}$.

**Step 1 — Write the likelihood:**

Using the Bernoulli PMF $P(X = x_i) = p^{x_i}(1-p)^{1-x_i}$:

$$\mathcal{L}(p) = \prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i} = p^k (1-p)^{n-k}$$

**Step 2 — Take the log-likelihood:**

$$\ell(p) = k \log p + (n - k) \log(1 - p)$$

**Step 3 — Differentiate and set to zero (Chapter 5 — setting derivative to zero finds the maximum):**

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

**Step 4 — Solve:**

$$\frac{k}{p} = \frac{n-k}{1-p}$$

$$k(1-p) = (n-k)p$$

$$k - kp = np - kp$$

$$k = np$$

$$\hat{p}_{\text{MLE}} = \frac{k}{n}$$

**Interpretation:** The MLE for the Bernoulli parameter is simply the fraction of observed successes. This is the intuitive answer — and now we have derived it from first principles.

**Step 5 — Verify it is a maximum, not a minimum:** Check the second derivative.

$$\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2}$$

For $k > 0$ and $k < n$, both terms are negative, so $\frac{d^2\ell}{dp^2} < 0$ at the critical point — confirming a maximum. $\checkmark$

### Worked Example 9.2.1 — MLE for Bernoulli

Dataset: $\{1, 0, 1, 1, 0\}$ — five coin flips.

$n = 5$, $k = 3$ (three heads).

$$\hat{p}_{\text{MLE}} = \frac{3}{5} = 0.6$$

**Verify by comparing log-likelihoods at nearby values:**

$$\ell(0.6) = 3\log(0.6) + 2\log(0.4) = 3(-0.5108) + 2(-0.9163) = -1.5325 - 1.8326 = -3.3651$$

$$\ell(0.5) = 3\log(0.5) + 2\log(0.5) = 5 \times (-0.6931) = -3.4657$$

$$\ell(0.7) = 3\log(0.7) + 2\log(0.3) = 3(-0.3567) + 2(-1.2040) = -1.0700 - 2.4079 = -3.4779$$

Indeed $\ell(0.6) = -3.3651 > \ell(0.5) = -3.4657$ and $\ell(0.6) > \ell(0.7) = -3.4779$. The log-likelihood is highest at $\hat{p} = 0.6$. $\checkmark$

### 9.2.3 Deriving MLE for the Gaussian Mean

**Setup:** We have $n$ i.i.d. observations from $\mathcal{N}(\mu, \sigma^2)$ where $\sigma^2$ is known. Find $\hat{\mu}_{\text{MLE}}$.

**Step 1 — Write the log-likelihood:**

Using the Gaussian PDF from Chapter 8:

$$\ell(\mu) = \sum_{i=1}^{n} \log \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

$$= \sum_{i=1}^{n} \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2}\right]$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Step 2 — Maximize over $\mu$:** The first term is a constant in $\mu$. Maximizing $\ell(\mu)$ is equivalent to minimizing $\sum_{i=1}^{n}(x_i - \mu)^2$ — the **sum of squared errors**.

**Here's why** this is important: MLE under a Gaussian noise assumption produces MSE (mean squared error) as the training loss. When you minimize squared error in linear regression, you are implicitly assuming Gaussian noise. The probabilistic model is baked into the loss function.

**Step 3 — Differentiate and set to zero:**

$$\frac{d}{d\mu}\left[-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2\right] = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

$$\sum_{i=1}^{n} x_i - n\mu = 0$$

$$\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

**Interpretation:** The MLE for the Gaussian mean is the sample mean. Again, the intuitive answer is the right one — and the derivation shows it follows directly from the Gaussian assumption plus the log-likelihood trick.

**A note on MLE for variance:** The MLE for $\sigma^2$ turns out to be $\frac{1}{n}\sum(x_i - \bar{x})^2$, which is the **biased** estimator. The unbiased estimator divides by $n-1$ (Bessel's correction): $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$. This is why statistics libraries report $n-1$ in the denominator — they use the unbiased version. MLE is optimal in many ways, but "unbiased" is not always one of them.

### Worked Example 9.2.2 — MLE for Gaussian Mean

Dataset: $\{2, 4, 6\}$ (three observations, known $\sigma^2 = 4$, so $\sigma = 2$).

$$\hat{\mu}_{\text{MLE}} = \frac{2 + 4 + 6}{3} = \frac{12}{3} = 4$$

**Verify the derivative is zero at $\mu = 4$:**

$$\sum_{i=1}^{3}(x_i - 4) = (2 - 4) + (4 - 4) + (6 - 4) = -2 + 0 + 2 = 0 \checkmark$$

**Compare log-likelihoods:**

$$\ell(4) = -\frac{3}{2}\log(8\pi) - \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{8} = -\frac{3}{2}\log(8\pi) - \frac{4+0+4}{8} = -\frac{3}{2}\log(8\pi) - 1$$

$$\ell(5) = -\frac{3}{2}\log(8\pi) - \frac{(2-5)^2 + (4-5)^2 + (6-5)^2}{8} = -\frac{3}{2}\log(8\pi) - \frac{9+1+1}{8} = -\frac{3}{2}\log(8\pi) - 1.375$$

Since $-1 > -1.375$, we confirm $\ell(4) > \ell(5)$: the MLE at $\mu = 4$ has a higher log-likelihood than the competitor at $\mu = 5$. $\checkmark$

### Summary: MLE Loss Functions

| Noise model (on $y$) | Resulting MLE loss |
|---|---|
| $y \sim \mathcal{N}(f(x), \sigma^2)$ | Mean Squared Error (MSE) |
| $y \sim \text{Bernoulli}(f(x))$ | Binary cross-entropy |
| $y \sim \text{Categorical}(\text{softmax}(f(x)))$ | Categorical cross-entropy |

These are not arbitrary choices. Each loss function is the negative log-likelihood of the corresponding distributional assumption. Choosing a loss function is choosing a noise model.

### Engineer's Angle

```python
import math

# MLE for Bernoulli
def bernoulli_log_likelihood(data, p):
    """Log-likelihood of data under Bernoulli(p)."""
    k = sum(data)
    n = len(data)
    return k * math.log(p) + (n - k) * math.log(1 - p)

def bernoulli_mle(data):
    """MLE estimate for Bernoulli parameter p."""
    return sum(data) / len(data)

data = [1, 0, 1, 1, 0]
p_hat = bernoulli_mle(data)
print(f"Bernoulli MLE: p_hat = {p_hat}")          # 0.6

# Verify this is the maximum by scanning nearby values
for p in [0.4, 0.5, 0.6, 0.7, 0.8]:
    ll = bernoulli_log_likelihood(data, p)
    marker = " <-- MLE" if p == p_hat else ""
    print(f"  log-lik(p={p}) = {ll:.6f}{marker}")

# MLE for Gaussian mean (known variance)
def gaussian_log_likelihood(data, mu, sigma_sq):
    """Log-likelihood of data under N(mu, sigma_sq)."""
    n = len(data)
    sigma = math.sqrt(sigma_sq)
    const = -n / 2 * math.log(2 * math.pi * sigma_sq)
    sq_term = -sum((x - mu) ** 2 for x in data) / (2 * sigma_sq)
    return const + sq_term

def gaussian_mle_mean(data):
    """MLE estimate for Gaussian mean (sample mean)."""
    return sum(data) / len(data)

def gaussian_mle_var_biased(data, mu):
    """Biased MLE estimate for Gaussian variance (divides by n)."""
    n = len(data)
    return sum((x - mu) ** 2 for x in data) / n

def gaussian_var_unbiased(data, mu):
    """Unbiased Bessel-corrected variance estimate (divides by n-1)."""
    n = len(data)
    return sum((x - mu) ** 2 for x in data) / (n - 1)

data_g = [2.0, 4.0, 6.0]
mu_hat = gaussian_mle_mean(data_g)
var_biased = gaussian_mle_var_biased(data_g, mu_hat)
var_unbiased = gaussian_var_unbiased(data_g, mu_hat)
print(f"\nGaussian MLE: mu_hat = {mu_hat}")                          # 4.0
print(f"Biased variance (MLE, n in denominator) = {var_biased:.4f}")   # 2.6667
print(f"Unbiased variance (Bessel, n-1 in denominator) = {var_unbiased:.4f}")  # 4.0

# Verify: likelihood is maximized at mu_hat
sigma_sq = 4.0  # known variance
for mu in [3.0, 4.0, 5.0]:
    ll = gaussian_log_likelihood(data_g, mu, sigma_sq)
    print(f"  log-lik(mu={mu}) = {ll:.6f}" + (" <-- MLE" if mu == mu_hat else ""))
# mu=4.0 has highest log-likelihood
```

---

## 9.3 The Normal Equation — Closed-Form Linear Regression

### Plain English First

In Chapter 6, we minimized the squared-error loss by gradient descent: update parameters repeatedly, one small step at a time. But for linear regression specifically, there is a better option: **solve exactly in one shot**.

Calculus tells us that at a minimum, all partial derivatives equal zero. For linear regression, that "all derivatives equal zero" condition is a system of linear equations — and we know how to solve those exactly using the matrix tools from Chapter 3. The result is the **Normal Equation**: a single matrix formula that gives the optimal weights directly.

### Formal Notation

Given a dataset with $n$ examples and $d$ features, define:

- Design matrix $X \in \mathbb{R}^{n \times d}$: row $i$ is the feature vector for example $i$.
- Target vector $\mathbf{y} \in \mathbb{R}^n$: the true outputs.
- Weight vector $\mathbf{w} \in \mathbb{R}^d$: parameters to find.

The **MSE loss** is:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|_2^2 = \frac{1}{n}(X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y})$$

**Here's why** the Normal Equation is $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$:

**Step 1 — Expand the loss (dropping the $\frac{1}{n}$ constant which does not affect the minimizer):**

$$\mathcal{L}(\mathbf{w}) = (X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y})$$
$$= \mathbf{w}^TX^TX\mathbf{w} - \mathbf{w}^TX^T\mathbf{y} - \mathbf{y}^TX\mathbf{w} + \mathbf{y}^T\mathbf{y}$$

Since $\mathbf{w}^TX^T\mathbf{y}$ is a scalar, it equals its own transpose: $\mathbf{y}^TX\mathbf{w}$. So:

$$\mathcal{L}(\mathbf{w}) = \mathbf{w}^TX^TX\mathbf{w} - 2\mathbf{y}^TX\mathbf{w} + \mathbf{y}^T\mathbf{y}$$

**Step 2 — Differentiate with respect to $\mathbf{w}$ (Chapter 6 — gradient of a quadratic form):**

$$\nabla_\mathbf{w} \mathcal{L} = 2X^TX\mathbf{w} - 2X^T\mathbf{y}$$

**Trust this result:** The gradient of $\mathbf{w}^TA\mathbf{w}$ is $2A\mathbf{w}$ when $A$ is symmetric (which $X^TX$ is), and the gradient of $\mathbf{b}^T\mathbf{w}$ is $\mathbf{b}$. Both follow from the matrix calculus rules in Chapter 6.

**Step 3 — Set gradient to zero:**

$$2X^TX\mathbf{w} - 2X^T\mathbf{y} = \mathbf{0}$$

$$X^TX\mathbf{w} = X^T\mathbf{y}$$

These are the **Normal Equations** — a system of $d$ linear equations in $d$ unknowns.

**Step 4 — Solve (if $X^TX$ is invertible):**

$$\boxed{\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}}$$

This is the Normal Equation.

### Worked Example 9.3.1 — Fitting a Line

Fit $\hat{y} = w_1 x + w_0$ to three data points: $(1, 2)$, $(2, 3)$, $(3, 5)$.

**Step 1 — Build the design matrix.** Include a bias column of ones:

$$X = \begin{bmatrix} 1 & 1 \\ 2 & 1 \\ 3 & 1 \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}$$

**Step 2 — Compute $X^TX$:**

$$X^TX = \begin{bmatrix} 1 & 2 & 3 \\ 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 2 & 1 \\ 3 & 1 \end{bmatrix}$$

$$= \begin{bmatrix} 1^2 + 2^2 + 3^2 & 1 + 2 + 3 \\ 1 + 2 + 3 & 1 + 1 + 1 \end{bmatrix} = \begin{bmatrix} 14 & 6 \\ 6 & 3 \end{bmatrix}$$

**Step 3 — Compute $X^T\mathbf{y}$:**

$$X^T\mathbf{y} = \begin{bmatrix} 1 & 2 & 3 \\ 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} = \begin{bmatrix} 1(2) + 2(3) + 3(5) \\ 1(2) + 1(3) + 1(5) \end{bmatrix} = \begin{bmatrix} 23 \\ 10 \end{bmatrix}$$

**Step 4 — Invert $X^TX$.** For a $2 \times 2$ matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is $\frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$:

$$\det(X^TX) = 14 \times 3 - 6 \times 6 = 42 - 36 = 6$$

$$(X^TX)^{-1} = \frac{1}{6}\begin{bmatrix} 3 & -6 \\ -6 & 14 \end{bmatrix} = \begin{bmatrix} 1/2 & -1 \\ -1 & 7/3 \end{bmatrix}$$

**Step 5 — Multiply to get $\mathbf{w}$:**

$$\mathbf{w} = (X^TX)^{-1} X^T\mathbf{y} = \begin{bmatrix} 1/2 & -1 \\ -1 & 7/3 \end{bmatrix} \begin{bmatrix} 23 \\ 10 \end{bmatrix}$$

$$w_1 = \frac{1}{2}(23) + (-1)(10) = 11.5 - 10 = 1.5$$

$$w_0 = (-1)(23) + \frac{7}{3}(10) = -23 + \frac{70}{3} = \frac{-69 + 70}{3} = \frac{1}{3} \approx 0.333$$

**Fitted model:** $\hat{y} = 1.5x + \tfrac{1}{3}$

**Step 6 — Verify predictions and residuals:**

| $x$ | $y$ | $\hat{y} = 1.5x + 1/3$ | residual |
|-----|-----|-------------------------|----------|
| 1 | 2 | $1.5 + 0.333 = 1.833$ | $+0.167$ |
| 2 | 3 | $3.0 + 0.333 = 3.333$ | $-0.333$ |
| 3 | 5 | $4.5 + 0.333 = 4.833$ | $+0.167$ |

**Verify the gradient is zero at these weights:**

The gradient condition is $X^TX\mathbf{w} = X^T\mathbf{y}$:

$$\begin{bmatrix} 14 & 6 \\ 6 & 3 \end{bmatrix} \begin{bmatrix} 1.5 \\ 1/3 \end{bmatrix} = \begin{bmatrix} 14(1.5) + 6(1/3) \\ 6(1.5) + 3(1/3) \end{bmatrix} = \begin{bmatrix} 21 + 2 \\ 9 + 1 \end{bmatrix} = \begin{bmatrix} 23 \\ 10 \end{bmatrix} = X^T\mathbf{y} \checkmark$$

### 9.3.1 Normal Equation vs. Gradient Descent

| | Normal Equation | Gradient Descent |
|---|---|---|
| **Cost** | $O(nd^2 + d^3)$ (matrix multiply + inversion) | $O(ndk)$ ($k$ iterations) |
| **When to prefer** | Small $d$ (up to ~10,000 features) | Large $d$, sparse data, online learning |
| **Exact?** | Yes — single-step exact solution | No — converges asymptotically |
| **Hyperparameters** | None | Learning rate, batch size, iterations |
| **Singular $X^TX$?** | Fails (use pseudoinverse instead) | Still works — converges to a solution |
| **Streaming data?** | No — needs full dataset | Yes — stochastic/mini-batch variants |

**The $d^3$ bottleneck:** The matrix inversion step costs $O(d^3)$ operations. For $d = 10{,}000$ features, that is $10^{12}$ operations — intractable. For $d = 100$ features, it is $10^6$ — fast. Gradient descent scales much better with $d$, which is why deep learning uses gradient descent despite having exact alternatives.

**Trust this result:** When $X^TX$ is singular (which happens when features are linearly dependent — collinear), $(X^TX)^{-1}$ does not exist. The fix is to use the Moore-Penrose pseudoinverse $X^+$ or to add regularization — which leads directly to Section 9.5.

### Engineer's Angle

```python
import math

# ── Matrix utilities (stdlib only) ────────────────────────────────────────────

def mat_transpose(A):
    """Transpose a matrix represented as list of row lists."""
    rows, cols = len(A), len(A[0])
    return [[A[r][c] for r in range(rows)] for c in range(cols)]

def mat_mul(A, B):
    """Matrix multiply A @ B."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "Incompatible shapes"
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def mat_vec_mul(A, v):
    """Matrix-vector product A @ v."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def inv2x2(A):
    """Invert a 2x2 matrix. Raises if singular."""
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular")
    return [[d / det, -b / det], [-c / det, a / det]]

# ── Normal Equation ────────────────────────────────────────────────────────────

def normal_equation_2d(data):
    """
    Fit y = w1*x + w0 using the Normal Equation.
    data: list of (x, y) pairs.
    Returns (w1, w0).
    """
    # Build design matrix X (x in col 0, bias=1 in col 1)
    X = [[xi, 1.0] for xi, yi in data]
    y = [yi for xi, yi in data]

    Xt    = mat_transpose(X)
    XtX   = mat_mul(Xt, X)
    Xty   = mat_vec_mul(Xt, y)
    XtX_inv = inv2x2(XtX)
    w     = mat_vec_mul(XtX_inv, Xty)
    return w[0], w[1]  # w1 (slope), w0 (intercept)

# ── Example ────────────────────────────────────────────────────────────────────

data = [(1, 2), (2, 3), (3, 5)]
w1, w0 = normal_equation_2d(data)
print(f"w1 (slope)     = {w1:.6f}")    # 1.500000
print(f"w0 (intercept) = {w0:.6f}")    # 0.333333

print("\nPredictions vs. actuals:")
for x, y in data:
    y_hat = w1 * x + w0
    residual = y - y_hat
    print(f"  x={x}, y={y}, y_hat={y_hat:.4f}, residual={residual:.4f}")

# Compute MSE
mse = sum((y - (w1 * x + w0)) ** 2 for x, y in data) / len(data)
print(f"\nMSE = {mse:.6f}")
```

---

## 9.4 Bias-Variance Tradeoff

### Plain English First

Every prediction error comes from three sources. The **dartboard analogy** makes them viscerally clear.

Imagine you throw many darts at a target. Your "model" is your throwing arm for that session. The true bullseye is the target you're trying to hit.

- **High bias, low variance:** All your darts cluster tightly — but they're all far from the bullseye. You're consistently wrong. A model that always predicts the training average has this problem: it ignores structure in the data (underfitting).
- **Low bias, high variance:** Your darts scatter widely around the bullseye — sometimes close, sometimes far. Average position is right, but individual throws are unreliable. A model that memorizes training examples has this problem: it fits training noise and fails on new data (overfitting).
- **Low bias, low variance:** Darts cluster tightly around the bullseye. This is the goal. Achieve it through the right model complexity and sufficient data.

The third source of error — **noise** — is the irreducible scatter in the data itself. No matter how good your model, if the data is inherently noisy, you cannot eliminate this error.

### Formal Notation

Consider a fixed test point $x_0$ with true output $y_0 = f(x_0) + \epsilon$, where $f$ is the true (unknown) function and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is irreducible noise.

Your model $\hat{f}$ is trained on a random training set, so $\hat{f}(x_0)$ is a random variable. Let $\bar{f} = \mathbb{E}[\hat{f}(x_0)]$ be the expected prediction over all possible training sets.

Define:

$$\text{Bias}(\hat{f}) = \bar{f} - f(x_0) = \mathbb{E}[\hat{f}(x_0)] - f(x_0)$$

$$\text{Variance}(\hat{f}) = \mathbb{E}\!\left[(\hat{f}(x_0) - \bar{f})^2\right]$$

$$\text{MSE} = \mathbb{E}\!\left[(\hat{f}(x_0) - y_0)^2\right]$$

**Theorem:**

$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

**Here's why** the decomposition holds:

$$\text{MSE} = \mathbb{E}\!\left[(\hat{f}(x_0) - y_0)^2\right] = \mathbb{E}\!\left[(\hat{f}(x_0) - f(x_0) - \epsilon)^2\right]$$

Add and subtract $\bar{f}$ inside:

$$= \mathbb{E}\!\left[(\hat{f}(x_0) - \bar{f} + \bar{f} - f(x_0) - \epsilon)^2\right]$$

Let $U = \hat{f}(x_0) - \bar{f}$ (zero-mean random variable: $\mathbb{E}[U] = 0$), $B = \bar{f} - f(x_0)$ (the bias, a constant), and keep $\epsilon$ (also zero-mean, independent of training data). Then:

$$\text{MSE} = \mathbb{E}[(U + B - \epsilon)^2]$$
$$= \mathbb{E}[U^2] + B^2 + \mathbb{E}[\epsilon^2] + 2B\,\mathbb{E}[U] - 2B\,\mathbb{E}[\epsilon] - 2\,\mathbb{E}[U\epsilon]$$

The cross-terms vanish: $\mathbb{E}[U] = 0$, $\mathbb{E}[\epsilon] = 0$, and $U$ and $\epsilon$ are independent so $\mathbb{E}[U\epsilon] = 0$. Therefore:

$$\boxed{\text{MSE} = \underbrace{\mathbb{E}[U^2]}_{\text{Variance}} + \underbrace{B^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[\epsilon^2]}_{\sigma^2} = \text{Variance} + \text{Bias}^2 + \sigma^2} \quad \square$$

### Worked Example 9.4.1 — Two Models at a Test Point

True function: $f(x) = 2x + 1$. Test point $x_0 = 3$, so $f(3) = 7$. Noise $\epsilon \sim \mathcal{N}(0, 0.25)$ so $\sigma^2 = 0.25$.

**Model A — High-bias constant predictor:** Always predicts $\hat{f}_A(x) = 3$ (the training mean, ignoring $x$). This is deterministic regardless of the training set.

$$\text{Bias}_A = \mathbb{E}[\hat{f}_A(3)] - f(3) = 3 - 7 = -4$$

$$\text{Bias}_A^2 = 16$$

$$\text{Variance}_A = \mathbb{E}[(\hat{f}_A(3) - 3)^2] = 0 \quad \text{(constant predictor has zero variance)}$$

$$\text{MSE}_A = 16 + 0 + 0.25 = 16.25$$

**Model B — Low-bias flexible predictor:** Interpolates training data, producing predictions $\hat{f}_B(3) \in \{6, 8\}$ with equal probability across training sets (noisy, but centered on the truth).

$$\mathbb{E}[\hat{f}_B(3)] = \frac{6 + 8}{2} = 7$$

$$\text{Bias}_B = 7 - 7 = 0$$

$$\text{Bias}_B^2 = 0$$

$$\text{Variance}_B = \frac{(6-7)^2 + (8-7)^2}{2} = \frac{1 + 1}{2} = 1$$

$$\text{MSE}_B = 0 + 1 + 0.25 = 1.25$$

**Conclusion:** Model B has MSE of 1.25 vs Model A's 16.25. Model A's large bias overwhelms its advantage in variance. We prefer Model B.

| | Bias$^2$ | Variance | Noise | MSE |
|---|---|---|---|---|
| Model A (constant predictor) | 16 | 0 | 0.25 | **16.25** |
| Model B (flexible predictor) | 0 | 1 | 0.25 | **1.25** |

### 9.4.1 The Tradeoff

Bias and variance are in tension:

- **Increasing model complexity** (more parameters, deeper network, higher polynomial degree) → lower bias, higher variance.
- **Decreasing model complexity** → higher bias, lower variance.
- **Adding more training data** → lower variance without changing the inherent bias.

The bias-variance tradeoff explains why:
- A degree-1 polynomial underfit (high bias) to data with quadratic structure.
- A degree-30 polynomial overfit (high variance) the same data.
- The optimal degree lies between, and can be found by cross-validation (Section 9.7).

### Engineer's Angle

```python
import math
import statistics

# Simulate bias-variance tradeoff for a constant vs mean predictor
# True f(x) = 2x + 1, test point x0 = 3, noise sigma^2 = 0.25

def true_f(x):
    return 2 * x + 1

x0 = 3
f_true = true_f(x0)   # 7
sigma_sq = 0.25

# ── Model A: always predicts 3 (high bias, zero variance) ──────────────────
bias_A    = 3 - f_true     # -4
bias_sq_A = bias_A ** 2    # 16
var_A     = 0.0
mse_A     = bias_sq_A + var_A + sigma_sq
print(f"Model A (constant=3):")
print(f"  Bias^2={bias_sq_A}, Variance={var_A}, Noise={sigma_sq}, MSE={mse_A}")

# ── Model B: noisy but centered (low bias, some variance) ──────────────────
predictions_B = [6.0, 8.0]
mean_B    = statistics.mean(predictions_B)   # 7.0
bias_B    = mean_B - f_true                  # 0.0
bias_sq_B = bias_B ** 2                      # 0.0
var_B     = statistics.mean([(p - mean_B)**2 for p in predictions_B])  # 1.0
mse_B     = bias_sq_B + var_B + sigma_sq
print(f"\nModel B (flexible, E[f_hat]=7):")
print(f"  Bias^2={bias_sq_B}, Variance={var_B}, Noise={sigma_sq}, MSE={mse_B}")

# ── Verify decomposition numerically ───────────────────────────────────────
# Simulate noise realizations
import random
random.seed(42)
n_sim = 100_000

def simulate_mse(predictions, sigma):
    """Estimate MSE by simulation."""
    errors = []
    for _ in range(n_sim):
        pred = random.choice(predictions)
        noise = random.gauss(0, sigma)
        y = f_true + noise
        errors.append((pred - y) ** 2)
    return statistics.mean(errors)

sigma = math.sqrt(sigma_sq)
sim_mse_A = simulate_mse([3.0], sigma)
sim_mse_B = simulate_mse([6.0, 8.0], sigma)
print(f"\nSimulated MSE (n={n_sim} trials):")
print(f"  Model A: {sim_mse_A:.4f} (analytical: {mse_A})")
print(f"  Model B: {sim_mse_B:.4f} (analytical: {mse_B})")
# Should be close to analytical values
```

---

## 9.5 Overfitting and Regularization

### Plain English First

Overfitting is the variance problem made concrete: the model learns the training data — including its noise — so well that it performs poorly on new data. Regularization is the cure: add a penalty for complexity directly into the loss function, discouraging the optimizer from finding weights that are large in magnitude.

There are two canonical forms of regularization, and they have very different behaviors.

### Formal Notation

The regularized loss adds a penalty term to the base loss $\mathcal{L}_0(\mathbf{w})$ (e.g., MSE or cross-entropy):

$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \mathcal{L}_0(\mathbf{w}) + \lambda\|\mathbf{w}\|_2^2 = \mathcal{L}_0(\mathbf{w}) + \lambda \sum_{j=1}^{d} w_j^2$$

$$\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \mathcal{L}_0(\mathbf{w}) + \lambda\|\mathbf{w}\|_1 = \mathcal{L}_0(\mathbf{w}) + \lambda \sum_{j=1}^{d} |w_j|$$

The **hyperparameter** $\lambda \geq 0$ controls the strength of regularization. $\lambda = 0$ gives the unregularized model; large $\lambda$ heavily penalizes large weights.

### 9.5.1 L2 Regularization (Ridge)

**What it does:** Shrinks all weights toward zero proportionally. Large weights are penalized quadratically, so all weights survive but in reduced magnitude.

**The Ridge Normal Equation:** For linear regression with L2 regularization, set the gradient of the penalized loss to zero:

$$\nabla_{\mathbf{w}}\bigl(\mathcal{L}_0 + \lambda\|\mathbf{w}\|^2\bigr) = 2X^TX\mathbf{w} - 2X^T\mathbf{y} + 2\lambda\mathbf{w} = \mathbf{0}$$

$$\Longrightarrow\quad (X^TX + \lambda I)\mathbf{w} = X^T\mathbf{y} \quad\Longrightarrow\quad \mathbf{w}_{\text{Ridge}} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$$

**Here's why** adding $\lambda I$ helps when $X^TX$ is nearly singular: the eigenvalues of $X^TX$ are non-negative (it is positive semi-definite), so some may be near zero, making the inverse unstable. Adding $\lambda > 0$ shifts all eigenvalues up by $\lambda$, making every eigenvalue at least $\lambda > 0$, so the matrix is now invertible. This is a stability improvement even beyond regularization.

### Worked Example 9.5.1 — Ridge vs OLS

Using the same data $(1, 2), (2, 3), (3, 5)$ from Section 9.3, apply Ridge with $\lambda = 1$.

**OLS result (from Section 9.3):** $\mathbf{w}_{\text{OLS}} = \begin{bmatrix} 1.5 \\ 1/3 \end{bmatrix}$

**Ridge computation:**

$$X^TX + \lambda I = \begin{bmatrix} 14 & 6 \\ 6 & 3 \end{bmatrix} + \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 15 & 6 \\ 6 & 4 \end{bmatrix}$$

$$\det\begin{bmatrix} 15 & 6 \\ 6 & 4 \end{bmatrix} = 15 \times 4 - 6 \times 6 = 60 - 36 = 24$$

$$\left(X^TX + I\right)^{-1} = \frac{1}{24}\begin{bmatrix} 4 & -6 \\ -6 & 15 \end{bmatrix} = \begin{bmatrix} 1/6 & -1/4 \\ -1/4 & 5/8 \end{bmatrix}$$

$$\mathbf{w}_{\text{Ridge}} = \begin{bmatrix} 1/6 & -1/4 \\ -1/4 & 5/8 \end{bmatrix} \begin{bmatrix} 23 \\ 10 \end{bmatrix}$$

$$w_1 = \frac{23}{6} - \frac{10}{4} = 3.8333 - 2.5 = 1.3333$$

$$w_0 = -\frac{23}{4} + \frac{50}{8} = -5.75 + 6.25 = 0.5$$

**Ridge result:** $\mathbf{w}_{\text{Ridge}} = \begin{bmatrix} 1.333 \\ 0.500 \end{bmatrix}$

**Comparison:**

| | $w_1$ (slope) | $w_0$ (intercept) |
|---|---|---|
| OLS (no regularization) | 1.500 | 0.333 |
| Ridge ($\lambda = 1$) | 1.333 | 0.500 |

Ridge shrinks the slope from 1.5 toward 0 (reduced from 1.500 to 1.333). The intercept shifts because the weights trade off against each other. Both weights moved toward smaller magnitude — the defining behavior of L2 regularization.

### 9.5.2 L1 Regularization (Lasso) and Why It Causes Sparsity

L1 regularization does something that L2 does not: it drives many weights to **exactly zero**, producing a **sparse** solution where only a subset of features matter.

**Why does L1 cause sparsity?**

There are two complementary ways to see this. Use whichever clicks for you.

**Explanation 1 — The geometric argument:**

Picture the weight space in 2D with weights $w_1$ and $w_2$. The unregularized loss function has elliptical level curves. The regularization term is a constraint: $|w_1| + |w_2| \leq t$ for some budget $t$. The L1 constraint defines a **diamond** (tilted square) with corners on the axes at $(\pm t, 0)$ and $(0, \pm t)$.

When you minimize the regularized loss, you are looking for the point where a level curve of the loss first touches the constraint region. The diamond has **corners** — pointed vertices on the axes. The loss's level curves are smooth ellipses. An ellipse expanding outward from its minimum will generically first touch the diamond at a **corner** (a sparse point with one coordinate = 0) rather than along a smooth edge. L2's constraint is a circle (no corners), so the first contact point is typically not on an axis.

**Explanation 2 — The subgradient argument:**

At a point where $w_j \neq 0$, the L1 penalty gradient is $\lambda \cdot \text{sign}(w_j)$. This pulls $w_j$ toward zero at a fixed rate $\lambda$ regardless of its magnitude.

At $w_j = 0$, the derivative of $|w_j|$ does not exist — but the **subgradient** is any value in $[-\lambda, \lambda]$. This means that if the loss's gradient $\partial \mathcal{L}_0 / \partial w_j$ at $w_j = 0$ is small (in $[-\lambda, \lambda]$), the subgradient condition is satisfied with $w_j = 0$ — meaning there is **no force to move $w_j$ away from zero**. The weight stays pinned at exactly zero. This "dead zone" of width $2\lambda$ around zero is what creates sparsity.

L2 does not have this property: the L2 gradient is $2\lambda w_j$, which shrinks to zero as $w_j \to 0$, meaning the penalty asymptotically vanishes and never actually zeros out the weight.

### 9.5.3 Bayesian Interpretation of Regularization

Regularization is not an ad hoc trick — it is Bayesian inference in disguise. Recall Bayes' theorem from Chapter 7:

$$p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \cdot p(\mathbf{w})$$

Taking the log:

$$\log p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \log p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) + \log p(\mathbf{w}) + \text{const}$$

The **MAP (Maximum A Posteriori)** estimate maximizes this, which is equivalent to minimizing:

$$-\log p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) - \log p(\mathbf{w})$$

Now choose a prior on $\mathbf{w}$:

**Gaussian prior** $p(\mathbf{w}) \propto \exp\!\left(-\frac{\|\mathbf{w}\|_2^2}{2\tau^2}\right)$:

$$-\log p(\mathbf{w}) = \frac{\|\mathbf{w}\|_2^2}{2\tau^2} + \text{const} \propto \|\mathbf{w}\|_2^2$$

This is exactly the L2 penalty with $\lambda = \frac{1}{2\tau^2}$.

**Laplace prior** $p(\mathbf{w}) \propto \exp\!\left(-\frac{\|\mathbf{w}\|_1}{b}\right)$:

$$-\log p(\mathbf{w}) = \frac{\|\mathbf{w}\|_1}{b} + \text{const} \propto \|\mathbf{w}\|_1$$

This is exactly the L1 penalty with $\lambda = \frac{1}{b}$.

**Interpretation:** When you add L2 regularization, you are saying "I believe the weights are small — my prior is Gaussian centered at zero." When you add L1 regularization, you are saying "I believe most weights are zero — my prior is Laplace (heavy-tailed, peaked sharply at zero), encouraging sparsity."

Regularization strength $\lambda$ encodes how strongly you hold this belief. Small $\lambda$ = weak prior (data dominates). Large $\lambda$ = strong prior (regularizer dominates, model barely fits data).

### Engineer's Angle

```python
import math

# ── Ridge regression via closed form (2D, reusing inv2x2) ─────────────────

def mat_add(A, B):
    """Add two matrices element-wise."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def mat_transpose(A):
    rows, cols = len(A), len(A[0])
    return [[A[r][c] for r in range(rows)] for c in range(cols)]

def mat_mul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def mat_vec_mul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def inv2x2(A):
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix")
    return [[d / det, -b / det], [-c / det, a / det]]

def ridge_regression_2d(data, lam):
    """Ridge regression: w = (X^T X + lam * I)^{-1} X^T y."""
    X   = [[xi, 1.0] for xi, yi in data]
    y   = [yi for xi, yi in data]
    Xt  = mat_transpose(X)
    XtX = mat_mul(Xt, X)
    reg = mat_add(XtX, [[lam * e for e in row] for row in identity(2)])
    Xty = mat_vec_mul(Xt, y)
    w   = mat_vec_mul(inv2x2(reg), Xty)
    return w[0], w[1]  # slope, intercept

data = [(1, 2), (2, 3), (3, 5)]

print("Lambda | w1 (slope) | w0 (intercept)")
print("-------|------------|---------------")
for lam in [0.0, 0.5, 1.0, 5.0, 10.0]:
    w1, w0 = ridge_regression_2d(data, lam)
    print(f"  {lam:4.1f} |   {w1:.6f} |    {w0:.6f}")
# As lambda increases, weights shrink toward 0

# ── Log prior comparison: Gaussian vs Laplace ──────────────────────────────

def gaussian_log_prior(w, tau_sq):
    """log p(w) for Gaussian prior N(0, tau_sq)."""
    return -sum(wi**2 / (2 * tau_sq) for wi in w)

def laplace_log_prior(w, b):
    """log p(w) for Laplace prior with scale b."""
    return -sum(abs(wi) / b for wi in w)

w_sparse = [0.0, 1.5, 0.0, 0.2, 0.0]   # mostly zeros
w_dense  = [0.3, 0.3, 0.3, 0.3, 0.3]   # all small, same L2 norm

tau_sq, b = 1.0, 1.0

print("\nComparing priors:")
print(f"Sparse weights {w_sparse}:")
print(f"  Gaussian log-prior: {gaussian_log_prior(w_sparse, tau_sq):.4f}")
print(f"  Laplace  log-prior: {laplace_log_prior(w_sparse, b):.4f}")

print(f"\nDense weights  {w_dense}:")
print(f"  Gaussian log-prior: {gaussian_log_prior(w_dense, tau_sq):.4f}")
print(f"  Laplace  log-prior: {laplace_log_prior(w_dense, b):.4f}")
# Laplace strongly prefers sparse over dense; Gaussian is more neutral
```

---

## 9.6 Train / Validation / Test Split

### Plain English First

You have a dataset. You want to report how good your model is. If you test it on the same data you trained it on, the number you get is meaningless — the model has already seen those examples and may have memorized them. You need data the model has **never seen**.

But there is a second, subtler trap: every time you tune a hyperparameter (learning rate, regularization strength, architecture choices) based on test-set performance, you are indirectly fitting the test set. After enough tuning decisions, your model is implicitly optimized for the test set, making the test error an overly optimistic estimate.

The solution is three-way data splitting.

### The Three Splits

**Training set (typically 60–80% of data):**

Used to fit the model parameters. The optimizer sees only these examples. The model's weights are updated based on training-set loss.

**Validation set (typically 10–20% of data):**

Used to make hyperparameter decisions: which regularization strength, which architecture, how many epochs. You run your model on the validation set and pick the version that performs best. Because you are looking at the validation score while making decisions, it is informing your choices and cannot be used as an unbiased performance estimate.

**Test set (typically 10–20% of data):**

**Used exactly once.** Report this number in your paper or to your stakeholders. Never make any model decision based on the test set — not architecture, not preprocessing, not feature selection. Once you have looked at the test set performance, it is "spent."

### The Information Hierarchy

```
Training data → model parameters (weights)
Validation data → hyperparameters (lambda, architecture, epochs)
Test data → final performance estimate (reported once)
```

Each level uses data from the level above to evaluate choices, and is itself evaluated by the level below.

**Why can't you just use training error?** As model complexity increases, training error decreases monotonically — you can always memorize training data by adding more parameters. Training error cannot detect overfitting.

**Why can't you just use test error?** If you tune on test error (even indirectly, by running many experiments and selecting the best), the test error is no longer unbiased. Each time you look and adapt, you "spend" some of the test set's value.

### Practical Guidance for Engineers

| Dataset size | Recommended split | Validation strategy |
|---|---|---|
| < 1,000 examples | 60/20/20 | Use cross-validation (Section 9.7) |
| 1,000 – 100,000 | 70/15/15 | Hold-out validation |
| > 100,000 | 80/10/10 | Hold-out (val set is large enough) |
| Time-series data | Chronological split only | No random shuffling — future cannot inform past |

**Time-series warning:** For sequential data (financial, sensor, text with temporal structure), always split chronologically. Training on future data to predict the past is data leakage — a silent but fatal mistake.

### Engineer's Angle

```python
import random

def train_val_test_split(data, train_frac=0.70, val_frac=0.15, seed=42):
    """
    Split data into train/val/test sets.
    test_frac = 1 - train_frac - val_frac.

    Returns (train, val, test) as lists.
    """
    rng = random.Random(seed)
    data_copy = list(data)
    rng.shuffle(data_copy)

    n = len(data_copy)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = data_copy[:n_train]
    val   = data_copy[n_train:n_train + n_val]
    test  = data_copy[n_train + n_val:]
    return train, val, test

# Example: 1000 samples
data = list(range(1000))
train, val, test = train_val_test_split(data)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
# Train: 700, Val: 150, Test: 150

# Time-series: always chronological
def time_series_split(data, train_frac=0.70, val_frac=0.15):
    """Chronological split — NO shuffling."""
    n = len(data)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = data[:n_train]
    val   = data[n_train:n_train + n_val]
    test  = data[n_train + n_val:]
    return train, val, test

# Pseudocode for hyperparameter tuning workflow:
# 
# best_val_loss = float('inf')
# best_lambda = None
#
# for lam in [0.001, 0.01, 0.1, 1.0]:          # hyperparameter search
#     model = train(train_set, lambda=lam)       # fit on train
#     val_loss = evaluate(model, val_set)        # score on val
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_lambda = lam
#
# final_model = train(train_set, lambda=best_lambda)
# test_loss = evaluate(final_model, test_set)    # report ONCE
# print(f"Test loss: {test_loss}")
```

---

## 9.7 Cross-Validation

### Plain English First

A 70/15/15 split works well when you have thousands of examples. But when your dataset is small — say, 200 examples — a single validation fold of 30 examples is noisy. The performance estimate on 30 examples may vary a lot depending on which 30 you happened to set aside.

**K-fold cross-validation** solves this by getting $k$ different performance estimates from the same data and averaging them. You train $k$ different models, each time leaving out a different fold for validation. Every example is in the validation set exactly once.

### How K-Fold Works

1. Randomly partition the dataset into $k$ equally-sized folds: $F_1, F_2, \ldots, F_k$.
2. For $i = 1$ to $k$:
   a. **Train** on all folds except $F_i$ (the "hold-out" fold).
   b. **Evaluate** on $F_i$. Record the performance metric (e.g., MSE or accuracy).
3. **Report** the mean (and optionally standard deviation) across the $k$ fold scores.

### Formal Notation

Let $\text{err}_i$ be the validation error on fold $i$.

$$\widehat{\text{CV}}_k = \frac{1}{k} \sum_{i=1}^{k} \text{err}_i$$

The standard deviation $\text{std}(\text{err}_1, \ldots, \text{err}_k)$ quantifies the **stability** of the estimate.

**Common choices of $k$:**
- $k = 5$: good default. Low computational cost, reasonable variance.
- $k = 10$: slightly better statistical properties, 2$\times$ cost of 5-fold.
- $k = n$ (leave-one-out cross-validation, LOO-CV): minimal bias, but requires $n$ training runs. Used when $n$ is very small.

### When to Use Cross-Validation

**Use it when:**
- Dataset is small (< 5,000 examples) and you cannot afford to "waste" data on a fixed validation set.
- You are comparing multiple model configurations and need reliable estimates.
- You want error bars on your performance metric.

**Do not use it as a substitute for a test set.** Cross-validation gives you the validation error across folds. You still need a held-out test set that was never used to make any model decision.

**Do not use it for time-series data.** Use time-series cross-validation instead (each fold respects chronological order: train on past, validate on future).

### Worked Example 9.7.1 — 5-Fold Cross-Validation

Dataset of 10 examples (for illustration). We use a constant-mean predictor and measure MSE.

True values: $\mathbf{y} = (1, 3, 2, 5, 4, 7, 6, 9, 8, 10)$.

Split into 5 folds of 2 examples each:

> **Note on fold assignment:** The table below assigns folds in sequential order (fold 1 = indices 0–1, fold 2 = indices 2–3, etc.) to keep the arithmetic easy to follow. The Python implementation in the Engineer's Angle section uses `rng.shuffle()` before splitting, so running the code will produce different fold assignments and different per-fold MSE values — though the cross-validated average should remain similar for well-mixed data.

| Fold | Val indices | Val $y$ | Train $y$ (remaining 8) | Train mean | Fold MSE |
|---|---|---|---|---|---|
| 1 | 0, 1 | 1, 3 | 2,5,4,7,6,9,8,10 | $51/8=6.375$ | $\frac{(1-6.375)^2+(3-6.375)^2}{2} = \frac{28.891+11.391}{2} = 20.141$ |
| 2 | 2, 3 | 2, 5 | 1,3,4,7,6,9,8,10 | $48/8=6.0$ | $\frac{(2-6)^2+(5-6)^2}{2} = \frac{16+1}{2} = 8.5$ |
| 3 | 4, 5 | 4, 7 | 1,3,2,5,6,9,8,10 | $44/8=5.5$ | $\frac{(4-5.5)^2+(7-5.5)^2}{2} = \frac{2.25+2.25}{2} = 2.25$ |
| 4 | 6, 7 | 6, 9 | 1,3,2,5,4,7,8,10 | $40/8=5.0$ | $\frac{(6-5)^2+(9-5)^2}{2} = \frac{1+16}{2} = 8.5$ |
| 5 | 8, 9 | 8, 10 | 1,3,2,5,4,7,6,9 | $37/8=4.625$ | $\frac{(8-4.625)^2+(10-4.625)^2}{2} = \frac{11.391+28.891}{2} = 20.141$ |

$$\widehat{\text{CV}}_5 = \frac{20.141 + 8.5 + 2.25 + 8.5 + 20.141}{5} = \frac{59.532}{5} = 11.906$$

**Verify fold 3 arithmetic:**

$(4 - 5.5)^2 = (-1.5)^2 = 2.25$, $(7 - 5.5)^2 = (1.5)^2 = 2.25$. Average: $\frac{2.25 + 2.25}{2} = 2.25$. $\checkmark$

**Verify fold 5 arithmetic:**

Train mean $= (1+3+2+5+4+7+6+9)/8 = 37/8 = 4.625$.
$(8 - 4.625)^2 = (3.375)^2 = 11.390625 \approx 11.391$.
$(10 - 4.625)^2 = (5.375)^2 = 28.890625 \approx 28.891$.
Average: $(11.391 + 28.891)/2 = 40.282/2 = 20.141$. $\checkmark$

> **Note:** The worked example above uses sequential (non-shuffled) fold assignment — fold 1 gets indices 0–1, fold 2 gets indices 2–3, and so on — so that the table arithmetic can be followed by hand. The Python code below follows best practice and shuffles the data before partitioning, which means the fold assignments and resulting MSE values will differ from the table. Both approaches are correct; the code is what you would use in practice.

### Engineer's Angle

```python
import math
import random
import statistics

def k_fold_split(data, k, seed=42):
    """
    Partition data into k approximately equal folds.
    Returns list of k folds, each a list of (index, item) pairs.
    Note: data is shuffled before partitioning (best practice), so fold
    assignments differ from the sequential worked example above.
    """
    rng = random.Random(seed)
    indexed = list(enumerate(data))
    rng.shuffle(indexed)
    fold_size = len(indexed) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(indexed)
        folds.append(indexed[start:end])
    return folds

def constant_predictor_mse(train_y, val_y):
    """MSE of the constant (training mean) predictor on val_y."""
    mean_pred = statistics.mean(train_y)
    return statistics.mean((y - mean_pred) ** 2 for y in val_y)

# Dataset
y = [1, 3, 2, 5, 4, 7, 6, 9, 8, 10]
k = 5

folds = k_fold_split(y, k)
fold_mses = []

for i, val_fold in enumerate(folds):
    val_indices, val_y = zip(*val_fold)
    train_y = [y[idx] for j, fold in enumerate(folds) if j != i
               for idx, _ in fold]
    mse = constant_predictor_mse(list(train_y), list(val_y))
    fold_mses.append(mse)
    print(f"Fold {i+1}: val_y={list(val_y)}, train_mean={statistics.mean(train_y):.4f}, MSE={mse:.4f}")

cv_score = statistics.mean(fold_mses)
cv_std   = statistics.stdev(fold_mses)
print(f"\n5-Fold CV MSE: {cv_score:.4f} ± {cv_std:.4f}")
# Report both the mean and std — the std tells you how stable the estimate is
```

---

## 9.8 Hypothesis Testing — Is Your Model Improvement Real?

### Plain English First

You trained Model B and it scores 82% accuracy on the test set, up from Model A's 70%. Should you ship Model B? Not yet — you need to ask: **could this difference be due to random chance?**

If you only have 100 test examples, a 12-point accuracy difference might be a statistical accident. If you have 10,000 test examples, a 12-point difference is almost certainly real. Hypothesis testing gives you a rigorous framework to decide.

The key idea: you assume the two models are actually equally good (the **null hypothesis**) and ask: if that were true, how surprising is the observed difference? If the difference is very surprising under the null hypothesis, you reject it — the improvement is likely real.

### Formal Notation

**Null hypothesis ($H_0$):** The two models have the same true accuracy. The observed difference is due to sampling variability.

**Alternative hypothesis ($H_1$):** The models have different true accuracies.

**P-value:** The probability, assuming $H_0$ is true, of observing a difference at least as extreme as the one observed.

$$p\text{-value} = P(\text{test statistic} \geq \text{observed} \mid H_0)$$

**Decision rule:** If $p\text{-value} < \alpha$ (the significance level, typically $\alpha = 0.05$), reject $H_0$. The improvement is statistically significant.

**Warning on p-values:** A small p-value does not mean the improvement is large or practically important. It means the data is inconsistent with the null hypothesis. A 0.1% improvement can be statistically significant with enough data; a 5% improvement can be statistically insignificant with too little data. Always pair p-values with **effect sizes** and **confidence intervals** (Section 9.9).

### Two-Proportion Z-Test

For comparing two model accuracies, the appropriate test is the **two-proportion z-test**.

**Setup:** Model A is correct on $k_A$ out of $n_A$ test examples. Model B is correct on $k_B$ out of $n_B$ test examples. Observed accuracies: $\hat{p}_A = k_A / n_A$ and $\hat{p}_B = k_B / n_B$.

**Pooled proportion (under $H_0$: both models have the same true accuracy):**

$$\hat{p}_{\text{pool}} = \frac{k_A + k_B}{n_A + n_B}$$

**Standard error of the difference (under $H_0$):**

$$\text{SE} = \sqrt{\hat{p}_{\text{pool}}(1 - \hat{p}_{\text{pool}})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}$$

**Z-statistic:**

$$z = \frac{\hat{p}_B - \hat{p}_A}{\text{SE}}$$

Under $H_0$, $z$ approximately follows $\mathcal{N}(0, 1)$ (using the Central Limit Theorem from Chapter 8). The two-tailed p-value is:

$$p\text{-value} = 2\bigl(1 - \Phi(|z|)\bigr)$$

where $\Phi$ is the standard normal CDF from Chapter 8.

### Worked Example 9.8.1 — Is Model B Better?

Model A: 70 correct out of 100. $\hat{p}_A = 0.70$.
Model B: 82 correct out of 100. $\hat{p}_B = 0.82$.

**Step 1 — Pooled proportion:**
$$\hat{p}_{\text{pool}} = \frac{70 + 82}{100 + 100} = \frac{152}{200} = 0.76$$

**Step 2 — Standard error:**
$$\text{SE} = \sqrt{0.76 \times 0.24 \times \left(\frac{1}{100} + \frac{1}{100}\right)} = \sqrt{0.1824 \times 0.02} = \sqrt{0.003648}$$
$$= 0.06040$$

**Step 3 — Z-statistic:**
$$z = \frac{0.82 - 0.70}{0.06040} = \frac{0.12}{0.06040} \approx 1.987$$

**Step 4 — P-value:**
$$\Phi(1.987) = \frac{1}{2}\!\left[1 + \text{erf}\!\left(\frac{1.987}{\sqrt{2}}\right)\right] = \frac{1}{2}\!\left[1 + \text{erf}(1.405)\right]$$

$\text{erf}(1.405) \approx 0.9531$, so $\Phi(1.987) \approx \frac{1}{2}(1.9531) = 0.9766$.

$$p\text{-value} = 2(1 - 0.9766) = 2(0.0234) = 0.0469$$

**Decision:** $p\text{-value} = 0.047 < 0.05$. We reject $H_0$ at the 5% significance level. The improvement from 70% to 82% on 100 test examples is statistically significant — it is unlikely to be random chance.

**Important context:** The result is barely significant ($p \approx 0.047$, just under the threshold of $0.05$). This should make you cautious:

1. The effect might not replicate. A result at the borderline of significance is fragile.
2. Consider collecting more test data to get a more reliable estimate.
3. Check whether a 12-percentage-point improvement matters in your application (practical significance vs. statistical significance).

### Engineer's Angle

```python
import math

def phi(z):
    """CDF of standard normal N(0,1). Uses math.erf (stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def two_proportion_z_test(k_A, n_A, k_B, n_B):
    """
    Two-proportion z-test: H0 is that both models have equal true accuracy.
    Returns (z_stat, p_value_two_tailed).
    """
    p_A = k_A / n_A
    p_B = k_B / n_B
    p_pool = (k_A + k_B) / (n_A + n_B)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))
    z  = (p_B - p_A) / se
    p_value = 2 * (1 - phi(abs(z)))

    print(f"p_A = {p_A:.4f}, p_B = {p_B:.4f}")
    print(f"p_pool = {p_pool:.4f}, SE = {se:.6f}")
    print(f"z = {z:.4f}, p-value = {p_value:.6f}")

    if p_value < 0.05:
        print(f"Result: SIGNIFICANT at alpha=0.05 (p={p_value:.4f} < 0.05)")
    else:
        print(f"Result: NOT significant at alpha=0.05 (p={p_value:.4f} >= 0.05)")

    return z, p_value

print("Example: 70 vs 82 out of 100")
z, pv = two_proportion_z_test(k_A=70, n_A=100, k_B=82, n_B=100)
# p-value ≈ 0.047 → significant

print("\nExample: 70 vs 75 out of 100 (small difference)")
z2, pv2 = two_proportion_z_test(k_A=70, n_A=100, k_B=75, n_B=100)
# p-value ≈ 0.35 → not significant

print("\nExample: 700 vs 820 out of 1000 (same difference, more data)")
z3, pv3 = two_proportion_z_test(k_A=700, n_A=1000, k_B=820, n_B=1000)
# p-value << 0.001 → highly significant
# Same 12-point gap, 10x more data → much stronger evidence
```

---

## 9.9 Confidence Intervals

### Plain English First

A point estimate like "our model achieves 82% accuracy" is incomplete. It tells you the center of the estimate but nothing about the uncertainty. A confidence interval gives you a range that quantifies that uncertainty.

The common reading of a 95% confidence interval $[\hat{p} - E, \hat{p} + E]$ is: "there is a 95% probability that the true accuracy lies in this range."

**This reading is wrong.**

Here is the correct interpretation, and why the wrong reading is so tempting but incorrect.

### The Correct Interpretation

A 95% confidence interval is constructed by a procedure. If you repeat the procedure — collect a new sample, compute the interval — across many repetitions, **95% of the intervals you construct will contain the true parameter**.

The true parameter $p^*$ is a fixed (unknown) number — not a random variable. The **interval** is the random object. Once you compute a specific interval $[0.7412, 0.8988]$, that interval either contains $p^*$ or it doesn't — there is no probability to assign to it. The 95% refers to the **long-run frequency** of the construction procedure, not to the probability that any particular realized interval is correct.

**Why the common reading feels natural:** After constructing the interval, you want to say something about the probability of $p^*$. That impulse is valid — it is the **Bayesian** question. In Bayesian inference, after seeing data, you compute a **credible interval**: a range that contains the true parameter with 95% posterior probability. That is a probability statement about $p^*$. The frequentist confidence interval is not.

**Practical implication:** "I am 95% confident" means "the procedure that generated this interval is correct 95% of the time." It is a statement about the reliability of your measurement process, not a probability about this specific interval.

### Formal Notation

For a proportion $\hat{p}$ estimated from $n$ observations, the **Wald confidence interval** at level $1 - \alpha$ is:

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

where $z_{\alpha/2}$ is the critical value from the standard normal such that $\Phi(z_{\alpha/2}) = 1 - \alpha/2$.

For $\alpha = 0.05$ (95% CI): $z_{0.025} = 1.96$ (because $\Phi(1.96) \approx 0.975$).

For $\alpha = 0.01$ (99% CI): $z_{0.005} \approx 2.576$.

**Trust this result:** $z_{0.025} = 1.96$ is the value such that 2.5% of the standard normal distribution lies above it. It comes from $\Phi^{-1}(0.975)$, which does not have a closed form but is a standard table entry.

### Worked Example 9.9.1 — Accuracy Confidence Interval

Model B achieved 82% accuracy on $n = 100$ test examples. Construct a 95% confidence interval for the true accuracy $p^*$.

**Step 1 — Point estimate:** $\hat{p} = 82/100 = 0.82$.

**Step 2 — Standard error:**
$$\text{SE} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.82 \times 0.18}{100}} = \sqrt{\frac{0.1476}{100}} = \sqrt{0.001476} = 0.03842$$

**Step 3 — Margin of error:**
$$E = 1.96 \times 0.03842 = 0.07530$$

**Step 4 — Confidence interval:**
$$[0.82 - 0.0753, \; 0.82 + 0.0753] = [0.7447, \; 0.8953]$$

**Correct interpretation:** The procedure used to construct this interval has the property that 95% of such intervals contain the true accuracy. This specific interval $[0.745, 0.895]$ is a realization of that procedure.

**Practical reading:** The true accuracy is somewhere in the range 74.5% to 89.5%. That is a wide interval — a 15-percentage-point spread — because $n = 100$ is not many test examples.

### Worked Example 9.9.2 — How Many Examples Do You Need?

How large a test set is needed for a 95% CI with margin of error $E = \pm 2\%$ around an accuracy of $\hat{p} = 0.82$?

**Rearrange** $E = 1.96\sqrt{\hat{p}(1-\hat{p})/n}$ for $n$:

$$n = \frac{(1.96)^2 \hat{p}(1-\hat{p})}{E^2} = \frac{3.8416 \times 0.82 \times 0.18}{(0.02)^2}$$

$$= \frac{3.8416 \times 0.1476}{0.0004} = \frac{0.5672}{0.0004} = 1418$$

**You need approximately 1,418 test examples** to achieve a 95% CI with half-width of 2 percentage points. With only 100 test examples (as in Example 9.9.1), the half-width is about 7.5 percentage points — four times wider.

| Test set size | 95% CI half-width (at $\hat{p}=0.82$) |
|---|---|
| 100 | ±7.5% |
| 400 | ±3.8% |
| 1,000 | ±2.4% |
| 1,418 | ±2.0% |
| 10,000 | ±0.75% |

**The key insight:** Halving the confidence interval width requires quadrupling the test set size (because SE scales as $1/\sqrt{n}$). Evaluating models reliably is expensive.

### Engineer's Angle

```python
import math

def phi(z):
    """CDF of standard normal."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def proportion_ci(k, n, confidence=0.95):
    """
    Wald confidence interval for a proportion.
    k: number of successes.
    n: total trials.
    Returns (point_estimate, lower, upper, margin_of_error).
    """
    p_hat = k / n
    alpha  = 1.0 - confidence
    # z_{alpha/2}: standard normal critical value
    # Use bisection to find z such that Phi(z) = 1 - alpha/2
    target = 1.0 - alpha / 2
    lo, hi = 0.0, 5.0
    for _ in range(60):          # 60 iterations → precision ~1e-18
        mid = (lo + hi) / 2
        if phi(mid) < target:
            lo = mid
        else:
            hi = mid
    z_crit = (lo + hi) / 2

    se  = math.sqrt(p_hat * (1 - p_hat) / n)
    moe = z_crit * se
    return p_hat, p_hat - moe, p_hat + moe, moe

# Model B: 82/100
p_hat, lo, hi, moe = proportion_ci(82, 100)
print(f"p_hat = {p_hat:.4f}")
print(f"95% CI: [{lo:.4f}, {hi:.4f}]  (±{moe:.4f})")
# [0.7447, 0.8953] — ±0.0753

# How many samples for ±2% CI?
def required_sample_size(p_hat, margin, confidence=0.95):
    """Minimum n for Wald CI with given half-width margin."""
    alpha  = 1.0 - confidence
    target = 1.0 - alpha / 2
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if phi(mid) < target else (lo, mid)
    z_crit = (lo + hi) / 2
    return math.ceil(z_crit**2 * p_hat * (1 - p_hat) / margin**2)

n_needed = required_sample_size(0.82, 0.02)
print(f"\nFor ±2% CI at 95% confidence: need n = {n_needed}")   # 1418

print("\nCI widths by test set size:")
for n in [100, 400, 1000, 1418, 10000]:
    _, lo, hi, moe = proportion_ci(int(0.82 * n), n)
    print(f"  n={n:6d}: ±{moe:.4f}  [{lo:.4f}, {hi:.4f}]")
```

---

## 9.10 Connecting Everything: The Full Model Evaluation Pipeline

This chapter has built a complete framework for statistical decision-making in machine learning. Here is how the pieces connect in a real evaluation pipeline.

**During training:** MLE (Section 9.2) tells you which loss to use. For regression under Gaussian noise, minimize MSE. For classification, minimize cross-entropy. These are not arbitrary — they are maximum likelihood under the appropriate probabilistic model.

**Choosing complexity:** The bias-variance tradeoff (Section 9.4) tells you that model error decomposes into bias, variance, and irreducible noise. Use cross-validation (Section 9.7) on the validation set to find the model complexity (and regularization strength from Section 9.5) that minimizes total validation error.

**Regularization decisions:** Use Ridge (L2) when you want to shrink all weights smoothly. Use Lasso (L1) when you expect most features to be irrelevant and want a sparse model. Interpret both as MAP estimation under a prior on weights (Section 9.5.3).

**Reporting results:** Report a confidence interval (Section 9.9), not just a point estimate. For any improvement claim, run a hypothesis test (Section 9.8) to verify the difference is not due to chance. Never look at the test set more than once.

**The closed-form exception:** When $d$ is small (< ~10,000 features) and the model is linear, the Normal Equation (Section 9.3) gives the exact MLE solution without gradient descent. Know when to use it.

---

## 9.11 Summary Table

| Concept | Formula / Rule | ML Application |
|---|---|---|
| **Point estimate** | $\hat{\theta} = $ function of data | Weight vector, predicted probability |
| **MLE** | $\hat{\theta} = \arg\max_\theta \sum_i \log p(x_i \mid \theta)$ | Foundation of all standard loss functions |
| **Bernoulli MLE** | $\hat{p} = k/n$ | Classifier calibration, A/B test counts |
| **Gaussian mean MLE** | $\hat{\mu} = \bar{x}$ | Regression, Gaussian VAE encoder mean |
| **Log-likelihood trick** | $\log \prod_i p_i = \sum_i \log p_i$ | Numerical stability of all training |
| **Normal Equation** | $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$ | Exact closed-form linear regression |
| **Use Normal Eq. when** | $d \lesssim 10{,}000$, data fits in RAM | Small tabular datasets, prototyping |
| **Bias$^2$** | $(\mathbb{E}[\hat{f}(x)] - f(x))^2$ | Underfitting, too-simple models |
| **Variance** | $\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$ | Overfitting, too-complex models |
| **MSE decomposition** | Bias$^2$ + Variance + $\sigma^2$ | Guide model complexity selection |
| **Ridge (L2)** | $+\lambda\|\mathbf{w}\|_2^2$; $(X^TX+\lambda I)^{-1}X^T\mathbf{y}$ | Shrinks all weights, stabilizes inversion |
| **Lasso (L1)** | $+\lambda\|\mathbf{w}\|_1$ | Drives weights to exactly zero (sparsity) |
| **Bayesian connection** | Gaussian prior $\Rightarrow$ L2; Laplace prior $\Rightarrow$ L1 | Regularization is MAP inference |
| **Train/Val/Test** | Train: fit params; Val: fit hyperparams; Test: report once | Core evaluation discipline |
| **K-fold CV** | $\widehat{\text{CV}}_k = \frac{1}{k}\sum_i \text{err}_i$ | Reliable estimates on small datasets |
| **Two-proportion z-test** | $z = (\hat{p}_B - \hat{p}_A)/\text{SE}$ | Testing whether model improvement is real |
| **P-value** | $P(\text{observed diff} \mid H_0)$ | Surprisingness of result under null |
| **95% CI for proportion** | $\hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/n}$ | Uncertainty in reported accuracy |
| **CI misconception** | "95% of constructed intervals contain $\theta^*$" (not: "$\theta^*$ is in this interval with 95% prob.") | Correct frequentist interpretation |

---

## 9.12 Exercises

### Exercise 9.1 [Easy] — MLE for Bernoulli

A binary spam classifier is evaluated on 8 emails, and correctly predicts 6 of them.

**(a)** Write the log-likelihood function $\ell(p)$ for this data under a Bernoulli model.

**(b)** Differentiate and solve to find $\hat{p}_{\text{MLE}}$.

**(c)** Evaluate the log-likelihood at $\hat{p}_{\text{MLE}}$ and at $p = 0.9$. Confirm that $\hat{p}_{\text{MLE}}$ gives the higher value.

**(d)** Why does the MLE estimate make intuitive sense?

---

**Solution:**

**(a)** $n = 8$, $k = 6$. The log-likelihood is:

$$\ell(p) = k \log p + (n - k)\log(1-p) = 6\log p + 2\log(1-p)$$

**(b)** Differentiating and setting to zero:

$$\frac{d\ell}{dp} = \frac{6}{p} - \frac{2}{1-p} = 0$$

$$6(1-p) = 2p$$

$$6 - 6p = 2p$$

$$6 = 8p$$

$$\hat{p}_{\text{MLE}} = \frac{6}{8} = 0.75$$

**(c)** Evaluating at $\hat{p} = 0.75$:

$$\ell(0.75) = 6\log(0.75) + 2\log(0.25)$$
$$= 6(-0.2877) + 2(-1.3863)$$
$$= -1.7262 - 2.7726 = -4.4988$$

Evaluating at $p = 0.9$:

$$\ell(0.9) = 6\log(0.9) + 2\log(0.1)$$
$$= 6(-0.1054) + 2(-2.3026)$$
$$= -0.6322 - 4.6052 = -5.2374$$

Since $-4.4988 > -5.2374$, the MLE at $p = 0.75$ has higher log-likelihood than $p = 0.9$. $\checkmark$

**(d)** $\hat{p}_{\text{MLE}} = 6/8 = 0.75$ is simply the observed success rate. Intuitively, if you see 6 successes in 8 trials, the most natural estimate for the underlying success probability is the fraction you observed. MLE confirms this intuition rigorously.

---

### Exercise 9.2 [Easy] — Normal Equation

Fit a linear model $\hat{y} = w_1 x + w_0$ to the data points $(0, 1)$, $(1, 3)$, $(2, 4)$ using the Normal Equation.

**(a)** Write the design matrix $X$ and target vector $\mathbf{y}$.

**(b)** Compute $X^TX$ and $X^T\mathbf{y}$.

**(c)** Compute $(X^TX)^{-1}$ using the $2 \times 2$ inversion formula.

**(d)** Compute $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$ and state the fitted model.

**(e)** Verify that the gradient condition $X^TX\mathbf{w} = X^T\mathbf{y}$ holds.

---

**Solution:**

**(a)**

$$X = \begin{bmatrix} 0 & 1 \\ 1 & 1 \\ 2 & 1 \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} 1 \\ 3 \\ 4 \end{bmatrix}$$

**(b)**

$$X^TX = \begin{bmatrix} 0 & 1 & 2 \\ 1 & 1 & 1 \end{bmatrix}\begin{bmatrix} 0 & 1 \\ 1 & 1 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 0+1+4 & 0+1+2 \\ 0+1+2 & 1+1+1 \end{bmatrix} = \begin{bmatrix} 5 & 3 \\ 3 & 3 \end{bmatrix}$$

$$X^T\mathbf{y} = \begin{bmatrix} 0 & 1 & 2 \\ 1 & 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 0+3+8 \\ 1+3+4 \end{bmatrix} = \begin{bmatrix} 11 \\ 8 \end{bmatrix}$$

**(c)**

$$\det\begin{bmatrix} 5 & 3 \\ 3 & 3 \end{bmatrix} = 5 \times 3 - 3 \times 3 = 15 - 9 = 6$$

$$(X^TX)^{-1} = \frac{1}{6}\begin{bmatrix} 3 & -3 \\ -3 & 5 \end{bmatrix} = \begin{bmatrix} 1/2 & -1/2 \\ -1/2 & 5/6 \end{bmatrix}$$

**(d)**

$$w_1 = \frac{1}{2}(11) + \left(-\frac{1}{2}\right)(8) = 5.5 - 4.0 = 1.5$$

$$w_0 = \left(-\frac{1}{2}\right)(11) + \frac{5}{6}(8) = -5.5 + \frac{40}{6} = -5.5 + 6.\overline{6} = 1.1\overline{6} = \frac{7}{6}$$

**Fitted model:** $\hat{y} = 1.5x + \frac{7}{6} \approx 1.5x + 1.167$

**(e)** Verify $X^TX\mathbf{w} = X^T\mathbf{y}$:

$$\begin{bmatrix} 5 & 3 \\ 3 & 3 \end{bmatrix}\begin{bmatrix} 3/2 \\ 7/6 \end{bmatrix} = \begin{bmatrix} 5(3/2) + 3(7/6) \\ 3(3/2) + 3(7/6) \end{bmatrix} = \begin{bmatrix} 15/2 + 7/2 \\ 9/2 + 7/2 \end{bmatrix} = \begin{bmatrix} 22/2 \\ 16/2 \end{bmatrix} = \begin{bmatrix} 11 \\ 8 \end{bmatrix} = X^T\mathbf{y} \checkmark$$

---

### Exercise 9.3 [Medium] — Bias-Variance Decomposition

True function: $f(x) = x^2$. Test point: $x_0 = 2$, so $f(2) = 4$. Noise: $\sigma^2 = 1$.

You have two models:
- **Model P** (polynomial): predicts $\hat{f}_P(2) \in \{3.5, 4.0, 4.5, 4.0\}$ with equal probability across four training datasets.
- **Model C** (constant): always predicts $\hat{f}_C(x) = 2$ regardless of training data.

**(a)** Compute $\text{Bias}^2$, Variance, and MSE for Model P.

**(b)** Compute $\text{Bias}^2$, Variance, and MSE for Model C.

**(c)** Which model has lower MSE? Which would you prefer, and why?

**(d)** If you doubled the training set size, which component (bias or variance) of Model P's error would decrease? What about Model C?

---

**Solution:**

**(a) Model P:**

$$\mathbb{E}[\hat{f}_P(2)] = \frac{3.5 + 4.0 + 4.5 + 4.0}{4} = \frac{16}{4} = 4$$

$$\text{Bias}_P = 4 - 4 = 0, \qquad \text{Bias}_P^2 = 0$$

$$\text{Var}_P = \frac{(3.5-4)^2 + (4.0-4)^2 + (4.5-4)^2 + (4.0-4)^2}{4}$$
$$= \frac{0.25 + 0 + 0.25 + 0}{4} = \frac{0.5}{4} = 0.125$$

$$\text{MSE}_P = 0 + 0.125 + 1 = 1.125$$

**(b) Model C:**

$$\mathbb{E}[\hat{f}_C(2)] = 2 \quad \text{(deterministic)}$$

$$\text{Bias}_C = 2 - 4 = -2, \qquad \text{Bias}_C^2 = 4$$

$$\text{Var}_C = 0 \quad \text{(constant predictor)}$$

$$\text{MSE}_C = 4 + 0 + 1 = 5$$

**(c)** Model P has MSE = 1.125 vs Model C's MSE = 5. **Model P is much better.** Model C's large bias ($-2$) overwhelms its zero variance. The constant predictor of $2$ is far from the true value of $4$.

**(d)** More training data helps reduce **variance** but not bias. 

For Model P: More data would reduce variance (the spread of predictions across training sets shrinks). Model P's bias is already zero, so more data does not help further with bias.

For Model C: More data does not help at all — it always predicts 2 regardless, so its bias stays at $-2$ and its variance stays at 0. More data cannot fix a model that ignores the input.

---

### Exercise 9.4 [Medium] — Regularization and the Bayesian Interpretation

A linear regression model has learned weight $w = 3.0$ on a dataset with $n = 50$ examples.

**(a)** With L2 regularization ($\lambda = 0.5$), the regularized loss adds $\lambda w^2 = 0.5 \times 9 = 4.5$ to the MSE. Suppose the unregularized gradient at $w = 3.0$ is $\partial \mathcal{L}_0 / \partial w = -0.2$ (pushing $w$ larger). Write the total gradient (MSE + L2 penalty) at $w = 3.0$.

**(b)** With L1 regularization ($\lambda = 0.5$) and the same conditions, write the total gradient at $w = 3.0$.

**(c)** Now consider the point $w = 0.03$. With L2 ($\lambda = 0.5$), the L2 gradient is $2\lambda w = 1 \times 0.03 = 0.03$. Suppose the data gradient is still $-0.2$. What is the total gradient, and which direction does the update push $w$?

**(d)** With L1 ($\lambda = 0.5$) at $w = 0.03$, the L1 subgradient is $+\lambda = +0.5$ (positive, pushing toward zero). Suppose the data gradient is $-0.2$. What is the total (sub)gradient? In which direction does the update push $w$? Contrast with the L2 case from (c).

**(e)** Suppose instead $w = 0.0$ and the data gradient is $-0.2$ (pushing toward positive $w$). With L1 ($\lambda = 0.5$), the subgradient of $|w|$ at $w=0$ is any value in $[-0.5, 0.5]$. Does there exist a subgradient value that makes the total (sub)gradient zero? What does this mean for $w$?

---

**Solution:**

**(a)** The L2 gradient contribution is $2\lambda w = 2(0.5)(3.0) = 3.0$.

Total gradient: $\frac{\partial \mathcal{L}_0}{\partial w} + 2\lambda w = -0.2 + 3.0 = +2.8$

The total gradient is positive, pushing $w$ downward (away from 3.0, toward smaller values). The L2 penalty dominates.

**(b)** The L1 gradient contribution (for $w > 0$) is $\lambda \cdot \text{sign}(w) = 0.5 \times (+1) = +0.5$.

Total gradient: $-0.2 + 0.5 = +0.3$

Again positive — pushing $w$ downward.

**(c)** Total gradient at $w = 0.03$ with L2: $-0.2 + 0.03 = -0.17$.

The total gradient is **negative** — pushing $w$ upward (toward positive values). The data gradient of $-0.2$ dominates the small L2 penalty of $0.03$. The weight will increase.

**(d)** Total (sub)gradient at $w = 0.03$ with L1: $-0.2 + 0.5 = +0.3$.

The total subgradient is **positive** — pushing $w$ downward toward zero. This is the key difference from L2: even when $w$ is very small (0.03), the L1 penalty contributes its full magnitude $\lambda = 0.5$, which here overcomes the data gradient and forces $w$ toward zero.

With L2, the penalty shrank to 0.03 and the data gradient won. With L1, the penalty stays at 0.5 and the data gradient ($-0.2$) loses.

**(e)** At $w = 0$, the L1 subgradient $g \in [-0.5, 0.5]$. Total (sub)gradient: $-0.2 + g$.

Setting to zero: $g = 0.2$. Since $0.2 \in [-0.5, 0.5]$, **yes**, there is a valid subgradient that makes the total zero.

**Meaning:** The optimality condition is satisfied with $w = 0$. The point $w = 0$ is a minimum of the L1-regularized loss even though the data gradient is $-0.2$ (wanting to push $w$ positive). The L1 subgradient "absorbs" the data gradient's pull, keeping the weight exactly at zero. This is the mechanism of L1 sparsity: a "dead zone" of width $2\lambda$ around zero where the L1 penalty prevents any movement away from $w = 0$.

---

### Exercise 9.5 [Hard] — Full Evaluation Pipeline

A team trains two image classifiers on 8,000 training examples. They evaluate on a 1,000-example test set. Model A achieves 88% accuracy (880/1000 correct). Model B achieves 91% accuracy (910/1000 correct).

**(a)** Compute a 95% confidence interval for each model's true accuracy.

**(b)** Run a two-proportion z-test for $H_0: p_A = p_B$. Compute $z$ and the p-value. Is the improvement statistically significant at $\alpha = 0.05$?

**(c)** How large would the test set need to be for a 95% CI of ±1% on each model's accuracy (using Model B's $\hat{p} = 0.91$)?

**(d)** The team wants to now tune Model B's regularization strength using the test set (trying $\lambda \in \{0.01, 0.1, 1.0\}$ and picking the best). Explain why this is problematic and what they should have done instead.

**(e)** Suppose the team ran the hypothesis test from (b) on 20 different pairs of models, using $\alpha = 0.05$ each time. Even if all models are equally good, how many "significant" results would you expect by chance? What does this imply about interpreting multiple comparisons?

---

**Solution:**

**(a)** Using $z_{0.025} = 1.96$:

**Model A** ($\hat{p}_A = 0.88$, $n = 1000$):

$$\text{SE}_A = \sqrt{\frac{0.88 \times 0.12}{1000}} = \sqrt{\frac{0.1056}{1000}} = \sqrt{0.0001056} = 0.010276$$

$$E_A = 1.96 \times 0.010276 = 0.020141$$

$$\text{CI}_A = [0.88 - 0.0201, \; 0.88 + 0.0201] = [0.8599, \; 0.9001]$$

**Model B** ($\hat{p}_B = 0.91$, $n = 1000$):

$$\text{SE}_B = \sqrt{\frac{0.91 \times 0.09}{1000}} = \sqrt{\frac{0.0819}{1000}} = \sqrt{0.0000819} = 0.009050$$

$$E_B = 1.96 \times 0.009050 = 0.017738$$

$$\text{CI}_B = [0.91 - 0.0177, \; 0.91 + 0.0177] = [0.8923, \; 0.9277]$$

Note that the confidence intervals barely overlap. This is suggestive, but CI overlap is not a reliable significance test — always run the hypothesis test directly, as in part (b).

**(b)** Two-proportion z-test:

$$\hat{p}_{\text{pool}} = \frac{880 + 910}{1000 + 1000} = \frac{1790}{2000} = 0.895$$

$$\text{SE} = \sqrt{0.895 \times 0.105 \times \left(\frac{1}{1000} + \frac{1}{1000}\right)} = \sqrt{0.09398 \times 0.002} = \sqrt{0.00018795}$$
$$= 0.013710$$

$$z = \frac{0.91 - 0.88}{0.013710} = \frac{0.03}{0.013710} = 2.188$$

$$\Phi(2.188) = \frac{1}{2}\!\left[1 + \text{erf}\!\left(\frac{2.188}{\sqrt{2}}\right)\right] = \frac{1}{2}[1 + \text{erf}(1.547)] \approx \frac{1}{2}[1 + 0.9714] = 0.9857$$

$$p\text{-value} = 2(1 - 0.9857) = 2(0.0143) = 0.0286$$

Since $0.0286 < 0.05$, the improvement is **statistically significant** at $\alpha = 0.05$. $\checkmark$

**(c)** Required $n$ for $\hat{p} = 0.91$, $E = 0.01$:

$$n = \frac{(1.96)^2 \times 0.91 \times 0.09}{(0.01)^2} = \frac{3.8416 \times 0.0819}{0.0001} = \frac{0.31463}{0.0001} = 3146.3$$

**Required test set size: $n \approx 3{,}147$ examples** for a ±1% CI at 95% confidence.

**(d)** This is **test set contamination**. By choosing $\lambda$ based on test set performance, the team is indirectly fitting the test set. After testing three values of $\lambda$, the test error they report is no longer an unbiased estimate of generalization performance — it optimistically reflects the one $\lambda$ that happened to work best on this particular test set.

**What they should have done:** Reserve a validation set from the beginning, use it to tune $\lambda$, then evaluate the final model on the test set exactly once. If they want to tune $\lambda$ at this point (no separate validation set), they must collect a new, never-before-seen test set.

**(e)** Under $H_0$ (all models equally good), each test has a 5% false-positive rate — a 5% chance of incorrectly declaring significance. Running 20 independent tests:

$$\text{Expected false positives} = 20 \times 0.05 = 1 \text{ false "significant" result}$$

This is the **multiple comparisons problem**. With 20 tests, you expect 1 spurious "significant" result purely by chance even if none of the models differ. To control the overall false-positive rate at 5%, apply a correction such as **Bonferroni correction**: use $\alpha' = 0.05 / 20 = 0.0025$ for each individual test, so that the family-wise error rate stays at 5%.

The implication: in any ML research with many ablations, hyperparameter comparisons, or architecture trials, some reported improvements may be statistical accidents. Treating every individual comparison as independent and applying a standard threshold is statistically incorrect. Be especially skeptical of results that are "just" significant on many comparisons.

---

*Next: Chapter 10 — Putting It All Together: a complete worked example integrating linear algebra, calculus, probability, and statistics to analyze a real machine learning system end to end.*
