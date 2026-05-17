# Chapter 8: Probability II — Key Distributions

> *"Knowing which distribution fits your problem is half the battle. The other half is knowing why."*

---

Chapter 7 gave you the machinery: sample spaces, probability axioms, Bayes' theorem, random variables, expected value, variance, and the PMF/PDF framework. This chapter puts that machinery to work. We tour the handful of distributions that appear constantly in machine learning — and for each one, we answer three questions: What does it model? What are its formulas? Where does it show up in code?

---

## 8.1 Bernoulli Distribution

### Plain English First

The Bernoulli distribution is the simplest possible random variable: one trial, two outcomes — success or failure, 1 or 0, yes or no. A single fair coin flip follows a Bernoulli distribution. So does a single binary classifier prediction: either the model says "positive" (1) or "negative" (0).

The distribution has exactly one parameter, $p$, which is the probability of the success outcome. Everything else follows from that single number.

### Formal Notation

Let $X$ be a Bernoulli random variable with parameter $p \in [0, 1]$. We write $X \sim \text{Bernoulli}(p)$.

**PMF:**

$$P(X = k) = \begin{cases} p & \text{if } k = 1 \\ 1 - p & \text{if } k = 0 \end{cases}$$

A compact single-formula version: $P(X = k) = p^k (1-p)^{1-k}$ for $k \in \{0, 1\}$.

**Expected value:**

$$\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1-p) = p$$

**Variance:**

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = p - p^2 = p(1-p)$$

**Here's why** $\mathbb{E}[X^2] = p$: since $X$ only takes values 0 or 1, we have $X^2 = X$, so $\mathbb{E}[X^2] = \mathbb{E}[X] = p$.

| Quantity | Value |
|----------|-------|
| Support | $\{0, 1\}$ |
| Parameter | $p \in [0,1]$ (success probability) |
| PMF | $P(X=1) = p$, $\; P(X=0) = 1-p$ |
| $\mathbb{E}[X]$ | $p$ |
| $\text{Var}(X)$ | $p(1-p)$ |

**Trust this result:** Variance $p(1-p)$ is maximized at $p = 0.5$ (most uncertain) and equals zero at $p = 0$ or $p = 1$ (completely certain). This makes intuitive sense.

### Worked Example 8.1.1 — A Biased Coin

A biased coin lands heads with probability $p = 0.7$. Define $X = 1$ for heads, $X = 0$ for tails.

**Step 1 — PMF:**
$$P(X = 1) = 0.7, \qquad P(X = 0) = 0.3$$

**Step 2 — Verify normalization:**
$$0.7 + 0.3 = 1.0 \checkmark$$

**Step 3 — Expected value:**
$$\mathbb{E}[X] = 0.7$$

**Step 4 — Variance:**
$$\text{Var}(X) = 0.7 \times (1 - 0.7) = 0.7 \times 0.3 = 0.21$$

### Worked Example 8.1.2 — Binary Classifier Output

A spam classifier outputs probability $\hat{y} = 0.82$ for a given email. We model the predicted label $\hat{L} \sim \text{Bernoulli}(0.82)$.

$$P(\hat{L} = 1) = 0.82, \qquad P(\hat{L} = 0) = 0.18$$

If the true label is $y = 1$, the **binary cross-entropy loss** for this prediction is:

$$\mathcal{L} = -[y \log(\hat{y}) + (1-y)\log(1 - \hat{y})]$$
$$= -[1 \cdot \log(0.82) + 0 \cdot \log(0.18)]$$
$$= -\log(0.82)$$
$$\approx 0.1985$$

For a wrong prediction $\hat{y} = 0.3$ with true label $y = 1$:

$$\mathcal{L} = -\log(0.3) \approx 1.2040$$

The loss is higher when the model is confidently wrong — exactly the behavior we want. We will derive where this formula comes from in Section 8.10.

### Engineer's Angle

The Bernoulli distribution underlies **logistic regression**. The sigmoid function maps a real-valued score $s$ to a probability $\hat{y} = \sigma(s) = \frac{1}{1 + e^{-s}}$, and then the predicted label is $\hat{L} \sim \text{Bernoulli}(\hat{y})$. Training minimizes the expected binary cross-entropy loss over the dataset.

```python
import math

def bernoulli_pmf(k, p):
    """PMF of Bernoulli(p) at outcome k in {0, 1}."""
    if k == 1:
        return p
    elif k == 0:
        return 1 - p
    else:
        raise ValueError("k must be 0 or 1")

def bernoulli_mean(p):
    return p

def bernoulli_var(p):
    return p * (1 - p)

def sigmoid(s):
    return 1.0 / (1.0 + math.exp(-s))

def binary_cross_entropy(y_true, y_pred):
    """Loss for one sample under Bernoulli model."""
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

# Biased coin example
p = 0.7
print(f"Bernoulli(p=0.7): E={bernoulli_mean(p)}, Var={bernoulli_var(p)}")
# Bernoulli(p=0.7): E=0.7, Var=0.21

# Logistic regression workflow
score = 1.5                       # raw model score (logit)
y_hat = sigmoid(score)            # predicted probability
loss  = binary_cross_entropy(1, y_hat)  # true label = 1
print(f"score={score}, y_hat={y_hat:.4f}, loss={loss:.4f}")
# score=1.5, y_hat=0.8176, loss=0.2014
```

---

## 8.2 Binomial Distribution

### Plain English First

If the Bernoulli distribution models one coin flip, the Binomial distribution models $n$ independent flips and asks: how many heads? More concretely: if you run 10 independent trials each with success probability $p$, the Binomial distribution tells you the probability of getting exactly $k$ successes.

The word "independent" is critical — each trial cannot depend on the previous ones. This assumption is what lets us multiply probabilities.

### Formal Notation

Let $X$ = number of successes in $n$ independent Bernoulli($p$) trials. We write $X \sim \text{Binomial}(n, p)$.

**PMF:**

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k = 0, 1, 2, \ldots, n$$

where the **binomial coefficient** $\binom{n}{k} = \frac{n!}{k!\,(n-k)!}$ counts the number of distinct ways to arrange $k$ successes among $n$ trials.

**Here's why** the formula has that shape: $p^k$ is the probability of $k$ successes in a specific order; $(1-p)^{n-k}$ is the probability of $n-k$ failures; $\binom{n}{k}$ counts how many orderings produce exactly $k$ successes. By independence, each ordering has the same probability, so we multiply and sum.

**Expected value:**

$$\mathbb{E}[X] = np$$

**Variance:**

$$\text{Var}(X) = np(1-p)$$

**Trust this result:** Both formulas follow from viewing $X$ as the sum of $n$ independent Bernoulli($p$) variables and applying the linearity of expectation and the variance-of-sum rule for independent variables.

| Quantity | Value |
|----------|-------|
| Support | $\{0, 1, \ldots, n\}$ |
| Parameters | $n \in \mathbb{N}$ (trials), $p \in [0,1]$ (success prob.) |
| PMF | $\binom{n}{k} p^k (1-p)^{n-k}$ |
| $\mathbb{E}[X]$ | $np$ |
| $\text{Var}(X)$ | $np(1-p)$ |

### Worked Example 8.2.1 — Quality Control

A factory produces widgets. Each widget has a 40% defect rate independently ($p = 0.4$). A batch of $n = 3$ widgets is inspected. Let $X$ = number of defective widgets.

$X \sim \text{Binomial}(3,\, 0.4)$

**Step 1 — Full PMF:**

$$P(X = 0) = \binom{3}{0}(0.4)^0(0.6)^3 = 1 \times 1 \times 0.216 = 0.216$$

$$P(X = 1) = \binom{3}{1}(0.4)^1(0.6)^2 = 3 \times 0.4 \times 0.36 = 0.432$$

$$P(X = 2) = \binom{3}{2}(0.4)^2(0.6)^1 = 3 \times 0.16 \times 0.6 = 0.288$$

$$P(X = 3) = \binom{3}{3}(0.4)^3(0.6)^0 = 1 \times 0.064 \times 1 = 0.064$$

**Step 2 — Verify normalization:**
$$0.216 + 0.432 + 0.288 + 0.064 = 1.000 \checkmark$$

**Step 3 — Expected value and variance:**
$$\mathbb{E}[X] = 3 \times 0.4 = 1.2$$
$$\text{Var}(X) = 3 \times 0.4 \times 0.6 = 0.72$$

**Step 4 — Verify $\mathbb{E}[X]$ directly:**
$$\mathbb{E}[X] = 0(0.216) + 1(0.432) + 2(0.288) + 3(0.064)$$
$$= 0 + 0.432 + 0.576 + 0.192 = 1.200 \checkmark$$

### Worked Example 8.2.2 — Model Accuracy on a Batch

A classifier has true accuracy 70% ($p = 0.7$). On a batch of $n = 10$ samples, let $X$ = number correct.

$$P(X = 7) = \binom{10}{7}(0.7)^7(0.3)^3$$

**Step 1 — Binomial coefficient:**
$$\binom{10}{7} = \frac{10!}{7!\,3!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = \frac{720}{6} = 120$$

**Step 2 — Powers:**
$$(0.7)^7 = 0.0823543, \qquad (0.3)^3 = 0.027$$

**Step 3 — Multiply:**
$$P(X = 7) = 120 \times 0.0823543 \times 0.027 = 120 \times 0.002224 \approx 0.2668$$

There is about a 26.7% chance the model gets exactly 7 out of 10 correct.

**Expected correct:** $\mathbb{E}[X] = 10 \times 0.7 = 7$

### Engineer's Angle

Binomial distributions model **batch accuracy** and **A/B test counts**. When you split traffic 50/50 between model A and model B and count how many users in the B group convert, that count follows a Binomial distribution. Statistical significance tests for A/B results rely on the Binomial (or the Normal approximation to it via the Central Limit Theorem — see Section 8.5).

```python
import math

def binom_coeff(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def binom_pmf(n, k, p):
    return binom_coeff(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binom_mean(n, p):
    return n * p

def binom_var(n, p):
    return n * p * (1 - p)

# Quality control example
n, p = 3, 0.4
pmf = [binom_pmf(n, k, p) for k in range(n + 1)]
print("Binomial(3, 0.4) PMF:")
for k, prob in enumerate(pmf):
    print(f"  P(X={k}) = {prob:.6f}")
print(f"  Sum = {sum(pmf):.10f}")       # 1.0000000000
print(f"  E[X] = {binom_mean(n, p)}")   # 1.2
print(f"  Var(X) = {binom_var(n, p)}")  # 0.72
```

---

## 8.3 Categorical and Multinomial Distributions

### Plain English First

The Bernoulli distribution handles two outcomes. The **Categorical** distribution generalizes it to $K$ outcomes — one draw that lands on exactly one of $K$ categories. A single die roll is Categorical. A single image classifier prediction (cat, dog, or bird?) is Categorical.

The **Multinomial** distribution then generalizes Binomial: run $n$ independent Categorical trials and count how many fall into each category. If you roll a die 60 times, the vector of counts (how many 1s, 2s, 3s, ...) follows a Multinomial distribution.

### Formal Notation

**Categorical distribution** with $K$ classes: parameters are a probability vector $\mathbf{p} = (p_1, p_2, \ldots, p_K)$ with $p_k \geq 0$ and $\sum_{k=1}^{K} p_k = 1$.

$$P(X = k) = p_k, \qquad k \in \{1, 2, \ldots, K\}$$

The **one-hot representation** encodes outcome $k$ as a vector $\mathbf{e}_k$ where the $k$-th entry is 1 and all others are 0.

**Multinomial distribution** with $n$ trials: the count vector $(C_1, C_2, \ldots, C_K)$ where $C_k$ = number of times category $k$ appears, $\sum_k C_k = n$.

$$P(C_1 = c_1, \ldots, C_K = c_K) = \frac{n!}{c_1!\, c_2!\, \cdots\, c_K!}\, p_1^{c_1} p_2^{c_2} \cdots p_K^{c_K}$$

**Expected value of each count:**
$$\mathbb{E}[C_k] = n \, p_k$$

| Quantity | Categorical | Multinomial |
|----------|------------|-------------|
| Support | One category $k \in \{1,\ldots,K\}$ | Count vector $(c_1,\ldots,c_K)$, $\sum c_k = n$ |
| Parameters | $\mathbf{p} = (p_1,\ldots,p_K)$ | $n$, $\mathbf{p}$ |
| $\mathbb{E}[\cdot]$ | N/A (each $p_k$) | $\mathbb{E}[C_k] = np_k$ |

### Worked Example 8.3.1 — Three-Class Image Classifier

A classifier assigns images to three classes with probabilities:
$$p_{\text{cat}} = 0.5, \quad p_{\text{dog}} = 0.3, \quad p_{\text{bird}} = 0.2$$

**Verify normalization:** $0.5 + 0.3 + 0.2 = 1.0 \checkmark$

For a batch of $n = 100$ images, the expected counts are:
$$\mathbb{E}[C_{\text{cat}}] = 100 \times 0.5 = 50$$
$$\mathbb{E}[C_{\text{dog}}] = 100 \times 0.3 = 30$$
$$\mathbb{E}[C_{\text{bird}}] = 100 \times 0.2 = 20$$

**Multinomial probability of an exact count** — say, 50 cats, 30 dogs, 20 birds from 100 draws:

$$P(C_{\text{cat}}=50, C_{\text{dog}}=30, C_{\text{bird}}=20) = \frac{100!}{50!\,30!\,20!}(0.5)^{50}(0.3)^{30}(0.2)^{20}$$

This number is astronomically small (there are an enormous number of equally likely orderings). In practice we use the expected counts, not this exact probability.

### Worked Example 8.3.2 — Rolling a Die Twice

A fair die has $K = 6$, each $p_k = \frac{1}{6}$. Roll $n = 2$ times. What is $P(C_1 = 1, C_2 = 1, C_3 = 0, \ldots, C_6 = 0)$? (Exactly one 1 and one 2.)

$$P = \frac{2!}{1!\,1!\,0!\cdots 0!} \left(\frac{1}{6}\right)^1 \left(\frac{1}{6}\right)^1 (1)^0 \cdots (1)^0 = 2 \times \frac{1}{36} = \frac{1}{18}$$

**Verify:** The probability of rolling a 1 then a 2 is $\frac{1}{6} \times \frac{1}{6} = \frac{1}{36}$, and the same for rolling a 2 then a 1. Total: $\frac{2}{36} = \frac{1}{18}$. $\checkmark$

### Engineer's Angle

The Categorical distribution is the output model for **multi-class classification**. The softmax function (Section 8.7) converts raw model scores into a Categorical probability vector. The true label in multiclass classification is one-hot — a degenerate Categorical with all probability on one class.

```python
import math

def categorical_pmf(k, probs):
    """P(X = k) for Categorical(probs). k is 0-indexed."""
    return probs[k]

def multinomial_expected_counts(n, probs):
    """Expected count for each category in n trials."""
    return [n * p for p in probs]

# Three-class classifier
probs = [0.5, 0.3, 0.2]
classes = ['cat', 'dog', 'bird']

print("Categorical PMF:")
for i, cls in enumerate(classes):
    print(f"  P(X={cls}) = {categorical_pmf(i, probs)}")

print("\nExpected counts from 100 images:")
for cls, ec in zip(classes, multinomial_expected_counts(100, probs)):
    print(f"  E[C_{cls}] = {ec}")
# Verify: sum of probs == 1
print(f"\nSum of probs: {sum(probs)}")   # 1.0
```

---

## 8.4 Uniform Distribution

### Plain English First

The Uniform distribution is the "I have no information" distribution. Every outcome in a range is equally likely. There are two flavors: discrete (a finite set of equally likely outcomes, like a fair die) and continuous (any value in an interval is equally likely).

### Formal Notation

**Discrete Uniform** on $\{a, a+1, \ldots, b\}$ with $n = b - a + 1$ values:

$$P(X = k) = \frac{1}{n}, \qquad k = a, a+1, \ldots, b$$

$$\mathbb{E}[X] = \frac{a + b}{2}, \qquad \text{Var}(X) = \frac{n^2 - 1}{12}$$

**Continuous Uniform** on interval $[a, b]$:

$$f(x) = \frac{1}{b - a}, \qquad a \leq x \leq b, \qquad f(x) = 0 \text{ otherwise}$$

$$\mathbb{E}[X] = \frac{a + b}{2}, \qquad \text{Var}(X) = \frac{(b-a)^2}{12}$$

**Trust this result:** The PDF value $\frac{1}{b-a}$ is chosen so that the total area under the curve equals 1: $\int_a^b \frac{1}{b-a}\, dx = \frac{b-a}{b-a} = 1$.

| Quantity | Discrete $\{a,\ldots,b\}$ | Continuous $[a,b]$ |
|----------|--------------------------|---------------------|
| PMF / PDF | $1/n$ | $1/(b-a)$ |
| $\mathbb{E}[X]$ | $(a+b)/2$ | $(a+b)/2$ |
| $\text{Var}(X)$ | $(n^2-1)/12$ | $(b-a)^2/12$ |

### Worked Example 8.4.1 — Fair Six-Sided Die

$X \sim \text{DiscreteUniform}(1, 6)$, so $n = 6$.

$$\mathbb{E}[X] = \frac{1 + 6}{2} = \frac{7}{2} = 3.5$$

$$\text{Var}(X) = \frac{6^2 - 1}{12} = \frac{36 - 1}{12} = \frac{35}{12} \approx 2.917$$

**Verify against Chapter 7 Example 7.8.1:** We computed $\mathbb{E}[X] = 3.5$ and $\text{Var}(X) = \frac{35}{12}$ directly. $\checkmark$

### Worked Example 8.4.2 — Continuous Uniform on $[{-1}, 1]$

$X \sim \text{Uniform}(-1, 1)$.

$$\mathbb{E}[X] = \frac{-1 + 1}{2} = 0$$

$$\text{Var}(X) = \frac{(1 - (-1))^2}{12} = \frac{4}{12} = \frac{1}{3} \approx 0.333$$

**Verify the PDF integrates to 1:**
$$\int_{-1}^{1} \frac{1}{2}\, dx = \frac{1}{2} \times 2 = 1 \checkmark$$

**Compute a probability:** $P(-0.5 \leq X \leq 0.5)$:

$$P(-0.5 \leq X \leq 0.5) = \int_{-0.5}^{0.5} \frac{1}{2}\, dx = \frac{1}{2} \times (0.5 - (-0.5)) = \frac{1}{2} \times 1 = 0.5$$

Exactly half the interval has half the probability. $\checkmark$

### Engineer's Angle

Uniform distributions appear in two key places:

1. **Weight initialization** (Xavier / Glorot initialization): initial weights are drawn from $\text{Uniform}(-c, c)$ where $c = \sqrt{6 / (n_{\text{in}} + n_{\text{out}})}$. The Uniform assumption here is a maximum-entropy prior given a fixed range.

2. **Dropout masking**: each neuron is independently kept (1) or dropped (0) with equal probability — a discrete Uniform-like sampling per neuron. The sampling is Bernoulli, but the *decision* to use $p = 0.5$ often comes from a Uniform-random perspective.

```python
import math

def discrete_uniform_mean(a, b):
    return (a + b) / 2

def discrete_uniform_var(a, b):
    n = b - a + 1
    return (n**2 - 1) / 12

def continuous_uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def continuous_uniform_prob(x_lo, x_hi, a, b):
    """P(x_lo <= X <= x_hi) for X ~ Uniform(a, b)."""
    lo = max(x_lo, a)
    hi = min(x_hi, b)
    return max(0.0, (hi - lo) / (b - a))

# Fair die
print(f"DiscreteUniform(1,6): E={discrete_uniform_mean(1,6)}, Var={discrete_uniform_var(1,6):.6f}")
# DiscreteUniform(1,6): E=3.5, Var=2.916667

# Continuous uniform [-1, 1]
print(f"Uniform(-1,1): E={discrete_uniform_mean(-1,1)}")
print(f"Uniform(-1,1): Var={(2**2)/12:.6f}")
print(f"P(-0.5 <= X <= 0.5) = {continuous_uniform_prob(-0.5, 0.5, -1, 1)}")
# P(-0.5 <= X <= 0.5) = 0.5

# Xavier init range for a layer: n_in=512, n_out=256
n_in, n_out = 512, 256
c = math.sqrt(6 / (n_in + n_out))
print(f"\nXavier init: Uniform(-{c:.4f}, {c:.4f})")
```

---

## 8.5 Gaussian (Normal) Distribution

### Plain English First

The Gaussian (Normal) distribution is the most important distribution in all of statistics and machine learning. It describes the bell-shaped curve you've seen everywhere: test scores, measurement errors, heights, and — after appropriate scaling — the outputs of almost any averaging process. It is symmetric around its center, tails off rapidly in both directions, and is completely characterized by just two numbers: the mean (center) and variance (spread).

Why is it everywhere? Section 8.5.4 gives the answer (Central Limit Theorem). First, let's understand the shape.

### Formal Notation

$X \sim \mathcal{N}(\mu, \sigma^2)$ (read: "$X$ is Normal with mean $\mu$ and variance $\sigma^2$").

**PDF:**

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \qquad x \in (-\infty, +\infty)$$

**Expected value and variance:**
$$\mathbb{E}[X] = \mu, \qquad \text{Var}(X) = \sigma^2$$

**Trust this result:** $\displaystyle\int_{-\infty}^{+\infty} f(x)\, dx = 1$. This integral requires a clever polar-coordinate trick; the result is taken as given. What matters is understanding why the formula has this shape.

**Here's why** the bell curve looks the way it does: The term $(x - \mu)^2$ is zero at the center ($x = \mu$) and grows as you move away. Negating it and exponentiating gives a function that is 1 at the center and decays toward 0 symmetrically on both sides. The factor $\frac{1}{\sigma\sqrt{2\pi}}$ is a normalizing constant that scales the curve so the total area equals 1. The larger $\sigma$ is, the wider the bell must be to maintain unit area, so the peak at $x = \mu$ is lower.

**Key shape properties:**
- The peak of the PDF is at $x = \mu$ with height $\frac{1}{\sigma\sqrt{2\pi}}$.
- The PDF has inflection points at $x = \mu \pm \sigma$ (where the curve transitions from concave down to concave up).
- The tails decay extremely fast because of the $e^{-x^2}$ factor.

| Quantity | Value |
|----------|-------|
| Support | $(-\infty, +\infty)$ |
| Parameters | $\mu \in \mathbb{R}$ (mean), $\sigma^2 > 0$ (variance) |
| PDF | $\frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$ |
| $\mathbb{E}[X]$ | $\mu$ |
| $\text{Var}(X)$ | $\sigma^2$ |
| Peak height | $\frac{1}{\sigma\sqrt{2\pi}}$ |

### 8.5.1 The Standard Normal and Z-Scores

The **standard normal** is the special case $\mu = 0$, $\sigma^2 = 1$, written $Z \sim \mathcal{N}(0, 1)$.

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

The peak of the standard normal is at $z = 0$ with height $\frac{1}{\sqrt{2\pi}} \approx 0.3989$.

The **CDF** of the standard normal is:

$$\Phi(z) = P(Z \leq z) = \int_{-\infty}^{z} \phi(t)\, dt$$

This integral has no closed form in elementary functions, but it is related to the **error function** $\text{erf}$:

$$\Phi(z) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right]$$

**Converting any normal to standard:** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then:

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

The value $Z$ is called the **Z-score**: it measures how many standard deviations $X$ is above or below the mean.

$$P(X \leq x) = P\!\left(Z \leq \frac{x - \mu}{\sigma}\right) = \Phi\!\left(\frac{x - \mu}{\sigma}\right)$$

### 8.5.2 The 68-95-99.7 Rule

For any Gaussian, almost all probability mass lies within three standard deviations of the mean:

$$P(\mu - \sigma \leq X \leq \mu + \sigma) = P(-1 \leq Z \leq 1) = \text{erf}\!\left(\frac{1}{\sqrt{2}}\right) \approx 0.6827$$

$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) = P(-2 \leq Z \leq 2) = \text{erf}\!\left(\frac{2}{\sqrt{2}}\right) \approx 0.9545$$

$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) = P(-3 \leq Z \leq 3) = \text{erf}\!\left(\frac{3}{\sqrt{2}}\right) \approx 0.9973$$

**Verification using $\text{erf}$** (so this is not magic):

$$P(-1 \leq Z \leq 1) = \Phi(1) - \Phi(-1) = \frac{1}{2}\!\left[1 + \text{erf}\!\left(\frac{1}{\sqrt{2}}\right)\right] - \frac{1}{2}\!\left[1 + \text{erf}\!\left(\frac{-1}{\sqrt{2}}\right)\right]$$

Since $\text{erf}(-x) = -\text{erf}(x)$ (erf is an odd function):

$$= \frac{1}{2}\!\left[\text{erf}\!\left(\frac{1}{\sqrt{2}}\right) + \text{erf}\!\left(\frac{1}{\sqrt{2}}\right)\right] = \text{erf}\!\left(\frac{1}{\sqrt{2}}\right)$$

Numerically: $\text{erf}(1/\sqrt{2}) = \text{erf}(0.7071) \approx 0.6827$. $\checkmark$

### 8.5.3 Worked Examples

**Worked Example 8.5.1 — Exam Scores**

Exam scores follow $X \sim \mathcal{N}(\mu = 75, \sigma^2 = 64)$, so $\sigma = 8$.

(a) What fraction of students score below 83?

$$z = \frac{83 - 75}{8} = \frac{8}{8} = 1.0$$

$$P(X < 83) = \Phi(1.0) \approx 0.8413$$

About **84.1%** of students score below 83.

(b) What fraction score between 67 and 83?

$$z_{\text{lo}} = \frac{67 - 75}{8} = -1.0, \qquad z_{\text{hi}} = \frac{83 - 75}{8} = +1.0$$

$$P(67 < X < 83) = \Phi(1) - \Phi(-1) \approx 0.8413 - 0.1587 = 0.6827$$

About **68.3%** — exactly the 68-95-99.7 rule in action (one standard deviation band). $\checkmark$

(c) What fraction score above 91?

$$z = \frac{91 - 75}{8} = \frac{16}{8} = 2.0$$

$$P(X > 91) = 1 - \Phi(2.0) \approx 1 - 0.9772 = 0.0228$$

About **2.3%** — consistent with the 95.45% rule (leaving 4.55% outside two standard deviations, split equally on both tails gives about 2.28% on each side). $\checkmark$

**Worked Example 8.5.2 — PDF Value at a Point**

For $X \sim \mathcal{N}(170, 100)$ ($\mu = 170$, $\sigma = 10$), what is the PDF at $x = 185$?

$$z = \frac{185 - 170}{10} = 1.5$$

$$f(185) = \frac{1}{10\sqrt{2\pi}} e^{-1.5^2/2} = \frac{1}{10 \times 2.5066} e^{-1.125}$$

$$= \frac{1}{25.066} \times 0.3247 \approx \frac{0.3247}{25.066} \approx 0.01295$$

**Remember:** this is a *density*, not a probability. The probability of $X$ falling in a tiny interval $[185, 185.01]$ would be approximately $0.01295 \times 0.01 = 0.0001295$.

### 8.5.4 Why Does the Gaussian Appear Everywhere? (Central Limit Theorem)

Here is one of the most important theorems in all of mathematics, stated intuitively:

> **Central Limit Theorem (informal):** The average of many independent, identically distributed random variables — regardless of the distribution of those individual variables — converges to a Gaussian as the number of variables grows.

In other words: add up 30 uniform random variables, or 30 exponential random variables, or 30 Bernoulli random variables, and the sum will look Gaussian. This is why:

- **Measurement noise** is Gaussian: physical sensors aggregate many tiny independent error sources.
- **Aggregated prediction errors** look Gaussian: a model's total error is the sum of many small independent mistakes.
- **Feature means in large datasets** are Gaussian: by the CLT, the sample mean of any feature has an approximately Gaussian distribution when the sample is large.
- **Neural network activations** before nonlinearities often look Gaussian for similar reasons.

No proof is required here. The takeaway for engineers: *when a quantity is the result of summing or averaging many independent contributions, expect it to look Gaussian.*

### Engineer's Angle

The Gaussian is central to:

1. **Weight initialization**: Drawing weights from $\mathcal{N}(0, \sigma^2)$ (He init uses $\sigma^2 = 2/n_{\text{in}}$) keeps activation magnitudes stable through deep networks.
2. **VAE latent space**: The encoder outputs $\mu$ and $\sigma^2$, and the latent code $z \sim \mathcal{N}(\mu, \sigma^2)$ is sampled from this Gaussian — the KL divergence (Section 8.8) from this to the standard normal is the regularizer.
3. **Diffusion models**: The forward process corrupts data by adding Gaussian noise, step by step. The reverse (denoising) process learns to invert this.
4. **Gaussian noise layers**: Regularization by adding $\mathcal{N}(0, \sigma^2)$ noise to inputs.

```python
import math

def gaussian_pdf(x, mu, sigma):
    """PDF of N(mu, sigma^2) at x."""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

def phi(z):
    """CDF of standard normal N(0,1) at z."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def normal_cdf(x, mu, sigma):
    """CDF of N(mu, sigma^2) at x."""
    return phi((x - mu) / sigma)

def normal_prob_between(lo, hi, mu, sigma):
    """P(lo <= X <= hi) for X ~ N(mu, sigma^2)."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# Standard normal peak
print(f"Standard normal peak: {1/math.sqrt(2*math.pi):.6f}")  # 0.398942

# Exam score example: N(75, 64)
mu, sigma = 75, 8
print(f"\nExam scores ~ N(75, 64):")
print(f"  P(X < 83) = {normal_cdf(83, mu, sigma):.6f}")     # 0.841345
print(f"  P(67<X<83) = {normal_prob_between(67, 83, mu, sigma):.6f}")  # 0.682689
print(f"  P(X > 91) = {1 - normal_cdf(91, mu, sigma):.6f}") # 0.022750

# 68-95-99.7 rule
for k in [1, 2, 3]:
    prob = 2 * phi(k) - 1   # = phi(k) - phi(-k)
    print(f"  P(within {k} std devs) = {prob:.6f}  ({prob*100:.2f}%)")
# 0.682689, 0.954500, 0.997300

# He init: weights ~ N(0, 2/n_in)
n_in = 512
sigma_he = math.sqrt(2 / n_in)
print(f"\nHe init for n_in={n_in}: N(0, {2/n_in:.6f}), sigma={sigma_he:.6f}")
```

---

## 8.6 Exponential Distribution

### Plain English First

If you are waiting for something to happen — a server request, a hardware failure, a customer arriving — and the events occur at a constant average rate and independently of each other, then the time between events follows an Exponential distribution.

The defining characteristic of the Exponential distribution is the **memoryless property**: if you have been waiting for 5 minutes, the probability of waiting another 3 minutes is exactly the same as it would have been if you had just started waiting. The distribution "forgets" how long you have already waited.

### Formal Notation

$X \sim \text{Exponential}(\lambda)$ where $\lambda > 0$ is the **rate parameter** (events per unit time).

**PDF:**

$$f(x) = \lambda e^{-\lambda x}, \qquad x \geq 0$$

**CDF** (probability that the wait is at most $t$):

$$F(t) = P(X \leq t) = 1 - e^{-\lambda t}$$

**Expected value and variance:**

$$\mathbb{E}[X] = \frac{1}{\lambda}, \qquad \text{Var}(X) = \frac{1}{\lambda^2}$$

**Trust this result:** The mean $1/\lambda$ makes intuitive sense: if events arrive at rate $\lambda = 2$ per minute, you expect to wait $1/2$ a minute on average.

**Memoryless property:** For $s, t > 0$:

$$P(X > s + t \mid X > s) = P(X > t)$$

**Here's why** the memoryless property holds:

$$P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = \frac{e^{-\lambda s} e^{-\lambda t}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t) \quad \square$$

| Quantity | Value |
|----------|-------|
| Support | $[0, +\infty)$ |
| Parameter | $\lambda > 0$ (rate) |
| PDF | $\lambda e^{-\lambda x}$ |
| CDF | $1 - e^{-\lambda x}$ |
| $\mathbb{E}[X]$ | $1/\lambda$ |
| $\text{Var}(X)$ | $1/\lambda^2$ |
| Standard deviation | $1/\lambda$ (equals the mean!) |

### Worked Example 8.6.1 — Web Server Requests

Requests arrive at an average rate of $\lambda = 0.5$ requests per minute. The time between requests follows $X \sim \text{Exponential}(0.5)$.

**Expected wait time:**
$$\mathbb{E}[X] = \frac{1}{0.5} = 2 \text{ minutes}$$

**Step 1 — Probability of waiting at most 3 minutes:**

$$P(X \leq 3) = 1 - e^{-0.5 \times 3} = 1 - e^{-1.5} = 1 - 0.2231 = 0.7769$$

About **77.7%** of the time, the next request arrives within 3 minutes.

**Step 2 — Probability of waiting more than 5 minutes:**

$$P(X > 5) = e^{-0.5 \times 5} = e^{-2.5} \approx 0.0821$$

About **8.2%** of inter-arrival gaps exceed 5 minutes.

### Worked Example 8.6.2 — Memoryless Property in Action

Using the same server ($\lambda = 0.5$). You have already waited 2 minutes without a request. What is the probability you wait at least 1 more minute?

**Using the memoryless property directly:**
$$P(X > 3 \mid X > 2) = P(X > 1) = e^{-0.5 \times 1} = e^{-0.5} \approx 0.6065$$

**Verify by direct computation:**

$$P(X > 3 \mid X > 2) = \frac{P(X > 3)}{P(X > 2)} = \frac{e^{-1.5}}{e^{-1.0}} = \frac{0.2231}{0.3679} \approx 0.6065$$

Both methods give the same answer. $\checkmark$ The 2 minutes already elapsed are completely irrelevant — the exponential distribution has no memory.

### Engineer's Angle

The Exponential distribution appears in:

1. **Queuing models for inference latency**: If requests arrive at constant average rate and service times are Exponential, the system follows a Poisson-process / M/M/1 queue. Understanding tail latency (the 99th percentile wait) requires knowing the CDF of the Exponential.

2. **Training time budgets**: Time to see the first gradient improvement, time between loss spikes — these can often be modeled as Exponential for simple rough calculations.

3. **Survival analysis**: In time-to-event models (how long before a user churns?), the Exponential is the simplest baseline model. More sophisticated models use Weibull distributions, but Exponential is the starting point.

```python
import math

def exp_pdf(x, lam):
    """PDF of Exponential(lam) at x."""
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)

def exp_cdf(x, lam):
    """CDF of Exponential(lam): P(X <= x)."""
    if x < 0:
        return 0.0
    return 1.0 - math.exp(-lam * x)

def exp_survival(x, lam):
    """P(X > x) for Exponential(lam)."""
    if x < 0:
        return 1.0
    return math.exp(-lam * x)

lam = 0.5
print(f"Exponential(lambda=0.5):")
print(f"  E[X] = {1/lam} minutes")
print(f"  P(X <= 3) = {exp_cdf(3, lam):.6f}")    # 0.776870
print(f"  P(X > 5) = {exp_survival(5, lam):.6f}") # 0.082085

# Memoryless property verification
s, t = 2, 1
cond = exp_survival(s + t, lam) / exp_survival(s, lam)
direct = exp_survival(t, lam)
print(f"\nMemoryless: P(X>3|X>2) = {cond:.6f}")
print(f"            P(X>1)      = {direct:.6f}")
print(f"            Equal: {abs(cond - direct) < 1e-10}")  # True
```

---

## 8.7 Softmax as a Distribution

### Plain English First

A neural network's final classification layer produces a vector of raw scores — one per class — called **logits**. Logits can be any real numbers: negative, zero, positive, large, small. We need to turn them into probabilities that sum to 1. The **softmax function** does exactly this, and the result is a valid Categorical distribution.

The name "softmax" comes from the fact that it is a smooth, differentiable approximation of the "hardmax" (argmax) function. The class with the largest logit gets the most probability, but not all of it.

### Formal Notation

Given a vector of logits $\mathbf{z} = (z_1, z_2, \ldots, z_K)$, the softmax produces a probability vector $\boldsymbol{\pi}$ where:

$$\pi_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}, \qquad k = 1, 2, \ldots, K$$

**Proof that $\sum_k \pi_k = 1$:**

$$\sum_{k=1}^{K} \pi_k = \sum_{k=1}^{K} \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} = \frac{\sum_{k=1}^{K} e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} = 1 \quad \square$$

**Proof that softmax reduces to sigmoid for $K = 2$:**

With $K=2$, logits $(z_1, z_2)$:

$$\pi_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{z_2 - z_1}} = \sigma(z_1 - z_2)$$

where $\sigma(s) = \frac{1}{1+e^{-s}}$ is the sigmoid function. Binary logistic regression is the two-class special case of softmax classification. $\square$

**Temperature scaling:** A parameter $T > 0$ can be introduced:

$$\pi_k = \frac{e^{z_k/T}}{\sum_j e^{z_j/T}}$$

- $T \to 0$: distribution collapses to a one-hot on the argmax (hardmax).
- $T \to \infty$: distribution approaches Uniform (maximum uncertainty).

### Worked Example 8.7.1 — Three-Class Logits

Logits from a classifier: $\mathbf{z} = (2.0, 1.0, 0.5)$.

**Step 1 — Compute exponentials:**
$$e^{2.0} = 7.3891, \quad e^{1.0} = 2.7183, \quad e^{0.5} = 1.6487$$

**Step 2 — Sum:**
$$S = 7.3891 + 2.7183 + 1.6487 = 11.7561$$

**Step 3 — Divide:**
$$\pi_1 = \frac{7.3891}{11.7561} \approx 0.6285, \quad \pi_2 = \frac{2.7183}{11.7561} \approx 0.2312, \quad \pi_3 = \frac{1.6487}{11.7561} \approx 0.1402$$

**Step 4 — Verify:**
$$0.6285 + 0.2312 + 0.1402 = 0.9999 \approx 1.0000 \checkmark$$

The largest logit (2.0) gets the majority of the probability mass, but the smaller logits still get non-zero probability.

### Worked Example 8.7.2 — Sigmoid as Two-Class Softmax

Logits $\mathbf{z} = (1.5, 0.0)$ (two classes).

**Softmax:**
$$\pi_1 = \frac{e^{1.5}}{e^{1.5} + e^{0.0}} = \frac{4.4817}{4.4817 + 1.0000} = \frac{4.4817}{5.4817} \approx 0.8176$$

**Sigmoid:**
$$\sigma(1.5 - 0.0) = \sigma(1.5) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.2231} = \frac{1}{1.2231} \approx 0.8176$$

Same result. $\checkmark$ Binary classification with logistic regression is exactly two-class softmax with logits $(s, 0)$.

### Engineer's Angle

In PyTorch/TensorFlow, `nn.CrossEntropyLoss` (or `tf.keras.losses.CategoricalCrossentropy`) combines softmax and cross-entropy in a single, numerically stable operation. The trick is the **log-sum-exp** computation:

$$\log \pi_k = z_k - \log\!\left(\sum_j e^{z_j}\right)$$

Subtracting the maximum logit before exponentiating prevents overflow — a "numeric stabilization" trick.

```python
import math

def softmax(z):
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    z_max = max(z)
    exps = [math.exp(zi - z_max) for zi in z]
    total = sum(exps)
    return [e / total for e in exps]

def sigmoid(s):
    return 1.0 / (1.0 + math.exp(-s))

def log_softmax(z):
    """log(softmax(z)) — numerically stable."""
    z_max = max(z)
    log_sum = z_max + math.log(sum(math.exp(zi - z_max) for zi in z))
    return [zi - log_sum for zi in z]

# Three-class example
z = [2.0, 1.0, 0.5]
probs = softmax(z)
print(f"softmax({z}) = {[round(p, 6) for p in probs]}")
# [0.628532, 0.231224, 0.140244]
print(f"Sum = {sum(probs):.10f}")   # 1.0000000000

# Temperature scaling
def softmax_temp(z, T):
    return softmax([zi / T for zi in z])

print(f"\nTemperature T=0.1 (sharp): {[round(p,4) for p in softmax_temp(z, 0.1)]}")
print(f"Temperature T=1.0 (normal): {[round(p,4) for p in softmax_temp(z, 1.0)]}")
print(f"Temperature T=10  (flat):  {[round(p,4) for p in softmax_temp(z, 10)]}")

# Binary: softmax vs sigmoid
z2 = [1.5, 0.0]
sm2 = softmax(z2)
sig = sigmoid(z2[0] - z2[1])
print(f"\nTwo-class softmax[0] = {sm2[0]:.6f}")
print(f"sigmoid(1.5 - 0.0)   = {sig:.6f}")
print(f"Equal: {abs(sm2[0] - sig) < 1e-10}")   # True
```

---

## 8.8 KL Divergence — Measuring Distance Between Distributions

### Plain English First

Suppose you have two probability distributions: $P$ (the "truth") and $Q$ (your model's approximation). How different are they? Intuitively, they are "close" if they assign similar probabilities to similar events. KL divergence gives a precise numerical measure of how much information you lose by using $Q$ to represent $P$.

There are two important warnings about KL divergence:

1. It is **not a distance** in the mathematical sense, because $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ in general. Direction matters.
2. It is always **non-negative**: $D_{KL}(P \| Q) \geq 0$, with equality if and only if $P = Q$ everywhere.

### Formal Notation

**Discrete case** ($P$ and $Q$ are distributions over the same finite set):

$$D_{KL}(P \| Q) = \sum_{k} P(k) \log \frac{P(k)}{Q(k)}$$

**Continuous case:**

$$D_{KL}(P \| Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)}\, dx$$

**Convention:** We use natural logarithm (base $e$) throughout, giving units of **nats**. (Using log base 2 gives bits; the choice is cosmetic.)

**Convention:** If $P(k) = 0$, the $k$-th term contributes 0 (by convention, $0 \log 0 = 0$). If $P(k) > 0$ but $Q(k) = 0$, the divergence is $+\infty$ — you have assigned zero probability to something that can actually happen.

**Trust this result:** $D_{KL}(P \| Q) \geq 0$, with equality iff $P = Q$ everywhere. This follows from Jensen's inequality applied to the convex function $-\log$.

### Worked Example 8.8.1 — KL Divergence and Its Asymmetry

Let $P = (0.9, 0.1)$ and $Q = (0.5, 0.5)$ over two outcomes.

**Compute $D_{KL}(P \| Q)$:** (how much $Q$ approximates $P$)

$$D_{KL}(P \| Q) = P(1)\log\frac{P(1)}{Q(1)} + P(2)\log\frac{P(2)}{Q(2)}$$

$$= 0.9 \times \log\frac{0.9}{0.5} + 0.1 \times \log\frac{0.1}{0.5}$$

$$= 0.9 \times \log(1.8) + 0.1 \times \log(0.2)$$

Using $\log(1.8) = 0.5878$ and $\log(0.2) = -1.6094$:

$$= 0.9 \times 0.5878 + 0.1 \times (-1.6094) = 0.5290 - 0.1609 = 0.3681 \text{ nats}$$

**Compute $D_{KL}(Q \| P)$:** (the reverse direction)

$$D_{KL}(Q \| P) = 0.5 \times \log\frac{0.5}{0.9} + 0.5 \times \log\frac{0.5}{0.1}$$

$$= 0.5 \times \log(0.5556) + 0.5 \times \log(5.0)$$

Using $\log(0.5556) = -0.5878$ and $\log(5.0) = 1.6094$:

$$= 0.5 \times (-0.5878) + 0.5 \times 1.6094 = -0.2939 + 0.8047 = 0.5108 \text{ nats}$$

**Conclusion:** $D_{KL}(P \| Q) = 0.3681 \neq 0.5108 = D_{KL}(Q \| P)$. KL divergence is asymmetric. $\checkmark$

The two directions penalize different mistakes:
- $D_{KL}(P \| Q)$: penalizes assigning low probability under $Q$ to events that are common under $P$ (it "forces" $Q$ to cover the support of $P$).
- $D_{KL}(Q \| P)$: penalizes $Q$ for spreading probability where $P$ is concentrated (it "forces" $Q$ to be cautious).

### Worked Example 8.8.2 — Zero KL When $P = Q$

For $P = Q = (0.6, 0.4)$:

$$D_{KL}(P \| Q) = 0.6 \times \log\frac{0.6}{0.6} + 0.4 \times \log\frac{0.4}{0.4} = 0.6 \times \log(1) + 0.4 \times \log(1) = 0 \checkmark$$

### Engineer's Angle

KL divergence is the workhorse of probabilistic ML:

1. **VAE regularizer**: The encoder outputs a Gaussian $q(z|x) = \mathcal{N}(\mu_\phi, \sigma^2_\phi)$. The KL term $D_{KL}(q(z|x) \| p(z))$ where $p(z) = \mathcal{N}(0, 1)$ penalizes the latent distribution for deviating from the prior. For Gaussians this has a closed form: $\frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$.

2. **Knowledge distillation**: Training a small student model to match the softmax outputs of a large teacher model uses $D_{KL}(\text{teacher} \| \text{student})$.

3. **RL policy optimization**: Many RL algorithms (PPO, TRPO) constrain the KL divergence between the old and new policy to prevent destructive policy updates.

```python
import math

def kl_divergence(P, Q):
    """
    KL(P || Q) for discrete distributions.
    P and Q are lists of probabilities over the same support.
    Uses natural log (nats).
    """
    total = 0.0
    for p, q in zip(P, Q):
        if p == 0:
            continue        # 0 * log(0/q) = 0 by convention
        if q == 0:
            return float('inf')  # cannot assign 0 where P > 0
        total += p * math.log(p / q)
    return total

P = [0.9, 0.1]
Q = [0.5, 0.5]

kl_pq = kl_divergence(P, Q)
kl_qp = kl_divergence(Q, P)

print(f"KL(P || Q) = {kl_pq:.6f} nats")   # 0.368064
print(f"KL(Q || P) = {kl_qp:.6f} nats")   # 0.510826
print(f"Asymmetric: {abs(kl_pq - kl_qp) > 0.01}")  # True

# Verify zero when P == Q
P_eq = [0.6, 0.4]
Q_eq = [0.6, 0.4]
print(f"\nKL(P || P) = {kl_divergence(P_eq, Q_eq):.10f}")  # 0.0000000000

# KL from standard Gaussian approximation (VAE term, scalar case)
def kl_gaussian_standard(mu, sigma_sq):
    """KL(N(mu, sigma^2) || N(0,1)) — closed form."""
    return 0.5 * (mu**2 + sigma_sq - math.log(sigma_sq) - 1)

print(f"\nVAE KL: N(0.5, 1.0) vs N(0,1) = {kl_gaussian_standard(0.5, 1.0):.6f}")
print(f"VAE KL: N(0.0, 1.0) vs N(0,1) = {kl_gaussian_standard(0.0, 1.0):.6f}")  # 0
```

---

## 8.9 Entropy

### Plain English First

Before we can fully explain cross-entropy, we need one more concept: **entropy**. Entropy measures the average uncertainty (or information content) of a distribution. A distribution that is always the same (a coin that always lands heads) has zero entropy — no uncertainty. A fair coin has maximum entropy — you never know what comes next.

### Formal Notation

For a discrete distribution $P$ over $K$ outcomes:

$$H(P) = -\sum_{k=1}^{K} P(k) \log P(k)$$

Using natural log, the unit is **nats**. Using log base 2, the unit is **bits**.

**Key properties:**
- $H(P) \geq 0$ always.
- $H(P) = 0$ iff $P$ is concentrated on a single outcome (certain outcome).
- $H(P)$ is maximized when $P$ is Uniform.

### Worked Example 8.9.1

For $P = (0.9, 0.1)$:
$$H(P) = -(0.9 \log 0.9 + 0.1 \log 0.1)$$
$$= -(0.9 \times (-0.1054) + 0.1 \times (-2.3026))$$
$$= -((-0.0948) + (-0.2303))$$
$$= -(-0.3251) = 0.3251 \text{ nats}$$

For $P = (0.5, 0.5)$ (maximum entropy for two outcomes):
$$H(P) = -(0.5 \log 0.5 + 0.5 \log 0.5) = -(2 \times 0.5 \times (-0.6931)) = 0.6931 \text{ nats}$$

The uniform distribution has higher entropy. $\checkmark$

---

## 8.10 Cross-Entropy and Its Connection to KL Divergence

### Plain English First

Cross-entropy answers the question: "What is the average number of nats needed to encode samples from $P$, if we design our encoding scheme assuming $Q$ is the true distribution?" If $Q = P$, the answer is just the entropy $H(P)$. If $Q \neq P$, we waste some bits — that waste is exactly the KL divergence.

In machine learning, $P$ is the true label distribution (often a one-hot vector) and $Q$ is the model's predicted distribution (the softmax output). Cross-entropy is the loss function that training minimizes.

### Formal Notation

**Cross-entropy** between $P$ (true) and $Q$ (model):

$$H(P, Q) = -\sum_{k} P(k) \log Q(k)$$

**Key identity:** Cross-entropy decomposes into entropy plus KL divergence:

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

**Here's why:**

$$H(P, Q) = -\sum_k P(k) \log Q(k)$$

Add and subtract $\sum_k P(k) \log P(k)$:

$$= -\sum_k P(k) \log P(k) + \sum_k P(k) \log P(k) - \sum_k P(k) \log Q(k)$$

$$= H(P) + \sum_k P(k) \log \frac{P(k)}{Q(k)}$$

$$= H(P) + D_{KL}(P \| Q) \quad \square$$

**Consequence for training:** Since $H(P)$ is fixed (the true label distribution does not depend on the model), minimizing cross-entropy over model parameters $\theta$ is equivalent to minimizing the KL divergence $D_{KL}(P \| Q_\theta)$:

$$\arg\min_\theta H(P, Q_\theta) = \arg\min_\theta D_{KL}(P \| Q_\theta)$$

This is why **cross-entropy loss and KL-based likelihood maximization are the same thing**.

### Worked Example 8.10.1 — Multiclass Cross-Entropy Loss

A three-class classifier receives an image. The true label is "cat" (class 0), so the true distribution is the one-hot vector $P = (1, 0, 0)$.

The model predicts $Q = (0.7, 0.2, 0.1)$.

**Cross-entropy loss:**

$$H(P, Q) = -(1 \times \log 0.7 + 0 \times \log 0.2 + 0 \times \log 0.1)$$
$$= -\log 0.7 = -(-0.3567) = 0.3567 \text{ nats}$$

**Here's why** the zero terms vanish: $P(k) = 0$ for all classes except the true class, so those terms contribute zero to the sum. The cross-entropy loss for one-hot $P$ reduces to:

$$H(P, Q) = -\log Q(\text{true class})$$

This is the **negative log-likelihood of the true class**. Training pushes $Q(\text{true class})$ toward 1, which pushes the loss toward 0.

**Verify the decomposition:**

$$H(P) = -(1\log 1 + 0 + 0) = 0 \quad \text{(one-hot has zero entropy)}$$

$$D_{KL}(P \| Q) = 1 \times \log\frac{1}{0.7} = \log(1/0.7) = -\log(0.7) = 0.3567$$

$$H(P) + D_{KL}(P \| Q) = 0 + 0.3567 = 0.3567 = H(P, Q) \checkmark$$

When the true distribution is one-hot, cross-entropy equals KL divergence (since $H(P)=0$).

### Worked Example 8.10.2 — Binary Cross-Entropy

For binary classification ($K = 2$), the true label $y \in \{0, 1\}$ gives a one-hot $P$. The model predicts $\hat{y} \in (0, 1)$, so $Q = (\hat{y}, 1-\hat{y})$.

The cross-entropy reduces to the familiar formula:

$$H(P, Q) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

**Case 1:** $y = 1$, $\hat{y} = 0.8$ (correct and confident):
$$\mathcal{L} = -\log(0.8) = 0.2231 \text{ nats}$$

**Case 2:** $y = 1$, $\hat{y} = 0.3$ (wrong and confident):
$$\mathcal{L} = -\log(0.3) = 1.2040 \text{ nats}$$

The loss is much larger when the model is confidently wrong. $\checkmark$ This is the loss from Section 8.1 — we now know exactly where it comes from: it is the cross-entropy between the one-hot true distribution and the Bernoulli model output.

### Engineer's Angle

```python
import math

def entropy(P):
    """Shannon entropy H(P) in nats."""
    return -sum(p * math.log(p) for p in P if p > 0)

def cross_entropy(P, Q):
    """Cross-entropy H(P, Q) in nats. P is true, Q is model."""
    return -sum(p * math.log(q) for p, q in zip(P, Q) if p > 0)

def kl_divergence(P, Q):
    return sum(p * math.log(p / q) for p, q in zip(P, Q) if p > 0)

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy for one sample."""
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

# ─── One-hot example ───────────────────────────────────────────────────────
P_true = [1.0, 0.0, 0.0]   # true class is 0
Q_pred = [0.7, 0.2, 0.1]   # model output

H_P  = entropy(P_true)                # 0.0 (one-hot has zero entropy)
KL   = kl_divergence(P_true, Q_pred)  # 0.356675
H_PQ = cross_entropy(P_true, Q_pred)  # 0.356675

print(f"H(P)        = {H_P:.6f}")
print(f"KL(P||Q)    = {KL:.6f}")
print(f"H(P,Q)      = {H_PQ:.6f}")
print(f"H(P)+KL = H(P,Q): {abs(H_P + KL - H_PQ) < 1e-10}")  # True

# ─── Binary cross-entropy ──────────────────────────────────────────────────
print(f"\nBCE(y=1, yhat=0.8) = {binary_cross_entropy(1, 0.8):.6f}")  # 0.223144
print(f"BCE(y=1, yhat=0.3) = {binary_cross_entropy(1, 0.3):.6f}")  # 1.203973

# ─── Training goal: minimizing CE minimizes KL ─────────────────────────────
# As model improves (Q gets closer to P), CE decreases
Q_better  = [0.9, 0.07, 0.03]
Q_perfect = [1.0 - 1e-9, 5e-10, 5e-10]   # near-perfect (avoid log(0))
print(f"\nCE with Q=[0.7,0.2,0.1]: {cross_entropy(P_true, Q_pred):.6f}")
print(f"CE with Q=[0.9,0.07,0.03]: {cross_entropy(P_true, Q_better):.6f}")
print(f"CE with Q≈[1,0,0]:         {cross_entropy(P_true, Q_perfect):.6f}")
# Loss approaches 0 as model approaches certainty on the correct class
```

---

## 8.11 Summary Table

| Distribution | Type | Parameters | $\mathbb{E}[X]$ | $\text{Var}(X)$ | ML Connection |
|---|---|---|---|---|---|
| $\text{Bernoulli}(p)$ | Discrete | $p \in [0,1]$ | $p$ | $p(1-p)$ | Binary classifier output, logistic regression |
| $\text{Binomial}(n, p)$ | Discrete | $n \in \mathbb{N},\, p \in [0,1]$ | $np$ | $np(1-p)$ | Batch accuracy, A/B test counts |
| $\text{Categorical}(\mathbf{p})$ | Discrete | $\mathbf{p}$: probability vector | N/A | N/A | Multi-class label, softmax output |
| $\text{Multinomial}(n, \mathbf{p})$ | Discrete | $n \in \mathbb{N},\, \mathbf{p}$: prob. vector | $np_k$ per class | $np_k(1-p_k)$ | Class count distribution |
| $\text{DiscreteUniform}(a, b)$ | Discrete | $a \leq b$ integers | $(a+b)/2$ | $(n^2-1)/12$ | Fair sampling, baseline model |
| $\text{Uniform}(a, b)$ | Continuous | $a < b$ reals | $(a+b)/2$ | $(b-a)^2/12$ | Xavier init, dropout sampling |
| $\mathcal{N}(\mu, \sigma^2)$ | Continuous | $\mu \in \mathbb{R},\, \sigma^2 > 0$ | $\mu$ | $\sigma^2$ | Weight init, VAE latent, diffusion noise |
| $\text{Exponential}(\lambda)$ | Continuous | $\lambda > 0$ | $1/\lambda$ | $1/\lambda^2$ | Inter-arrival times, latency modeling |
| Softmax output | Continuous | Logits $\mathbf{z} \in \mathbb{R}^K$ | — | — | Final layer of classifier |
| KL Divergence | Measure | Two distributions $P$, $Q$ | — | — | VAE regularizer, distillation |
| Cross-Entropy | Loss | True $P$, predicted $Q$ | — | — | Classification loss function |

---

## 8.12 Exercises

### Exercise 8.1 [Easy] — Bernoulli and Binomial

A model predicts "positive" with probability $p = 0.6$ on any given input, independently.

(a) Write the PMF of a single prediction $X \sim \text{Bernoulli}(0.6)$.

(b) In a batch of $n = 5$ inputs, let $Y$ = number of positive predictions. Compute $P(Y = 3)$ from first principles (show the binomial coefficient computation).

(c) Compute $\mathbb{E}[Y]$ and $\text{Var}(Y)$.

(d) Compute $P(Y \geq 4) = P(Y = 4) + P(Y = 5)$.

**Solution:**

(a) $P(X = 1) = 0.6$, $P(X = 0) = 0.4$. Compact form: $P(X = k) = (0.6)^k (0.4)^{1-k}$ for $k \in \{0, 1\}$.

(b) $Y \sim \text{Binomial}(5,\, 0.6)$.

$$\binom{5}{3} = \frac{5!}{3!\,2!} = \frac{5 \times 4}{2 \times 1} = 10$$

$$P(Y = 3) = \binom{5}{3}(0.6)^3(0.4)^2 = 10 \times 0.216 \times 0.16 = 10 \times 0.03456 = 0.3456$$

(c)

$$\mathbb{E}[Y] = 5 \times 0.6 = 3.0$$

$$\text{Var}(Y) = 5 \times 0.6 \times 0.4 = 1.2$$

(d)

$$P(Y = 4) = \binom{5}{4}(0.6)^4(0.4)^1 = 5 \times 0.1296 \times 0.4 = 5 \times 0.05184 = 0.2592$$

$$P(Y = 5) = \binom{5}{5}(0.6)^5(0.4)^0 = 1 \times 0.07776 \times 1 = 0.07776$$

$$P(Y \geq 4) = 0.2592 + 0.07776 = 0.33696$$

**Verify:** Sum all probabilities.
$$P(Y=0) = (0.4)^5 = 0.01024$$
$$P(Y=1) = 5(0.6)(0.4)^4 = 5 \times 0.6 \times 0.0256 = 0.07680$$
$$P(Y=2) = 10(0.6)^2(0.4)^3 = 10 \times 0.36 \times 0.064 = 0.23040$$
$$P(Y=3) = 0.34560$$
$$P(Y=4) = 0.25920$$
$$P(Y=5) = 0.07776$$
$$\text{Total} = 0.01024 + 0.07680 + 0.23040 + 0.34560 + 0.25920 + 0.07776 = 1.00000 \checkmark$$

---

### Exercise 8.2 [Easy] — Gaussian Z-Scores

The latency of a web service (in milliseconds) is modeled as $X \sim \mathcal{N}(\mu = 200, \sigma^2 = 900)$, so $\sigma = 30$.

(a) Compute the Z-score for $x = 260$ ms and $x = 170$ ms.

(b) Use the 68-95-99.7 rule to answer: what fraction of requests have latency between 140 ms and 260 ms?

(c) A response takes longer than 290 ms. Approximately what fraction of requests are this slow? (Use the 68-95-99.7 rule.)

**Solution:**

(a)

$$z_{260} = \frac{260 - 200}{30} = \frac{60}{30} = 2.0$$

$$z_{170} = \frac{170 - 200}{30} = \frac{-30}{30} = -1.0$$

(b) We need $P(140 \leq X \leq 260)$.

$$z_{\text{lo}} = \frac{140 - 200}{30} = -2.0, \qquad z_{\text{hi}} = \frac{260 - 200}{30} = +2.0$$

This is the $\pm 2\sigma$ band: $P(-2 \leq Z \leq 2) \approx 0.9545$. About **95.5%** of requests fall in this range.

(c) $z = \frac{290 - 200}{30} = 3.0$. The 99.7% rule says $P(-3 \leq Z \leq 3) \approx 0.9973$, leaving $1 - 0.9973 = 0.0027$ outside. By symmetry, the upper tail $P(Z > 3) \approx 0.0027/2 \approx 0.00135$.

About **0.135%** of requests exceed 290 ms. Only about 1 in 740 requests.

---

### Exercise 8.3 [Medium] — Exponential Memoryless Property

A GPU job scheduler receives jobs at a Poisson rate with average inter-arrival time of 4 minutes, so inter-arrival times follow $X \sim \text{Exponential}(\lambda = 0.25)$.

(a) Verify that $\mathbb{E}[X] = 4$ minutes.

(b) Compute $P(X \leq 5)$ — the probability the next job arrives within 5 minutes. Show full arithmetic.

(c) You have already been waiting 3 minutes with no job. Using the memoryless property, what is the probability you wait at least 2 more minutes?

(d) Verify part (c) by computing the conditional probability directly.

**Solution:**

(a) $\mathbb{E}[X] = \frac{1}{\lambda} = \frac{1}{0.25} = 4$ minutes. $\checkmark$

(b)

$$P(X \leq 5) = 1 - e^{-0.25 \times 5} = 1 - e^{-1.25} = 1 - 0.2865 = 0.7135$$

(Check: $e^{-1.25} = e^{-1} \times e^{-0.25} \approx 0.3679 \times 0.7788 = 0.2865$.)

About **71.4%** of the time the next job arrives within 5 minutes.

(c) By the memoryless property:

$$P(X > 3 + 2 \mid X > 3) = P(X > 2) = e^{-0.25 \times 2} = e^{-0.5} \approx 0.6065$$

The 3 minutes already elapsed are irrelevant. There is about a **60.7%** chance of waiting at least 2 more minutes.

(d) Direct computation:

$$P(X > 5 \mid X > 3) = \frac{P(X > 5)}{P(X > 3)} = \frac{e^{-0.25 \times 5}}{e^{-0.25 \times 3}} = \frac{e^{-1.25}}{e^{-0.75}} = e^{-1.25 + 0.75} = e^{-0.5} \approx 0.6065 \checkmark$$

Both approaches give the same answer. $\checkmark$

---

### Exercise 8.4 [Medium] — Softmax and Temperature

A three-class model produces logits $\mathbf{z} = (3.0,\, 1.0,\, 0.0)$.

(a) Compute the softmax probabilities. Show all arithmetic.

(b) A two-class version has logits $(3.0, 1.0)$. Verify that $\text{softmax}(\mathbf{z})_1 = \sigma(z_1 - z_2)$.

(c) What happens to the distribution at temperature $T = 2.0$? Compute the new softmax.

(d) As $T \to \infty$, what does the distribution approach?

**Solution:**

(a)

$$e^{3.0} = 20.0855, \quad e^{1.0} = 2.7183, \quad e^{0.0} = 1.0000$$

$$S = 20.0855 + 2.7183 + 1.0000 = 23.8038$$

$$\pi_1 = \frac{20.0855}{23.8038} \approx 0.8437, \quad \pi_2 = \frac{2.7183}{23.8038} \approx 0.1142, \quad \pi_3 = \frac{1.0000}{23.8038} \approx 0.0420$$

**Verify:** $0.8437 + 0.1142 + 0.0420 = 0.9999 \approx 1.0000$ $\checkmark$

(b) Two-class logits $(3.0, 1.0)$:

$$\text{softmax}_1 = \frac{e^{3.0}}{e^{3.0} + e^{1.0}} = \frac{20.0855}{20.0855 + 2.7183} = \frac{20.0855}{22.8038} \approx 0.8808$$

$$\sigma(3.0 - 1.0) = \sigma(2.0) = \frac{1}{1 + e^{-2.0}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} \approx 0.8808 \checkmark$$

(c) At $T = 2.0$, divide logits by $T$: scaled logits $= (1.5,\, 0.5,\, 0.0)$.

$$e^{1.5} = 4.4817, \quad e^{0.5} = 1.6487, \quad e^{0.0} = 1.0000$$

$$S = 4.4817 + 1.6487 + 1.0000 = 7.1304$$

$$\pi_1 = \frac{4.4817}{7.1304} \approx 0.6286, \quad \pi_2 = \frac{1.6487}{7.1304} \approx 0.2312, \quad \pi_3 = \frac{1.0000}{7.1304} \approx 0.1402$$

The distribution is flatter (less confident) at $T=2$ compared to $T=1$.

(d) As $T \to \infty$, all logits divided by $T$ approach $0$. Then all exponentials approach $e^0 = 1$, and the distribution approaches $\pi_k = \frac{1}{K} = \frac{1}{3}$ for all $k$ — a uniform distribution. At very high temperature, the model is maximally uncertain.

---

### Exercise 8.5 [Hard] — KL Divergence and Cross-Entropy in Classifier Training

A binary classifier with sigmoid output is trained on a dataset with balanced classes. The true distribution over labels is $P = (0.5, 0.5)$ (50% positive, 50% negative). The model currently predicts $Q_1 = (0.8, 0.2)$ (overconfident on positive class).

(a) Compute $H(P)$, $H(P, Q_1)$, and $D_{KL}(P \| Q_1)$. Verify the identity $H(P, Q_1) = H(P) + D_{KL}(P \| Q_1)$.

(b) After training improves the model to $Q_2 = (0.55, 0.45)$, compute $H(P, Q_2)$ and $D_{KL}(P \| Q_2)$. Which direction did both quantities move?

(c) If the model reaches $Q_3 = P = (0.5, 0.5)$, what is the minimum achievable cross-entropy loss? What does this correspond to?

(d) A different model predicts $Q_4 = (0.5, 0.5)$ from the start. Compute $H(P, Q_4)$ and explain why this is the theoretical minimum for this problem.

**Solution:**

(a)

**Entropy of $P$:**
$$H(P) = -(0.5 \log 0.5 + 0.5 \log 0.5) = -(2 \times 0.5 \times (-0.6931)) = 0.6931 \text{ nats}$$

**Cross-entropy $H(P, Q_1)$ with $Q_1 = (0.8, 0.2)$:**
$$H(P, Q_1) = -(0.5 \log 0.8 + 0.5 \log 0.2)$$
$$= -(0.5 \times (-0.2231) + 0.5 \times (-1.6094))$$
$$= -((-0.1116) + (-0.8047))$$
$$= 0.9163 \text{ nats}$$

**KL divergence:**
$$D_{KL}(P \| Q_1) = 0.5 \log\frac{0.5}{0.8} + 0.5 \log\frac{0.5}{0.2}$$
$$= 0.5 \times \log(0.625) + 0.5 \times \log(2.5)$$
$$= 0.5 \times (-0.4700) + 0.5 \times 0.9163$$
$$= -0.2350 + 0.4581 = 0.2231 \text{ nats}$$

**Verify identity:**
$$H(P) + D_{KL}(P \| Q_1) = 0.6931 + 0.2231 = 0.9162 \approx 0.9163 \checkmark$$

(Small rounding difference in the last digit from truncated intermediate values.)

(b) **Cross-entropy with $Q_2 = (0.55, 0.45)$:**
$$H(P, Q_2) = -(0.5 \log 0.55 + 0.5 \log 0.45)$$
$$= -(0.5 \times (-0.5978) + 0.5 \times (-0.7985))$$
$$= -((-0.2989) + (-0.3993))$$
$$= 0.6982 \text{ nats}$$

**KL divergence:**
$$D_{KL}(P \| Q_2) = H(P, Q_2) - H(P) = 0.6982 - 0.6931 = 0.0050 \text{ nats}$$

Both $H(P, Q_2) = 0.6982 < H(P, Q_1) = 0.9163$ and $D_{KL}(P \| Q_2) = 0.0050 < D_{KL}(P \| Q_1) = 0.2231$. The model improved: both cross-entropy and KL divergence decreased.

(c) If $Q_3 = P = (0.5, 0.5)$:
$$H(P, Q_3) = H(P) + D_{KL}(P \| Q_3) = H(P) + 0 = H(P) = 0.6931 \text{ nats}$$

The minimum achievable cross-entropy is the entropy of the true label distribution. This corresponds to a perfectly calibrated model: it cannot be improved further without changing the true distribution (i.e., getting better data).

(d) $H(P, Q_4) = H(P, (0.5, 0.5)) = H(P) = 0.6931$ nats, same as (c). This is the theoretical minimum because the model perfectly matches the true distribution $P = (0.5, 0.5)$.

In practice, the true label distribution $P$ is unknown. We approximate $H(P)$ using the empirical cross-entropy on the training set. The irreducible lower bound $H(P)$ represents the inherent uncertainty in the labeling problem — no model, no matter how large, can do better.

---

*Next: Chapter 9 — Information Theory: Entropy, Mutual Information, and their role in feature selection and generative models.*
