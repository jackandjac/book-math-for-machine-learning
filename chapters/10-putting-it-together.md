# Chapter 10: Putting It All Together — Math Inside a Neural Network

> *"A neural network is not magic. It is matrix multiplication, followed by a nonlinearity, followed by calculus, followed by probability. You know all of that now."*

---

## 10.0 What This Chapter Does

You have spent nine chapters building four mathematical toolkits:

| Toolkit | Chapters | Core ideas |
|---------|----------|-----------|
| **Linear Algebra** | 2–4 | Vectors, matrices, dot products, SVD, PCA |
| **Calculus** | 5–6 | Derivatives, chain rule, gradient descent, Adam |
| **Probability** | 7–8 | Bayes, Gaussian, KL divergence, cross-entropy, softmax |
| **Statistics** | 9 | MLE, bias-variance tradeoff, regularization, cross-validation |

This chapter shows how those four toolkits assemble into a single working machine: a **neural network**. Not conceptually — concretely. We will derive every equation from first principles, then implement a 2-layer network from scratch in pure Python (no numpy, no PyTorch) and watch it solve the XOR problem.

Every equation will be tagged with the chapter that introduced it. By the end, you should be able to look at any line of a real training loop and say, *"That's the chain rule from Ch 5"* or *"That's MLE from Ch 9."*

---

## 10.1 The Big Picture — How the Four Pillars Combine

Here is the complete flow through a neural network:

```
DATA
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  FORWARD PASS                            (Linear Algebra) │
│                                                           │
│  Input x ──[W1, b1]──► z1 = W1 x + b1                   │
│              ↓                                            │
│         [activation] ──► h = ReLU(z1)                    │
│              ↓                                            │
│         [W2, b2] ──────► z2 = W2 h + b2                  │
│              ↓                                            │
│         [softmax] ──────► p = softmax(z2)   (Probability) │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  LOSS FUNCTION                    (Probability+Statistics) │
│                                                           │
│  L = -Σ y_k log(p_k)   ← cross-entropy from MLE         │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  BACKWARD PASS                               (Calculus)   │
│                                                           │
│  ∂L/∂W2 via chain rule                                   │
│  ∂L/∂W1 via chain rule (through 2 layers)                │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  WEIGHT UPDATE                               (Calculus)   │
│                                                           │
│  W ← W - α ∇_W L     ← gradient descent / Adam          │
└──────────────────────────────────────────────────────────┘
  │
  └──► repeat until convergence
```

This loop — forward pass, loss, backward pass, update — is **all** that happens during training. The math from this book tells us exactly what to compute at every step.

---

## 10.2 The Forward Pass — Computing a Prediction

### 10.2.1 The Linear Layer: $\mathbf{h} = W\mathbf{x} + \mathbf{b}$

The fundamental operation in every neural network layer is a **linear transformation** — a matrix-vector product followed by adding a bias vector.

For a layer with $n_{\text{in}}$ input features and $n_{\text{out}}$ neurons:

$$\mathbf{z} = W\mathbf{x} + \mathbf{b} \quad \text{(Ch 3 — matrix-vector product)}$$

where:
- $\mathbf{x} \in \mathbb{R}^{n_{\text{in}}}$ is the input vector
- $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$ is the **weight matrix** — each row is the weights for one neuron
- $\mathbf{b} \in \mathbb{R}^{n_{\text{out}}}$ is the **bias vector**
- $\mathbf{z} \in \mathbb{R}^{n_{\text{out}}}$ is the **pre-activation** output

**In component form (from Ch 3):**

$$z_j = \sum_{i=1}^{n_{\text{in}}} W_{ji}\, x_i + b_j, \quad j = 1, \ldots, n_{\text{out}}$$

This is the dot product of the $j$-th row of $W$ with $\mathbf{x}$, plus a bias. It computes a weighted sum of the inputs for neuron $j$.

**Why matrices?** A layer with $n_{\text{out}}$ neurons, each with $n_{\text{in}}$ inputs, needs $n_{\text{out}} \times n_{\text{in}}$ weight values. Packing them into a matrix $W$ lets us compute all neurons simultaneously with a single matrix-vector multiply — exactly the operation we studied in Chapter 3.

**Concrete example for our 2→4 network:**

$$W_1 = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \\ w_{41} & w_{42} \end{bmatrix} \in \mathbb{R}^{4 \times 2}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \quad \mathbf{b}_1 \in \mathbb{R}^4$$

$$\mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1 \in \mathbb{R}^4 \quad \text{(Ch 3)}$$

---

### 10.2.2 ReLU Activation — Why We Need Nonlinearity

After the linear layer, we apply an **activation function**. Without it, the whole network would just be a single big linear transformation — it could only learn linear boundaries. Activation functions introduce nonlinearity, giving the network expressive power.

The most common activation is **ReLU** (Rectified Linear Unit):

$$\text{ReLU}(z) = \max(0, z) \quad \text{(Ch 5)}$$

Applied element-wise to a vector:

$$\mathbf{h} = \text{ReLU}(\mathbf{z}_1) = \begin{bmatrix} \max(0, z_{1,1}) \\ \max(0, z_{1,2}) \\ \vdots \\ \max(0, z_{1,4}) \end{bmatrix}$$

**The derivative of ReLU** (needed for backpropagation) is the step function:

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases} \quad \text{(Ch 5 — derivative of a piecewise function)}$$

**Why ReLU works:** ReLU is not differentiable at exactly $z=0$, but in practice we set the derivative to 0 there (the "subgradient" convention). The key properties are:
- **Cheap to compute:** just a max with zero
- **No vanishing gradient on the positive side:** derivative is exactly 1, so gradients flow through ReLU-active neurons without attenuation
- **Creates sparsity:** neurons with negative pre-activation output exactly 0, which often helps generalization

**Why we can't use a linear activation:** if $h = W_2(W_1 x + b_1) + b_2$, this equals $(W_2 W_1)x + (W_2 b_1 + b_2)$, which is just another linear transformation. The whole network collapses to one layer.

---

### 10.2.3 Softmax Output — Converting Scores to Probabilities

The second (output) layer produces raw **logit scores** $\mathbf{z}_2 \in \mathbb{R}^{n_{\text{out}}}$. To interpret these as probabilities over classes, we apply **softmax** (from Ch 8):

$$p_k = \text{softmax}(\mathbf{z}_2)_k = \frac{e^{z_{2,k}}}{\displaystyle\sum_{j=1}^{n_{\text{out}}} e^{z_{2,j}}}, \quad k = 1, \ldots, n_{\text{out}} \quad \text{(Ch 8)}$$

**Properties (from Ch 8):**
- $p_k > 0$ for all $k$ (probabilities are positive)
- $\sum_k p_k = 1$ (they sum to 1 — a valid probability distribution)
- Large $z_{2,k}$ → large $p_k$ (the class with the highest logit gets the most probability mass)

**Numerical stability (from Ch 8):** In practice we subtract the maximum logit before exponentiating:

$$p_k = \frac{e^{z_{2,k} - \max_j z_{2,j}}}{\displaystyle\sum_{j} e^{z_{2,j} - \max_j z_{2,j}}} \quad \text{(Ch 8 — numerically stable softmax)}$$

This does not change the result mathematically (the max cancels), but prevents floating-point overflow when logit values are large.

---

### 10.2.4 Complete Forward Pass Summary

For a 2-layer network with input $\mathbf{x}$, parameters $(W_1, \mathbf{b}_1, W_2, \mathbf{b}_2)$:

$$\boxed{\begin{aligned}
\mathbf{z}_1 &= W_1 \mathbf{x} + \mathbf{b}_1 & &\text{(Ch 3 — linear layer)} \\
\mathbf{h} &= \text{ReLU}(\mathbf{z}_1) & &\text{(Ch 5 — nonlinear activation)} \\
\mathbf{z}_2 &= W_2 \mathbf{h} + \mathbf{b}_2 & &\text{(Ch 3 — linear layer)} \\
\mathbf{p} &= \text{softmax}(\mathbf{z}_2) & &\text{(Ch 8 — probability distribution)}
\end{aligned}}$$

The network has transformed a raw input vector into a probability distribution over classes. The question now is: how do we measure whether that distribution is good?

---

## 10.3 The Loss Function — Cross-Entropy from MLE

### 10.3.1 What the Loss Measures

The loss function answers: *how wrong is the prediction?* We need a scalar number that we can minimize with gradient descent.

For classification with $K$ classes, we use **cross-entropy loss**:

$$L = -\sum_{k=1}^{K} y_k \log p_k \quad \text{(Ch 8 — cross-entropy)}$$

where $\mathbf{y} \in \{0,1\}^K$ is the **one-hot** true label vector ($y_k = 1$ for the true class, 0 for all others) and $\mathbf{p}$ is the softmax output.

Because $y_k = 1$ for exactly one class, this simplifies to:

$$L = -\log p_{\text{true class}} \quad \text{(Ch 8)}$$

**Intuition:** If the network assigns probability 0.99 to the correct class, $L = -\log(0.99) \approx 0.01$ — small loss. If it assigns 0.01, $L = -\log(0.01) \approx 4.6$ — large loss. The log function amplifies confidence in the wrong direction.

### 10.3.2 Why Cross-Entropy Comes from MLE

This is not an arbitrary choice — cross-entropy loss **is** maximum likelihood estimation for a categorical distribution.

**Recall from Ch 9 (MLE):** Given a model parameterized by $\theta$, MLE finds the parameters that maximize the likelihood of the observed data:

$$\hat{\theta} = \arg\max_\theta \prod_{i=1}^{n} P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}; \theta) \quad \text{(Ch 9 — MLE)}$$

For a neural network with softmax output, the probability the model assigns to the true label for example $i$ is $p_{\text{true class}}^{(i)}$. Taking the log-likelihood (from Ch 9):

$$\log \mathcal{L}(\theta) = \sum_{i=1}^{n} \log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}; \theta) = \sum_{i=1}^{n} \sum_{k=1}^{K} y_k^{(i)} \log p_k^{(i)} \quad \text{(Ch 9)}$$

**Maximizing** log-likelihood is equivalent to **minimizing** negative log-likelihood:

$$L = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_k^{(i)} \log p_k^{(i)} \quad \text{(Ch 8 + Ch 9 — cross-entropy = negative MLE)}$$

This is exactly the cross-entropy loss. So every time you minimize cross-entropy, you are doing MLE — finding the network weights that make the training data most probable under the model. The connection between Ch 8 and Ch 9 is not coincidence; it is the mathematical foundation.

---

## 10.4 The Backward Pass — Backpropagation

Backpropagation is the **chain rule from Chapter 5 applied recursively** to compute the gradient of the loss with respect to every parameter in the network. We need $\frac{\partial L}{\partial W_1}$, $\frac{\partial L}{\partial \mathbf{b}_1}$, $\frac{\partial L}{\partial W_2}$, $\frac{\partial L}{\partial \mathbf{b}_2}$ in order to run gradient descent.

### 10.4.1 The Key Insight: Softmax + Cross-Entropy Collapses Beautifully

Before working through the full chain, let's derive the gradient of cross-entropy loss through the softmax layer — this is the heart of the calculation.

**Setup:** We have logits $\mathbf{z}_2 \in \mathbb{R}^K$, probabilities $p_k = \frac{e^{z_{2,k}}}{\sum_j e^{z_{2,j}}}$, and loss $L = -\sum_k y_k \log p_k$.

**We want:** $\frac{\partial L}{\partial z_{2,k}}$ for each $k$.

Using the chain rule (Ch 5):

$$\frac{\partial L}{\partial z_{2,k}} = \sum_{j=1}^{K} \frac{\partial L}{\partial p_j} \cdot \frac{\partial p_j}{\partial z_{2,k}}$$

**Step 1:** $\frac{\partial L}{\partial p_j} = -\frac{y_j}{p_j}$ (derivative of $-y_j \log p_j$)

**Step 2:** The derivative of softmax (from Ch 8):

$$\frac{\partial p_j}{\partial z_{2,k}} = \begin{cases} p_k(1 - p_k) & \text{if } j = k \\ -p_j p_k & \text{if } j \neq k \end{cases} \quad \text{(Ch 8 — softmax Jacobian)}$$

**Step 3:** Putting it together:

$$\frac{\partial L}{\partial z_{2,k}} = \frac{\partial L}{\partial p_k} \cdot \frac{\partial p_k}{\partial z_{2,k}} + \sum_{j \neq k} \frac{\partial L}{\partial p_j} \cdot \frac{\partial p_j}{\partial z_{2,k}}$$

$$= \left(-\frac{y_k}{p_k}\right) p_k(1-p_k) + \sum_{j \neq k}\left(-\frac{y_j}{p_j}\right)(-p_j p_k)$$

$$= -y_k(1-p_k) + \sum_{j \neq k} y_j p_k$$

$$= -y_k + y_k p_k + p_k \sum_{j \neq k} y_j$$

$$= -y_k + p_k \underbrace{\sum_{j=1}^{K} y_j}_{=1 \text{ (one-hot: exactly one } y_j=1 \text{)}}$$

$$\boxed{\frac{\partial L}{\partial z_{2,k}} = p_k - y_k} \quad \text{(Ch 5 + Ch 8 + Ch 9 — the miracle)}$$

In vector form: $\frac{\partial L}{\partial \mathbf{z}_2} = \mathbf{p} - \mathbf{y}$.

**This is the capstone of the book.** Three chapters of mathematics — the chain rule (Ch 5), softmax (Ch 8), and MLE / cross-entropy (Ch 9) — combine into a two-symbol answer: prediction minus truth. The cross-terms from the softmax Jacobian cancel exactly because the one-hot labels sum to 1. If the model predicts $p_k = 1$ for the true class, the gradient is zero. If it predicts the wrong class confidently, the gradient is large and in the right direction.

### 10.4.2 Gradient with Respect to Output Layer Weights $W_2$

Now we work backward through the output layer. We have $\mathbf{z}_2 = W_2 \mathbf{h} + \mathbf{b}_2$.

**For the weight matrix $W_2$:**

Each entry $W_{2,kj}$ affects $\mathbf{z}_2$ only through $z_{2,k} = \sum_j W_{2,kj} h_j + b_{2,k}$.

$$\frac{\partial L}{\partial W_{2,kj}} = \frac{\partial L}{\partial z_{2,k}} \cdot \frac{\partial z_{2,k}}{\partial W_{2,kj}} = (p_k - y_k) \cdot h_j \quad \text{(Ch 5 — chain rule)}$$

Stacking these into matrix form (Ch 3):

$$\frac{\partial L}{\partial W_2} = (\mathbf{p} - \mathbf{y})\, \mathbf{h}^\top \quad \text{(Ch 3 — outer product, Ch 5 — chain rule)}$$

**For the bias $\mathbf{b}_2$:**

$$\frac{\partial z_{2,k}}{\partial b_{2,k}} = 1, \quad \text{so} \quad \frac{\partial L}{\partial \mathbf{b}_2} = \mathbf{p} - \mathbf{y} \quad \text{(Ch 5)}$$

### 10.4.3 Gradient with Respect to Hidden Layer Weights $W_1$

To reach $W_1$, we must pass through two more operations: the output-layer multiplication and the ReLU. This is where the chain rule chains.

**Step 1: Gradient with respect to hidden output $\mathbf{h}$**

$$\frac{\partial L}{\partial \mathbf{h}} = W_2^\top \cdot (\mathbf{p} - \mathbf{y}) \quad \text{(Ch 3 — matrix transpose, Ch 5 — chain rule for linear layer)}$$

**Intuition:** $W_2^\top$ "sends back" the error signal from the output to each hidden unit, weighted by how strongly that hidden unit influenced each output.

**Step 2: Gradient through ReLU**

Since $\mathbf{h} = \text{ReLU}(\mathbf{z}_1)$:

$$\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{h}} \odot \text{ReLU}'(\mathbf{z}_1) \quad \text{(Ch 5 — chain rule, element-wise)}$$

where $\odot$ is element-wise multiplication and $\text{ReLU}'(z) = \mathbb{1}[z > 0]$.

This is the **gating** behavior of ReLU: if a hidden neuron was "off" during the forward pass ($z_{1,j} \leq 0$), no gradient flows back through it. That neuron neither contributed to the prediction nor receives any update signal.

**Step 3: Gradient with respect to $W_1$**

Since $\mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1$, by the same logic as Step 2 above:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{z}_1} \cdot \mathbf{x}^\top \quad \text{(Ch 3 — outer product, Ch 5 — chain rule)}$$

$$\frac{\partial L}{\partial \mathbf{b}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \quad \text{(Ch 5)}$$

### 10.4.4 Complete Backpropagation — All Gradients

Let $\boldsymbol{\delta}_2 = \mathbf{p} - \mathbf{y}$ (the output error) and $\boldsymbol{\delta}_1 = (W_2^\top \boldsymbol{\delta}_2) \odot \text{ReLU}'(\mathbf{z}_1)$ (the hidden error).

$$\boxed{\begin{aligned}
\boldsymbol{\delta}_2 &= \mathbf{p} - \mathbf{y} & &\text{(Ch 5+8+9 — combined gradient)} \\[4pt]
\frac{\partial L}{\partial W_2} &= \boldsymbol{\delta}_2\, \mathbf{h}^\top & &\text{(Ch 3+5 — outer product)} \\[2pt]
\frac{\partial L}{\partial \mathbf{b}_2} &= \boldsymbol{\delta}_2 & &\text{(Ch 5)} \\[6pt]
\boldsymbol{\delta}_1 &= (W_2^\top \boldsymbol{\delta}_2) \odot \text{ReLU}'(\mathbf{z}_1) & &\text{(Ch 3+5 — backprop through ReLU)} \\[4pt]
\frac{\partial L}{\partial W_1} &= \boldsymbol{\delta}_1\, \mathbf{x}^\top & &\text{(Ch 3+5 — outer product)} \\[2pt]
\frac{\partial L}{\partial \mathbf{b}_1} &= \boldsymbol{\delta}_1 & &\text{(Ch 5)}
\end{aligned}}$$

The pattern is systematic: every backward step multiplies by the transpose of the forward weight matrix and then gates by the activation derivative. This is the structure that makes backpropagation efficient — each gradient is a local computation at each layer, reusing the error signal from the layer above.

---

## 10.5 Weight Update — Gradient Descent and Adam

With the gradients computed, we apply the update rule.

### 10.5.1 Vanilla SGD

$$W \leftarrow W - \alpha \frac{\partial L}{\partial W} \quad \text{(Ch 6 — gradient descent)}$$

$$\mathbf{b} \leftarrow \mathbf{b} - \alpha \frac{\partial L}{\partial \mathbf{b}} \quad \text{(Ch 6)}$$

The learning rate $\alpha$ controls the step size. Too large and training oscillates; too small and it is slow (Section 6.4 in Ch 6 showed this tradeoff explicitly).

### 10.5.2 Adam (for reference)

In practice, most deep learning training uses **Adam** (from Ch 6), which adapts the learning rate per parameter:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla L \quad \text{(Ch 6 — first moment, like momentum)}$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla L)^2 \quad \text{(Ch 6 — second moment)}$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t} \quad \text{(Ch 6 — bias correction)}$$

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \quad \text{(Ch 6 — Adam update)}$$

Default hyperparameters: $\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

**Why Adam beats SGD in practice:** It normalizes updates by historical gradient magnitude, so parameters that historically receive large gradients get smaller step sizes and vice versa. This effectively sets an adaptive per-parameter learning rate — as if you ran SGD with a different $\alpha$ for every weight in the network.

The from-scratch implementation in Section 10.7 uses vanilla SGD for clarity. Adam would require tracking two extra momentum vectors per parameter but would change none of the mathematics.

---

## 10.6 Regularization in Neural Networks

Large neural networks have millions of parameters and can memorize training data perfectly — fitting noise rather than patterns. **Regularization** (Ch 9) fights this overfitting.

### 10.6.1 L2 Weight Decay

Add a penalty on large weights to the loss:

$$L_{\text{reg}} = L_{\text{CE}} + \frac{\lambda}{2} \|W\|_F^2 \quad \text{(Ch 9 — L2 regularization)}$$

where $\|W\|_F^2 = \sum_{i,j} W_{ij}^2$ is the squared Frobenius norm of the weight matrix (the matrix analog of the Euclidean norm from Ch 2).

The gradient of the regularization term is:

$$\frac{\partial}{\partial W}\left(\frac{\lambda}{2}\|W\|_F^2\right) = \lambda W \quad \text{(Ch 5 — derivative of squared norm)}$$

So the modified weight update becomes:

$$W \leftarrow W - \alpha\left(\frac{\partial L_{\text{CE}}}{\partial W} + \lambda W\right) = W(1 - \alpha\lambda) - \alpha\frac{\partial L_{\text{CE}}}{\partial W} \quad \text{(Ch 6 — weight decay update)}$$

The factor $(1 - \alpha\lambda)$ multiplies the weights before the gradient step — it *decays* the weights toward zero at every update, which is why this is called weight decay.

**Statistical interpretation (from Ch 9):** L2 regularization is equivalent to placing a Gaussian prior on the weights and doing MAP (maximum a posteriori) estimation rather than pure MLE. The MAP solution is:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log \mathcal{L}(\theta) - \frac{\lambda}{2}\|\theta\|^2\right] \quad \text{(Ch 9 — MAP = MLE + prior)}$$

### 10.6.2 Dropout — Regularization by Forgetting

**Dropout** is a regularization technique specific to neural networks. During training, each neuron's output is randomly zeroed out with probability $p$ (typically $p=0.5$):

$$h_j^{\text{drop}} = \begin{cases} h_j / (1-p) & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}$$

The division by $(1-p)$ is the **inverted dropout** scaling — it ensures the expected value of $h_j^{\text{drop}}$ equals $h_j$, so we can use the network at test time without any dropout (just forward-pass as normal).

**Why it works (probability perspective, Ch 7):** Dropout forces the network to not rely on any single neuron. Each training step uses a different random subset of neurons, so the network must learn *redundant* representations. This is analogous to training an ensemble of exponentially many smaller networks and averaging them.

**Connection to bias-variance tradeoff (Ch 9):** Dropout increases bias slightly (the dropped neurons can't contribute) but substantially reduces variance (the model cannot overfit to specific neurons). This is the same bias-variance tradeoff from Ch 9 — regularization always trades some bias for variance reduction.

**Note on the XOR implementation:** We omit dropout from the from-scratch code. With only 4 training examples and a 4-hidden-unit network, dropout would be counterproductive — we have more regularization than we need. In real networks with millions of parameters and thousands of examples, dropout is essential.

---

## 10.7 Complete From-Scratch Implementation

We now build everything from scratch. This is a 2-layer neural network:

```
Input (2) → Hidden (4, ReLU) → Output (2, softmax) → cross-entropy loss
```

trained on the **XOR problem**: given inputs $(x_1, x_2) \in \{0,1\}^2$, predict $x_1 \oplus x_2$ (exclusive OR).

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR is the canonical test for neural networks because it is **not linearly separable** — no single straight line can separate the two classes. A hidden layer is required.

**Important:** We pin the random seed to 7. Different seeds can yield dead ReLU units that prevent convergence. In production code you would try multiple seeds; here we use a known-good one.

```python
"""
XOR Neural Network — Pure Python (no numpy/pytorch)
Architecture: 2 → 4 (ReLU) → 2 (softmax)
Optimizer:    SGD with full-batch gradient descent
Seed: 7 | lr: 0.5 | epochs: 5000
"""
import random
import math

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(7)   # pinned — different seeds may stall due to dead ReLU units

# ═════════════════════════════════════════════════════════════════════════════
#  MATRIX / VECTOR UTILITIES
#  (These replace numpy. Every function maps to a Ch 3 concept.)
# ═════════════════════════════════════════════════════════════════════════════

def zeros(rows, cols):
    """Return a rows×cols matrix of zeros.  (Ch 3 — zero matrix)"""
    return [[0.0] * cols for _ in range(rows)]

def randn_matrix(rows, cols, scale=0.5):
    """Random Gaussian weight matrix.  (Ch 7 — Gaussian distribution)"""
    return [[random.gauss(0, scale) for _ in range(cols)]
            for _ in range(rows)]

def matvec(M, v):
    """Matrix–vector product M @ v.   (Ch 3 — matrix multiplication)
    M is shape (m × n), v is length-n list, returns length-m list."""
    return [sum(M[i][j] * v[j] for j in range(len(v)))
            for i in range(len(M))]

def outer(u, v):
    """Outer product u ⊗ v → matrix.  (Ch 3 — outer product)
    u is length-m, v is length-n, returns (m × n) matrix."""
    return [[u[i] * v[j] for j in range(len(v))]
            for i in range(len(u))]

def transpose(M):
    """Transpose a matrix.             (Ch 3 — matrix transpose)"""
    rows, cols = len(M), len(M[0])
    return [[M[r][c] for r in range(rows)] for c in range(cols)]

def add_vec(a, b):
    """Element-wise vector addition.   (Ch 2 — vector addition)"""
    return [a[i] + b[i] for i in range(len(a))]

def scale_mat(M, s):
    """Multiply every entry of matrix M by scalar s."""
    return [[M[i][j] * s for j in range(len(M[0]))] for i in range(len(M))]

def add_mat(A, B):
    """Element-wise matrix addition.   (Ch 3 — matrix addition)"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]

# ═════════════════════════════════════════════════════════════════════════════
#  ACTIVATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def relu(z):
    """ReLU activation: max(0, z) element-wise.  (Ch 5 — piecewise function)"""
    return [max(0.0, zi) for zi in z]

def relu_deriv(z):
    """ReLU derivative: 1 if z > 0, else 0.      (Ch 5 — step derivative)"""
    return [1.0 if zi > 0 else 0.0 for zi in z]

def softmax(z):
    """Numerically stable softmax.               (Ch 8 — categorical distribution)
    Subtract max before exp to prevent overflow (Ch 8 — log-sum-exp trick)."""
    m = max(z)
    exps = [math.exp(zi - m) for zi in z]
    s = sum(exps)
    return [e / s for e in exps]

def cross_entropy(p, y_onehot):
    """Cross-entropy loss: -Σ y_k log(p_k).     (Ch 8+9 — negative log-likelihood)
    The 1e-15 guard prevents log(0).             (Ch 7 — numerical stability)"""
    return -sum(y_onehot[k] * math.log(p[k] + 1e-15)
                for k in range(len(p)))

# ═════════════════════════════════════════════════════════════════════════════
#  NETWORK ARCHITECTURE: 2 → 4 (ReLU) → 2 (softmax)
# ═════════════════════════════════════════════════════════════════════════════
n_in, n_hidden, n_out = 2, 4, 2

#  Weight initialisation: small random Gaussians  (Ch 7 — Gaussian)
#  Bias initialisation: zeros
W1 = randn_matrix(n_hidden, n_in, scale=0.5)   # shape 4×2 — (Ch 3)
b1 = [0.0] * n_hidden                           # shape 4
W2 = randn_matrix(n_out, n_hidden, scale=0.5)  # shape 2×4 — (Ch 3)
b2 = [0.0] * n_out                              # shape 2

# ═════════════════════════════════════════════════════════════════════════════
#  XOR DATASET  (4 examples, 2 input features, 2 classes)
# ═════════════════════════════════════════════════════════════════════════════
X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_data = [0,      1,      1,      0     ]   # XOR labels

def to_onehot(label, n_classes=2):
    """Convert integer label to one-hot vector.  (Ch 8 — categorical distribution)"""
    v = [0.0] * n_classes
    v[label] = 1.0
    return v

# ═════════════════════════════════════════════════════════════════════════════
#  FORWARD PASS
# ═════════════════════════════════════════════════════════════════════════════

def forward(x):
    """
    Forward pass for one example.
    Returns intermediate values needed for backpropagation.

    z1 — pre-activation of hidden layer,    shape: (n_hidden,)  [Ch 3]
    h  — hidden layer output = ReLU(z1),    shape: (n_hidden,)  [Ch 5]
    z2 — pre-activation of output layer,    shape: (n_out,)     [Ch 3]
    p  — output probabilities = softmax(z2),shape: (n_out,)     [Ch 8]
    """
    z1 = add_vec(matvec(W1, x), b1)   # z1 = W1 @ x + b1     [Ch 3]
    h  = relu(z1)                      # h  = ReLU(z1)         [Ch 5]
    z2 = add_vec(matvec(W2, h), b2)   # z2 = W2 @ h + b2     [Ch 3]
    p  = softmax(z2)                   # p  = softmax(z2)      [Ch 8]
    return z1, h, z2, p

# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
lr       = 0.5    # learning rate (step size for gradient descent)  [Ch 6]
n_epochs = 5000   # number of full passes over the training data

print(f"Training XOR network: {n_in}→{n_hidden}(ReLU)→{n_out}(softmax)")
print(f"Optimizer: SGD, lr={lr}, seed=7, {n_epochs} epochs")
print()
print(f"{'Epoch':>6} | {'Avg Loss':>10}")
print(f"{'------':>6}-+-{'----------':>10}")

for epoch in range(1, n_epochs + 1):
    total_loss = 0.0

    # Accumulate gradients over all 4 training examples
    # (full-batch gradient descent — Ch 6)
    dW1_acc = zeros(n_hidden, n_in)
    db1_acc = [0.0] * n_hidden
    dW2_acc = zeros(n_out, n_hidden)
    db2_acc = [0.0] * n_out

    for x, label in zip(X_data, Y_data):
        y_oh = to_onehot(label)

        # ── FORWARD PASS ──────────────────────────────────────────────────
        z1, h, z2, p = forward(x)
        total_loss += cross_entropy(p, y_oh)   # accumulate loss  [Ch 8+9]

        # ── BACKWARD PASS — output layer ──────────────────────────────────
        # Combined softmax + cross-entropy gradient: dL/dz2 = p - y
        # Derivation in Section 10.4.1                         [Ch 5+8+9]
        dz2   = [p[k] - y_oh[k] for k in range(n_out)]

        # dL/dW2 = dz2 ⊗ h  (outer product)                   [Ch 3+5]
        dW2_g = outer(dz2, h)

        # dL/db2 = dz2                                         [Ch 5]
        db2_g = dz2[:]

        # ── BACKWARD PASS — hidden layer ──────────────────────────────────
        # Backpropagate error through W2: dL/dh = W2^T @ dz2   [Ch 3+5]
        W2T   = transpose(W2)
        dh    = matvec(W2T, dz2)

        # Gate through ReLU derivative: dL/dz1 = dL/dh ⊙ relu'(z1)  [Ch 5]
        rd    = relu_deriv(z1)
        dz1   = [dh[i] * rd[i] for i in range(n_hidden)]

        # dL/dW1 = dz1 ⊗ x  (outer product)                   [Ch 3+5]
        dW1_g = outer(dz1, x)

        # dL/db1 = dz1                                         [Ch 5]
        db1_g = dz1[:]

        # ── ACCUMULATE gradients ──────────────────────────────────────────
        dW2_acc = add_mat(dW2_acc, dW2_g)
        db2_acc = add_vec(db2_acc, db2_g)
        dW1_acc = add_mat(dW1_acc, dW1_g)
        db1_acc = add_vec(db1_acc, db1_g)

    # ── SGD UPDATE: θ ← θ - α * ∇L  (average gradient over batch) ──────
    # [Ch 6 — gradient descent update rule]
    n = len(X_data)
    W1[:] = add_mat(W1, scale_mat(dW1_acc, -lr / n))
    b1[:] = add_vec(b1, [-lr / n * g for g in db1_acc])
    W2[:] = add_mat(W2, scale_mat(dW2_acc, -lr / n))
    b2[:] = add_vec(b2, [-lr / n * g for g in db2_acc])

    avg_loss = total_loss / n
    if epoch % 100 == 0 or epoch == 1:
        print(f"{epoch:>6} | {avg_loss:>10.6f}")

# ═════════════════════════════════════════════════════════════════════════════
#  FINAL PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
print()
print("Final predictions:")
print(f"{'Input':>8} | {'True':>5} | {'Pred':>5} | {'p(class=1)':>10}")
print(f"{'--------':>8}-+-{'-----':>5}-+-{'-----':>5}-+-{'----------':>10}")
for x, label in zip(X_data, Y_data):
    _, _, _, p = forward(x)
    pred = 1 if p[1] > 0.5 else 0
    print(f"  {x}  |  {label:3d}  |  {pred:3d}  |  {p[1]:8.4f}")

print()
print(f"Final avg cross-entropy loss: {avg_loss:.6f}")
```

### 10.7.1 Running the Code — Verified Output

When you run the code above (Python 3, no dependencies), you get:

```
Training XOR network: 2→4(ReLU)→2(softmax)
Optimizer: SGD, lr=0.5, seed=7, 5000 epochs

 Epoch |   Avg Loss
-------+-----------
     1 |   0.718120
   100 |   0.102532
   200 |   0.025275
   300 |   0.013052
   400 |   0.008558
   500 |   0.006275
   600 |   0.004944
   700 |   0.004050
   800 |   0.003417
   900 |   0.002952
  1000 |   0.002596
  1100 |   0.002314
  1200 |   0.002084
  1300 |   0.001894
  1400 |   0.001733
  1500 |   0.001599
  1600 |   0.001483
  1700 |   0.001381
  1800 |   0.001292
  1900 |   0.001214
  2000 |   0.001145
  2100 |   0.001082
  2200 |   0.001026
  2300 |   0.000975
  2400 |   0.000929
  2500 |   0.000886
  2600 |   0.000848
  2700 |   0.000813
  2800 |   0.000780
  2900 |   0.000750
  3000 |   0.000722
  3100 |   0.000695
  3200 |   0.000671
  3300 |   0.000648
  3400 |   0.000627
  3500 |   0.000607
  3600 |   0.000588
  3700 |   0.000571
  3800 |   0.000554
  3900 |   0.000538
  4000 |   0.000523
  4100 |   0.000509
  4200 |   0.000496
  4300 |   0.000483
  4400 |   0.000471
  4500 |   0.000459
  4600 |   0.000448
  4700 |   0.000438
  4800 |   0.000428
  4900 |   0.000418
  5000 |   0.000409

Final predictions:
   Input |  True |  Pred | p(class=1)
---------+-------+-------+-----------
  [0, 0]  |    0  |    0  |    0.0011
  [0, 1]  |    1  |    1  |    0.9998
  [1, 0]  |    1  |    1  |    0.9998
  [1, 1]  |    0  |    0  |    0.0001

Final avg cross-entropy loss: 0.000409
```

The loss starts near $\ln(2) \approx 0.693$ (the entropy of a uniform binary distribution — from Ch 8), drops rapidly to below 0.01 by epoch 500, and converges to **0.000409** — far below the 0.05 target. All 4 XOR predictions are correct.

### 10.7.2 Tracing the Math Through One Training Step

Let's trace one backward pass step by step, mapping each computation to its equation:

**Example:** input $\mathbf{x} = [0, 1]$, label $= 1$, one-hot $\mathbf{y} = [0, 1]$.

After the forward pass, suppose $\mathbf{p} = [0.45, 0.55]$ (network slightly favors class 1, correct but not confident).

**Output error (Section 10.4.1):**
$$\boldsymbol{\delta}_2 = \mathbf{p} - \mathbf{y} = [0.45-0, 0.55-1] = [0.45, -0.45] \quad \text{(Ch 5+8+9)}$$

Interpretation: class 0 is getting too much probability (0.45 instead of 0), class 1 gets too little (0.55 instead of 1). The error signal has the right sign to correct both.

**Gradient for $W_2$ (Section 10.4.2):**
$$\frac{\partial L}{\partial W_2} = \boldsymbol{\delta}_2 \otimes \mathbf{h} \quad \text{(Ch 3+5)}$$

Each entry: $\frac{\partial L}{\partial W_{2,kj}} = \delta_{2,k} \cdot h_j$. The gradient is large where the error $\delta_{2,k}$ is large **and** the hidden activation $h_j$ is large — both conditions must hold.

**Backprop through $W_2$ (Section 10.4.3):**
$$\frac{\partial L}{\partial \mathbf{h}} = W_2^\top \boldsymbol{\delta}_2 \quad \text{(Ch 3+5)}$$

Each hidden unit gets a gradient signal proportional to its contribution to the output error, weighted by $W_2$.

**Gate through ReLU:**
$$\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{h}} \odot \text{ReLU}'(\mathbf{z}_1) \quad \text{(Ch 5)}$$

Hidden units that were "off" (output 0 during forward pass) receive zero gradient. They do not update. Hidden units that were "on" pass the full gradient through.

**SGD update:**
$$W_1 \leftarrow W_1 - \alpha \cdot \boldsymbol{\delta}_1 \otimes \mathbf{x} \quad \text{(Ch 6)}$$

The weight connecting input feature $x_j$ to hidden unit $i$ is updated proportionally to how much the feature was active ($x_j$) and how much the hidden unit's error signal is ($\delta_{1,i}$).

---

## 10.8 What You Now Know — The Complete Map

Every concept in this book just appeared in the code above. Here is the full mapping:

| Concept | Chapter | Where it appears in the neural network |
|---------|---------|---------------------------------------|
| **Vectors** | Ch 2 | $\mathbf{x}$, $\mathbf{h}$, $\mathbf{p}$, $\mathbf{b}$ — every layer input/output is a vector |
| **Matrix-vector product** | Ch 3 | $W\mathbf{x} + \mathbf{b}$ — the linear layer |
| **Matrix transpose** | Ch 3 | $W_2^\top$ — backpropagating error to previous layer |
| **Outer product** | Ch 3 | $\boldsymbol{\delta} \otimes \mathbf{h}$ — weight gradient |
| **Frobenius norm** | Ch 3 | $\|W\|_F^2$ — L2 regularization penalty |
| **SVD / PCA** | Ch 4 | Feature preprocessing; understanding what each layer "sees" |
| **Derivatives** | Ch 5 | Gradient of loss w.r.t. every parameter |
| **Chain rule** | Ch 5 | Backpropagation — the entire backward pass |
| **ReLU and its derivative** | Ch 5 | Activation function and its gradient gate |
| **Gradient descent** | Ch 6 | Weight update: $\theta \leftarrow \theta - \alpha\nabla L$ |
| **Adam optimizer** | Ch 6 | Adaptive per-parameter learning rates in real training |
| **Probability distributions** | Ch 7 | Network output as a probability distribution |
| **Gaussian distribution** | Ch 7 | Weight initialization: `random.gauss(0, scale)` |
| **Bayes' theorem** | Ch 7 | MAP estimation; understanding dropout as an ensemble |
| **Softmax** | Ch 8 | Final layer: converting logits to probabilities |
| **Cross-entropy** | Ch 8 | The loss function |
| **KL divergence** | Ch 8 | Cross-entropy = KL(true dist ‖ model dist) + constant; VAE losses |
| **MLE** | Ch 9 | Cross-entropy loss IS negative log-likelihood |
| **Bias-variance tradeoff** | Ch 9 | Underfitting vs overfitting; why we regularize |
| **L2 regularization** | Ch 9 | Weight decay: $\lambda\|W\|_F^2$ penalty |
| **Cross-validation** | Ch 9 | Choosing hyperparameters (learning rate, $\lambda$, depth) |

There is not a single line of the training loop that doesn't connect to a chapter in this book.

---

## 10.9 Common Failure Modes — Using the Math to Debug

Now that you can read the math, you can debug with it.

**Symptom: Loss doesn't decrease (or increases)**
- **Diagnosis:** Learning rate too high (Ch 6). The gradient step overshoots the minimum.
- **Fix:** Reduce $\alpha$ by 10×. Plot loss vs epoch — if it oscillates, $\alpha$ is the culprit.

**Symptom: Loss decreases then plateaus far from zero**
- **Diagnosis 1:** Dead ReLU units (Ch 5). Neurons with negative $\mathbf{z}_1$ output zero and receive zero gradient — they never recover. Check: print the fraction of hidden activations that are positive. If it's near zero, this is the problem.
- **Diagnosis 2:** Bad initialization. Very large initial weights saturate the softmax, leaving tiny gradients.
- **Fix:** Try different seeds, use He initialization: scale $= \sqrt{2/n_{\text{in}}}$.

**Symptom: Training loss low, validation loss high**
- **Diagnosis:** Overfitting — high variance (Ch 9 — bias-variance tradeoff).
- **Fix:** Add L2 weight decay (Ch 9), dropout (Section 10.6), more training data, or reduce model size.

**Symptom: Training loss and validation loss both high**
- **Diagnosis:** Underfitting — high bias (Ch 9).
- **Fix:** Increase model capacity (more hidden units, more layers), train longer, reduce regularization.

**Symptom: NaN loss**
- **Diagnosis:** Numerical overflow. Large logits → `exp(large)` → `inf`. Or `log(0)`.
- **Fix:** Use numerically stable softmax (subtract max — Ch 8), add `1e-15` inside `log` (Ch 7 — numerical stability conventions).

---

## 10.10 Scaling Up — From XOR to Real Networks

The XOR network has 4 parameters per weight matrix. Real networks have millions. How does everything we derived scale?

**The math stays exactly the same.** The formulas $\boldsymbol{\delta} = \mathbf{p} - \mathbf{y}$, $\frac{\partial L}{\partial W} = \boldsymbol{\delta} \otimes \mathbf{h}$, $\frac{\partial L}{\partial \mathbf{h}} = W^\top \boldsymbol{\delta}$ hold for any network width.

What changes is **engineering**, not math:

| Aspect | XOR network | Real network |
|--------|-------------|-------------|
| Data | 4 examples | Millions of examples |
| Batch | Full batch | Mini-batches of 32–256 |
| Optimizer | SGD | Adam (Ch 6) |
| Initialization | Gaussian(0, 0.5) | He init: $\mathcal{N}(0, \sqrt{2/n_{\text{in}}})$ |
| Layers | 2 | 10–1000+ (residual connections) |
| Regularization | None | Dropout, weight decay, batch normalization |
| Hardware | CPU, milliseconds | GPU/TPU, hours/days |

**What PyTorch / TensorFlow actually do:** They implement the same forward pass, the same cross-entropy loss, the same backpropagation chain rule, and the same Adam update — but using:
1. **Automatic differentiation** instead of hand-derived gradients (the framework tracks the computation graph and applies the chain rule symbolically)
2. **GPU-accelerated matrix multiplication** (BLAS/cuBLAS instead of our nested Python loops)
3. **Optimized numerical routines** for softmax, loss, etc.

The math you derived in this chapter is what those frameworks compute. When you call `loss.backward()` in PyTorch, it runs the chain rule through the computation graph, computing exactly the gradients in Section 10.4.

---

## 10.11 Where to Go Next

You have the mathematical foundation. Here are the most productive next steps:

### Implement Before Moving On

Before touching a framework, implement one more from-scratch network:
- Replace ReLU with sigmoid and verify the chain rule changes
- Add a third hidden layer and derive the backprop for a 3-layer network
- Implement Adam in place of SGD and compare convergence speed

These exercises will make the abstractions in PyTorch and JAX feel transparent rather than magical.

### Core Resources

**Linear Algebra:**
- Gilbert Strang's *Introduction to Linear Algebra* (MIT OpenCourseWare)
- 3Blue1Brown's "Essence of Linear Algebra" (YouTube) — the best visual introduction

**Calculus and Backpropagation:**
- Andrej Karpathy's "micrograd" (GitHub) — a 100-line autograd engine with the same math as this chapter
- Karpathy's "Neural Networks: Zero to Hero" lecture series (YouTube)

**Probability and Information Theory:**
- Christopher Bishop's *Pattern Recognition and Machine Learning* — Chapter 1 (free PDF online)
- Cover and Thomas, *Elements of Information Theory* — for KL divergence and cross-entropy depth

**Neural Networks:**
- Michael Nielsen's *Neural Networks and Deep Learning* (free online at neuralnetworksanddeeplearning.com) — builds everything from scratch, same spirit as this book
- Ian Goodfellow et al., *Deep Learning* (deeplearningbook.org) — the standard graduate reference

**Practical ML:**
- Andrej Karpathy's *makemore* and *nanoGPT* — transformer implementations with hand-written training loops that you can now read
- *Dive into Deep Learning* (d2l.ai) — mathematical rigor with PyTorch/JAX code side by side

### The Next Layer of Math

Once you're comfortable with what's in this book, these are the natural next topics:

| Topic | Why it matters |
|-------|---------------|
| **Attention and transformers** | Self-attention is a learned softmax over dot products — Ch 8 + Ch 3 |
| **Variational inference** | KL divergence minimization for latent variable models — Ch 8 |
| **Natural gradient** | Optimization using the Fisher information matrix — Ch 9 + Ch 4 |
| **Convolutions as structured matrices** | CNNs are matrix operations with shared weights — Ch 3 |
| **PAC learning theory** | Formal generalization bounds — extends Ch 9 |

---

## 10.12 Chapter Summary

You started this book knowing high school algebra. You now understand every piece of mathematics that runs inside a neural network.

The forward pass is matrix multiplication (Ch 3) followed by a piecewise linear activation function (Ch 5) followed by a softmax that turns scores into probabilities (Ch 8).

The loss is cross-entropy — the negative log-likelihood of the true class under the model's probability distribution (Ch 8 + Ch 9). Minimizing it is exactly MLE.

The backward pass is the chain rule applied recursively (Ch 5). The key identity — $\frac{\partial L}{\partial \mathbf{z}_2} = \mathbf{p} - \mathbf{y}$ — emerges from the interaction between softmax and cross-entropy, with terms canceling because one-hot labels sum to 1.

The weight update is gradient descent or Adam (Ch 6), moving parameters in the direction that reduces the loss. Regularization via L2 weight decay (Ch 9) constrains the weights and reduces overfitting.

**None of this required magic.** Every step followed from definitions you learned in Chapters 2 through 9.

---

## Exercises

**10.1** In the from-scratch code, `dz2 = [p[k] - y_oh[k] for k in range(n_out)]` computes the combined softmax + cross-entropy gradient. Verify this formula by expanding the chain rule for $K=2$ classes: compute $\frac{\partial L}{\partial z_{2,0}}$ and $\frac{\partial L}{\partial z_{2,1}}$ separately using the softmax derivative from Ch 8 and the chain rule from Ch 5.

*Solution:* For $K=2$ with $y = [0, 1]$ (true class is 1):

$p_0 = \frac{e^{z_0}}{e^{z_0}+e^{z_1}}$, $p_1 = \frac{e^{z_1}}{e^{z_0}+e^{z_1}}$, $L = -\log p_1$.

$\frac{\partial L}{\partial z_0} = -\frac{1}{p_1} \cdot \frac{\partial p_1}{\partial z_0} = -\frac{1}{p_1} \cdot (-p_1 p_0) = p_0 = p_0 - y_0$. ✓

$\frac{\partial L}{\partial z_1} = -\frac{1}{p_1} \cdot \frac{\partial p_1}{\partial z_1} = -\frac{1}{p_1} \cdot p_1(1-p_1) = -(1-p_1) = p_1 - 1 = p_1 - y_1$. ✓

Both match the formula $p_k - y_k$.

**10.2** What is the gradient $\frac{\partial L}{\partial W_1}$ in terms of the intermediates $\boldsymbol{\delta}_1$, $\mathbf{x}$? Write it as a matrix equation and explain why it is an outer product.

*Solution:* $\frac{\partial L}{\partial W_1} = \boldsymbol{\delta}_1 \otimes \mathbf{x}$, where $[\boldsymbol{\delta}_1 \otimes \mathbf{x}]_{ij} = \delta_{1,i} x_j$. It is an outer product because $W_{1,ij}$ affects the loss only through $z_{1,i} = \sum_j W_{1,ij} x_j$, so $\frac{\partial z_{1,i}}{\partial W_{1,ij}} = x_j$, and by the chain rule $\frac{\partial L}{\partial W_{1,ij}} = \frac{\partial L}{\partial z_{1,i}} \cdot x_j = \delta_{1,i} \cdot x_j$.

**10.3** In the XOR output, `p(class=1)` for input `[0,0]` is 0.0011 and for `[0,1]` is 0.9998. What is the cross-entropy loss for each of these examples? Which one dominates the average loss?

*Solution:*
- $[0,0]$: true class 0, loss $= -\log(p_0) = -\log(1-0.0011) = -\log(0.9989) \approx 0.0011$.
- $[0,1]$: true class 1, loss $= -\log(0.9998) \approx 0.0002$.

The `[0,0]` example contributes slightly more loss. Both are tiny. Average $\approx (0.0011+0.0002+0.0002+0.0001)/4 \approx 0.0004$, consistent with the reported 0.000409.

**10.4** Suppose you add L2 regularization with $\lambda=0.01$ to the XOR training. Write the modified loss function and the modified gradient expression for $W_1$.

*Solution:*

$$L_{\text{reg}} = L_{\text{CE}} + \frac{0.01}{2}\left(\|W_1\|_F^2 + \|W_2\|_F^2\right) \quad \text{(Ch 9)}$$

The gradient becomes:

$$\frac{\partial L_{\text{reg}}}{\partial W_1} = \frac{\partial L_{\text{CE}}}{\partial W_1} + 0.01 \cdot W_1 = \boldsymbol{\delta}_1 \otimes \mathbf{x} + 0.01 \cdot W_1$$

The SGD update: $W_1 \leftarrow W_1 - \alpha(\boldsymbol{\delta}_1 \otimes \mathbf{x} + 0.01 \cdot W_1) = W_1(1 - 0.01\alpha) - \alpha\,\boldsymbol{\delta}_1 \otimes \mathbf{x}$.

**10.5 (Challenge)** The XOR network uses 4 hidden units. Explain why 2 hidden units (the theoretical minimum for XOR) often fails to converge with ReLU, and why adding more units helps even though 2 would suffice in principle.

*Solution:* In theory, 2 ReLU units suffice to compute XOR (each unit draws a decision boundary, and their intersection separates the classes). In practice, with random initialization:

(a) **Dead ReLU problem (Ch 5):** If both units initialize with negative weights for one input, they both output 0 for that input — providing no gradient signal. With only 2 units, one dead unit halves the network's capacity; both dead means no signal at all.

(b) **Symmetry breaking (Ch 6):** Gradient descent can get stuck at saddle points (Section 6.5) when units are nearly identical. With 2 units, symmetry is not easily broken.

(c) **Wider networks are easier to optimize:** Adding units provides redundant paths through the network. If one path is blocked (dead ReLU), others can still carry gradient signal. The final loss landscape with 4 units has fewer problematic saddle points than with 2. This is a practical consequence of the bias-variance perspective (Ch 9): extra capacity doesn't hurt generalization on this problem, but it dramatically helps optimization.

---

*You have finished the mathematical foundations of machine learning. The next step is to build things. Start with micrograd, then nanoGPT. The math will follow you.*
