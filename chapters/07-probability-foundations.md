# Chapter 7: Probability I — Foundations

> *"All models are wrong, but some are useful. Probability theory is how we quantify exactly how wrong."*

---

## 7.1 Sample Spaces, Events, and Outcomes

### Plain English First

When something random happens — flipping a coin, classifying an image, measuring sensor noise — there is a set of all the things that *could* happen. Probability theory starts by forcing you to write that set down explicitly. Once you have the full menu of possibilities, you can assign numbers (probabilities) to subsets of it.

Three vocabulary words matter:

- **Outcome**: a single, specific result of one experiment (e.g., "heads", "image is a cat", "temperature is 98.6°F").
- **Sample space** $\Omega$: the complete set of all possible outcomes. Nothing outside $\Omega$ can happen.
- **Event**: a collection (subset) of outcomes you care about (e.g., "I got a number greater than 4 on a die").

Think of the sample space as an enum in code — the complete, exhaustive list of variants. An event is a predicate that matches some of those variants.

### Formal Notation

Let $\Omega$ be the **sample space** — a non-empty set. Any **outcome** $\omega \in \Omega$ is a single element. An **event** $A$ is a subset $A \subseteq \Omega$.

| Symbol | Meaning | Code analogy |
|--------|---------|-------------|
| $\Omega$ | Sample space (all outcomes) | `enum` or `set` of all variants |
| $\omega$ | A single outcome | One enum value |
| $A \subseteq \Omega$ | An event | `{x for x in outcomes if predicate(x)}` |
| $A^c = \Omega \setminus A$ | Complement event ("not A") | `outcomes - A` in Python |
| $A \cup B$ | Union event ("A or B") | `A \| B` |
| $A \cap B$ | Intersection event ("A and B") | `A & B` |

### Worked Example 7.1.1 — A Fair Die

The experiment: roll one fair six-sided die.

$$\Omega = \{1, 2, 3, 4, 5, 6\}$$

Define events:

- $A$ = "roll an even number" $= \{2, 4, 6\}$
- $B$ = "roll a number greater than 4" $= \{5, 6\}$
- $A \cap B$ = "even AND greater than 4" $= \{6\}$
- $A \cup B$ = "even OR greater than 4" $= \{2, 4, 5, 6\}$
- $A^c$ = "not even" $= \{1, 3, 5\}$

**Verification:** $|A| + |A^c| = 3 + 3 = 6 = |\Omega|$. Correct.

### Worked Example 7.1.2 — Email Classification

The experiment: a classifier observes one email and assigns a label.

$$\Omega = \{\text{spam}, \text{ham}\}$$

- Event $S$ = "email is spam" $= \{\text{spam}\}$
- Event $S^c$ = "email is not spam" $= \{\text{ham}\}$

For a document classifier with three classes:

$$\Omega = \{\text{positive}, \text{negative}, \text{neutral}\}$$

The event "model is wrong on a negative example" is $A = \{\text{positive}, \text{neutral}\}$ — everything except the correct label.

### Engineer's Angle

In ML, the sample space is often implicit. When you define a classification task with $K$ classes, you are implicitly defining:

$$\Omega = \{0, 1, 2, \ldots, K-1\}$$

A model's output (a probability vector of length $K$) assigns a number to each singleton event $\{\omega\}$. The requirement that "they all sum to 1" is exactly the third Kolmogorov axiom — coming in the next section.

```python
# Sample space and events as Python sets
omega = {1, 2, 3, 4, 5, 6}
A = {2, 4, 6}          # even numbers
B = {5, 6}             # greater than 4

intersection = A & B   # {6}
union = A | B          # {2, 4, 5, 6}
complement_A = omega - A  # {1, 3, 5}

print(f"A ∩ B = {sorted(intersection)}")   # [6]
print(f"A ∪ B = {sorted(union)}")          # [2, 4, 5, 6]
print(f"A^c   = {sorted(complement_A)}")   # [1, 3, 5]
print(f"|A| + |A^c| == |Ω|: {len(A) + len(complement_A) == len(omega)}")  # True
```

---

## 7.2 Probability Axioms (Kolmogorov)

### Plain English First

We want to assign a "likelihood number" to every event. For that assignment to be internally consistent — so we never end up with paradoxes like "the probability of something happening is 120%" — we need three ground rules. Andrey Kolmogorov wrote these down in 1933 and they haven't changed since. Everything else in probability follows from these three rules.

### Formal Notation

A **probability measure** $P$ is a function that assigns a real number to every event. It must satisfy three axioms:

**Axiom 1 (Non-negativity):**
$$P(A) \geq 0 \quad \text{for every event } A$$

**Axiom 2 (Normalization):**
$$P(\Omega) = 1$$

**Axiom 3 (Countable Additivity):**  
If $A$ and $B$ are **mutually exclusive** (disjoint: $A \cap B = \emptyset$), then:
$$P(A \cup B) = P(A) + P(B)$$

More generally, for any countable collection of pairwise disjoint events $A_1, A_2, \ldots$:
$$P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

**Trust this result:** The entire edifice of probability — Bayes' theorem, expected value, everything — is derived from these three axioms.

### Key Consequences Derived from the Axioms

**Here's why** the complement rule holds: $P(A^c) = 1 - P(A)$

Since $A$ and $A^c$ are disjoint and their union is $\Omega$:
$$P(A \cup A^c) = P(A) + P(A^c) \quad \text{(by Axiom 3)}$$
$$P(\Omega) = P(A) + P(A^c) \quad \text{(since } A \cup A^c = \Omega\text{)}$$
$$1 = P(A) + P(A^c) \quad \text{(by Axiom 2)}$$
$$\therefore \quad P(A^c) = 1 - P(A) \quad \square$$

**Here's why** the inclusion-exclusion rule holds: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

Write $A \cup B$ as three disjoint pieces:

$$A \cup B = (A \setminus B) \cup (A \cap B) \cup (B \setminus A)$$

By Axiom 3: $P(A \cup B) = P(A \setminus B) + P(A \cap B) + P(B \setminus A)$

Also: $P(A) = P(A \setminus B) + P(A \cap B)$ and $P(B) = P(B \setminus A) + P(A \cap B)$

Adding: $P(A) + P(B) = P(A \setminus B) + 2P(A \cap B) + P(B \setminus A)$

Subtracting $P(A \cap B)$: $P(A) + P(B) - P(A \cap B) = P(A \cup B) \quad \square$

### Worked Example 7.2.1 — Die Probabilities

For a fair six-sided die, each outcome is equally likely:

$$P(\{k\}) = \frac{1}{6} \quad \text{for } k = 1, 2, 3, 4, 5, 6$$

**Verify Axiom 2:**
$$P(\Omega) = P(\{1\}) + P(\{2\}) + \cdots + P(\{6\}) = \frac{1}{6} \times 6 = 1 \checkmark$$

**Compute $P(A)$ where $A = \{2, 4, 6\}$:**

Since $\{2\}$, $\{4\}$, $\{6\}$ are disjoint, by Axiom 3:
$$P(A) = P(\{2\}) + P(\{4\}) + P(\{6\}) = \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = \frac{3}{6} = \frac{1}{2}$$

**Complement:** $P(A^c) = 1 - \frac{1}{2} = \frac{1}{2}$. Indeed $A^c = \{1, 3, 5\}$ also has three elements.

**Inclusion-exclusion** with $B = \{5, 6\}$:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B) = \frac{3}{6} + \frac{2}{6} - \frac{1}{6} = \frac{4}{6} = \frac{2}{3}$$

Cross-check: $A \cup B = \{2, 4, 5, 6\}$ has 4 elements, so $P(A \cup B) = \frac{4}{6} = \frac{2}{3}$. $\checkmark$

### Engineer's Angle: Axioms in Softmax

A softmax layer in a neural network takes a vector of raw scores (logits) $\mathbf{z} \in \mathbb{R}^K$ and outputs:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

This construction is deliberately engineered to satisfy all three Kolmogorov axioms:

- **Axiom 1:** $e^{z_k} > 0$ always, so each output is positive.
- **Axiom 2:** The sum equals $\frac{\sum_k e^{z_k}}{\sum_j e^{z_j}} = 1$.
- **Axiom 3:** Disjoint classes, non-overlapping.

```python
import math

def softmax(logits):
    # Subtract max for numerical stability
    max_z = max(logits)
    exps = [math.exp(z - max_z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

logits = [2.1, 0.5, -0.3]
probs = softmax(logits)
print(f"Probabilities: {[round(p,4) for p in probs]}")
print(f"Sum: {sum(probs):.6f}")    # Axiom 2: must be 1.0
print(f"All >= 0: {all(p >= 0 for p in probs)}")  # Axiom 1
```

---

## 7.3 Conditional Probability

### Plain English First

Conditional probability answers: "Given that I already know $B$ happened, how likely is $A$?" Learning new information shrinks the universe of possibilities. If you know the first die came up 5, you no longer care about the other 35 outcomes — your world has collapsed to just the 6 outcomes where the first die shows 5.

### Formal Notation

The **conditional probability** of $A$ given $B$ (assuming $P(B) > 0$) is:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Read as: "the fraction of $B$'s probability mass that also belongs to $A$."

Rearranging (this is the **multiplication rule**, used constantly):

$$P(A \cap B) = P(A \mid B) \cdot P(B)$$

### Worked Example 7.3.1 — Two Dice

Roll two fair dice. Let:
- $B$ = "first die shows 5" $\Rightarrow P(B) = \frac{6}{36} = \frac{1}{6}$
- $A$ = "sum equals 8"

**Find $P(A \mid B)$.**

Step 1 — Find $A \cap B$: pairs where first die = 5 AND sum = 8.
$$\text{Need second die} = 8 - 5 = 3 \Rightarrow (5, 3) \text{ only}$$
$$P(A \cap B) = \frac{1}{36}$$

Step 2 — Apply the formula:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{1/36}{6/36} = \frac{1/36}{1/6} = \frac{1}{36} \times \frac{6}{1} = \frac{6}{36} = \frac{1}{6}$$

**Sanity check:** Given that first die = 5, the second die can be 1–6 equally. Sum = 8 requires second = 3. That's 1 outcome out of 6. So $P(A \mid B) = \frac{1}{6}$. $\checkmark$

### Worked Example 7.3.2 — Email Spam

Suppose 30% of emails are spam ($P(S) = 0.3$), and 80% of spam emails contain the word "win" ($P(\text{"win"} \mid S) = 0.8$). What fraction of *all* emails are spam emails containing "win"?

This is a multiplication rule question:

$$P(S \cap \text{"win"}) = P(\text{"win"} \mid S) \cdot P(S) = 0.8 \times 0.3 = 0.24$$

24% of all emails are spam emails that contain "win". Note this is NOT the same as $P(S \mid \text{"win"})$ — that requires Bayes' theorem (Section 7.5).

### Engineer's Angle: Generative vs. Discriminative Models

Every supervised ML model is essentially trying to learn one of two things:

- **Discriminative:** learn $P(y \mid \mathbf{x})$ directly — given features $\mathbf{x}$, predict the label $y$. (Logistic regression, neural networks)
- **Generative:** learn $P(\mathbf{x}, y) = P(\mathbf{x} \mid y) \cdot P(y)$ — model how features are generated given a class. (Naive Bayes, GMMs)

The conditional probability $P(y \mid \mathbf{x})$ is the *fundamental object* of classification.

```python
from fractions import Fraction

# P(S) = spam prior, P("win"|S) = likelihood
p_spam = Fraction(3, 10)      # 30%
p_win_given_spam = Fraction(8, 10)  # 80%

# Multiplication rule
p_spam_and_win = p_win_given_spam * p_spam
print(f"P(spam ∩ 'win') = {p_win_given_spam} × {p_spam} = {p_spam_and_win}")
# = 24/100 = 6/25

# Verify conditional probability definition
p_B = Fraction(6, 36)   # P(first die = 5)
p_AB = Fraction(1, 36)  # P(sum=8 AND first die=5)
p_A_given_B = p_AB / p_B
print(f"P(sum=8 | first=5) = {p_A_given_B}")  # 1/6
```

---

## 7.4 Independence of Events

### Plain English First

Two events are independent if knowing one happened gives you zero information about whether the other happened. Flipping a coin twice: the second flip doesn't care what the first flip did. This is distinct from events being *mutually exclusive* (which means knowing one happened tells you the other definitely didn't happen — the opposite of independent).

Independence is one of the most important — and most abused — assumptions in ML.

### Formal Notation

Events $A$ and $B$ are **independent** if and only if:

$$P(A \cap B) = P(A) \cdot P(B)$$

**Equivalently** (when $P(B) > 0$):

$$P(A \mid B) = P(A)$$

**Here's why** these are equivalent: if $P(A \cap B) = P(A) \cdot P(B)$, then:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A) \cdot P(B)}{P(B)} = P(A)$$

Knowing $B$ occurred doesn't change $A$'s probability. $\square$

Events $A_1, A_2, \ldots, A_n$ are **mutually independent** if for every subset $\{A_{i_1}, \ldots, A_{i_k}\}$:
$$P(A_{i_1} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \cdots P(A_{i_k})$$

Pairwise independence does NOT imply mutual independence — but mutual independence implies pairwise.

### Worked Example 7.4.1 — Two Coin Flips

Flip a fair coin twice. Sample space:
$$\Omega = \{HH, HT, TH, TT\}, \quad P(\text{each}) = \frac{1}{4}$$

Let $A$ = "first flip is H" $= \{HH, HT\}$, $P(A) = \frac{2}{4} = \frac{1}{2}$

Let $B$ = "second flip is H" $= \{HH, TH\}$, $P(B) = \frac{2}{4} = \frac{1}{2}$

Check independence:
$$P(A \cap B) = P(\{HH\}) = \frac{1}{4}$$
$$P(A) \cdot P(B) = \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$$

Since $P(A \cap B) = P(A) \cdot P(B)$, the events are **independent**. $\checkmark$

Intuitively: the second flip doesn't "remember" the first.

### Worked Example 7.4.2 — Dependent Events

Roll one die. Let $A$ = "result is even" $= \{2,4,6\}$ and $B$ = "result is greater than 4" $= \{5, 6\}$.

$$P(A) = \frac{3}{6} = \frac{1}{2}, \quad P(B) = \frac{2}{6} = \frac{1}{3}$$
$$P(A \cap B) = P(\{6\}) = \frac{1}{6}$$
$$P(A) \cdot P(B) = \frac{1}{2} \times \frac{1}{3} = \frac{1}{6}$$

These happen to be independent! (The outcome being even gives no information about whether it exceeds 4, on a fair die.)

Now let $C$ = "result is greater than 3" $= \{4, 5, 6\}$, $P(C) = \frac{3}{6} = \frac{1}{2}$.
$$P(A \cap C) = P(\{4, 6\}) = \frac{2}{6} = \frac{1}{3}$$
$$P(A) \cdot P(C) = \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$$

Since $\frac{1}{3} \neq \frac{1}{4}$, events $A$ and $C$ are **dependent**. Knowing the die shows more than 3 increases the chance it's even (from $\frac{1}{2}$ to $\frac{2}{3}$).

### Engineer's Angle: The Naive Bayes Assumption

Naive Bayes classifiers classify documents by computing:

$$P(y \mid x_1, x_2, \ldots, x_n) \propto P(y) \cdot \prod_{i=1}^{n} P(x_i \mid y)$$

The word "Naive" refers to the assumption that features $x_1, \ldots, x_n$ are **mutually independent** given the class $y$. In spam filtering, this means "the probability of seeing 'free' and 'winner' together equals the product of their individual probabilities, given the email is spam."

This is usually **false** in the real world — "free winner" is more common than chance would predict. Yet Naive Bayes often works surprisingly well in practice despite this violated assumption. This is an important engineering lesson: a wrong assumption that leads to tractable computation can outperform a correct model that's too expensive to use.

```python
from fractions import Fraction

# Test for independence: P(A ∩ B) == P(A) * P(B)?
omega = {1, 2, 3, 4, 5, 6}
A = {2, 4, 6}
B = {5, 6}

p = lambda event: Fraction(len(event), len(omega))

p_A = p(A)
p_B = p(B)
p_AB = p(A & B)

print(f"P(A) = {p_A}")
print(f"P(B) = {p_B}")
print(f"P(A ∩ B) = {p_AB}")
print(f"P(A)*P(B) = {p_A * p_B}")
print(f"Independent: {p_AB == p_A * p_B}")  # True
```

---

## 7.5 Bayes' Theorem

### Plain English First

Bayes' theorem is the mathematical rule for *updating your beliefs* when you get new evidence. Before seeing evidence: you have a **prior** belief ($P(\text{hypothesis})$). After seeing evidence: you compute a **posterior** belief ($P(\text{hypothesis} \mid \text{evidence})$). The theorem tells you exactly how to do this update.

It is arguably the most important formula in machine learning.

### Derivation — "Here's why"

Start from the **symmetry of joint probability**. The joint event "$A$ and $B$" can be factored two ways using the multiplication rule:

$$P(A \cap B) = P(A \mid B) \cdot P(B) \quad \text{...(i)}$$
$$P(A \cap B) = P(B \mid A) \cdot P(A) \quad \text{...(ii)}$$

Since both equal $P(A \cap B)$, set them equal:

$$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Divide both sides by $P(B)$ (assuming $P(B) > 0$):

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

This is **Bayes' theorem**. $\square$

The names for each piece:
- $P(A)$: **prior** — your belief before seeing evidence
- $P(B \mid A)$: **likelihood** — probability of the evidence given the hypothesis
- $P(B)$: **marginal likelihood** or **evidence** — how probable the evidence is overall
- $P(A \mid B)$: **posterior** — your updated belief after seeing evidence

### Worked Example 7.5.1 — Medical Test (the Classic)

A disease affects 1 in 1,000 people. A test for it has:
- **Sensitivity:** $P(+ \mid \text{disease}) = 0.99$ (detects 99% of real cases)
- **False positive rate:** $P(+ \mid \text{no disease}) = 0.05$ (5% false alarms)

You test positive. What is $P(\text{disease} \mid +)$?

**Step 1 — State the priors:**
$$P(D) = \frac{1}{1000} = 0.001, \qquad P(D^c) = \frac{999}{1000} = 0.999$$

**Step 2 — State the likelihoods:**
$$P(+ \mid D) = 0.99, \qquad P(+ \mid D^c) = 0.05$$

**Step 3 — Compute $P(+)$ using the total probability theorem** (derived in Section 7.6):
$$P(+) = P(+ \mid D) \cdot P(D) + P(+ \mid D^c) \cdot P(D^c)$$
$$= 0.99 \times 0.001 + 0.05 \times 0.999$$
$$= 0.00099 + 0.04995 = 0.05094$$

**Step 4 — Apply Bayes:**
$$P(D \mid +) = \frac{P(+ \mid D) \cdot P(D)}{P(+)} = \frac{0.99 \times 0.001}{0.05094} = \frac{0.00099}{0.05094}$$

**Arithmetic (exact fractions):**

$$P(D) = \frac{1}{1000}, \quad P(+ \mid D) = \frac{99}{100}$$
$$\text{Numerator} = \frac{99}{100} \times \frac{1}{1000} = \frac{99}{100000}$$

$$P(D^c) = \frac{999}{1000}, \quad P(+ \mid D^c) = \frac{5}{100}$$
$$P(+ \mid D^c) \times P(D^c) = \frac{5}{100} \times \frac{999}{1000} = \frac{4995}{100000}$$

$$P(+) = \frac{99}{100000} + \frac{4995}{100000} = \frac{5094}{100000} = \frac{2547}{50000}$$

$$P(D \mid +) = \frac{99/100000}{2547/50000} = \frac{99}{100000} \times \frac{50000}{2547} = \frac{99 \times 50000}{100000 \times 2547} = \frac{4950000}{254700000} = \frac{11}{566}$$

$$P(D \mid +) = \frac{11}{566} \approx 0.0194 = \mathbf{1.94\%}$$

**Interpretation:** Even with a 99% sensitive test, a positive result means you have less than a 2% chance of actually having the disease. Why? Because the disease is so rare ($0.1\%$ prevalence) that the 5% false positive rate generates far more false alarms than real cases.

This result shocks most people and is the core of why medical tests need to be interpreted in context.

### Worked Example 7.5.2 — Naive Bayes Spam Classification

An email arrives with words "free" and "meeting". Prior probabilities and likelihoods:

$$P(S) = 0.4, \quad P(H) = 0.6$$
$$P(\text{"free"} \mid S) = 0.6, \quad P(\text{"free"} \mid H) = 0.1$$
$$P(\text{"meeting"} \mid S) = 0.05, \quad P(\text{"meeting"} \mid H) = 0.5$$

Under the Naive Bayes (independence) assumption:

$$P(S \mid \text{email}) \propto P(S) \cdot P(\text{"free"} \mid S) \cdot P(\text{"meeting"} \mid S)$$
$$= 0.4 \times 0.6 \times 0.05 = 0.012$$

$$P(H \mid \text{email}) \propto P(H) \cdot P(\text{"free"} \mid H) \cdot P(\text{"meeting"} \mid H)$$
$$= 0.6 \times 0.1 \times 0.5 = 0.030$$

**Normalize** (these are proportional, not the actual posteriors yet):
$$P(S \mid \text{email}) = \frac{0.012}{0.012 + 0.030} = \frac{0.012}{0.042} = \frac{12}{42} = \frac{2}{7} \approx 0.286$$
$$P(H \mid \text{email}) = \frac{0.030}{0.042} = \frac{30}{42} = \frac{5}{7} \approx 0.714$$

**Decision:** Classify as ham (probability 71.4%). The word "meeting" was a strong ham signal that outweighed "free".

### Engineer's Angle

```python
from fractions import Fraction

# --- Medical Test ---
p_D  = Fraction(1, 1000)
p_Dc = 1 - p_D
p_pos_D  = Fraction(99, 100)    # sensitivity
p_pos_Dc = Fraction(5, 100)     # false positive rate

# Total probability
p_pos = p_pos_D * p_D + p_pos_Dc * p_Dc
# Bayes
p_D_given_pos = (p_pos_D * p_D) / p_pos

print(f"P(+)             = {p_pos} ≈ {float(p_pos):.5f}")
print(f"P(disease | +)   = {p_D_given_pos} ≈ {float(p_D_given_pos):.5f}")

# --- Naive Bayes spam ---
p_S = Fraction(4, 10)
p_H = Fraction(6, 10)

p_free_S    = Fraction(6, 10)
p_free_H    = Fraction(1, 10)
p_meet_S    = Fraction(5, 100)
p_meet_H    = Fraction(5, 10)

score_S = p_S * p_free_S * p_meet_S
score_H = p_H * p_free_H * p_meet_H
total   = score_S + score_H

print(f"\nP(spam | email)  = {score_S}/{total} = {score_S/total} ≈ {float(score_S/total):.4f}")
print(f"P(ham  | email)  = {score_H}/{total} = {score_H/total} ≈ {float(score_H/total):.4f}")
```

---

## 7.6 Total Probability Theorem

### Plain English First

Sometimes you want $P(B)$ but you don't know it directly. However, you do know how likely $B$ is within each "category" of some partition. The total probability theorem says: take each category's probability of $B$, weight it by how common that category is, and add them up.

This is exactly how you compute the denominator in Bayes' theorem.

### Formal Notation

If events $A_1, A_2, \ldots, A_n$ form a **partition** of $\Omega$ (mutually exclusive, collectively exhaustive: $A_i \cap A_j = \emptyset$ for $i \neq j$ and $\bigcup_i A_i = \Omega$), then for any event $B$:

$$P(B) = \sum_{i=1}^{n} P(B \mid A_i) \cdot P(A_i)$$

**Here's why** this works: since the $A_i$ partition $\Omega$, we can write:
$$B = B \cap \Omega = B \cap \left(\bigcup_{i=1}^{n} A_i\right) = \bigcup_{i=1}^{n} (B \cap A_i)$$

The sets $B \cap A_i$ are disjoint (since $A_i$ are disjoint), so by Axiom 3:
$$P(B) = \sum_{i=1}^{n} P(B \cap A_i) = \sum_{i=1}^{n} P(B \mid A_i) \cdot P(A_i) \quad \square$$

### Worked Example 7.6.1 — Spam Word Frequency

Three types of emails: spam (30%), newsletters (50%), personal (20%). Probability the word "win" appears:

- $P(\text{"win"} \mid \text{spam}) = 0.80$
- $P(\text{"win"} \mid \text{newsletter}) = 0.15$
- $P(\text{"win"} \mid \text{personal}) = 0.05$

Total probability of seeing "win" in a randomly chosen email:

$$P(\text{"win"}) = 0.80 \times 0.30 + 0.15 \times 0.50 + 0.05 \times 0.20$$

**Arithmetic:**
$$= 0.24 + 0.075 + 0.010 = 0.325$$

Cross-check: the weighted average should be between the lowest (0.05) and highest (0.80) likelihood. $0.325$ is in that range. $\checkmark$

### Worked Example 7.6.2 — Mixture Model (ML Connection)

A Gaussian Mixture Model (GMM) assumes data is generated from $K$ Gaussian components with mixing weights $\pi_k$. The marginal density of an observation $x$ is:

$$p(x) = \sum_{k=1}^{K} \pi_k \cdot p(x \mid \text{component } k)$$

This is the total probability theorem in continuous form. The "partition" is over the latent variable (which component generated the point).

### Engineer's Angle

```python
from fractions import Fraction

# Partition: spam, newsletter, personal
types      = ['spam', 'newsletter', 'personal']
priors     = [Fraction(3,10), Fraction(5,10), Fraction(2,10)]
likelihoods = [Fraction(8,10), Fraction(15,100), Fraction(5,100)]

# Verify partition
assert sum(priors) == 1, "Priors must sum to 1"

# Total probability
p_win = sum(lk * pr for lk, pr in zip(likelihoods, priors))
print(f"P('win') = {p_win} = {float(p_win):.4f}")  # 13/40 = 0.325

# Cross-check
terms = [(lk, pr, lk*pr) for lk, pr in zip(likelihoods, priors)]
for t, (lk, pr, product) in zip(types, terms):
    print(f"  P('win'|{t}) × P({t}) = {lk} × {pr} = {product}")
```

---

## 7.7 Random Variables

### Plain English First

A **random variable** is a function that assigns a number to every outcome in the sample space. Instead of working with abstract outcomes like "the email is spam", we work with numbers — which we can do math on. A random variable turns qualitative events into quantities.

There are two flavors:
- **Discrete:** takes a countable set of values (e.g., number of errors in a batch, class label 0/1/2)
- **Continuous:** takes any value in an interval (e.g., a model's confidence score, a pixel intensity)

### Formal Notation

A **random variable** $X$ is a function $X: \Omega \rightarrow \mathbb{R}$.

**Discrete random variable:** $X$ takes values in $\{x_1, x_2, \ldots\}$. Characterized by its **probability mass function (PMF)**:

$$p_X(x) = P(X = x)$$

Requirements: $p_X(x) \geq 0$ and $\sum_x p_X(x) = 1$ (Kolmogorov's axioms).

**Continuous random variable:** $X$ takes values in an interval. Characterized by its **probability density function (PDF)** $f_X(x)$ where:

$$P(a \leq X \leq b) = \int_a^b f_X(x)\, dx$$

**Trust this result:** The integral $\int_a^b f_X(x)\, dx$ computes the area under the curve of $f_X$ between $a$ and $b$ — think of it as the sum of infinitely many thin slivers, each of width $dx$ and height $f_X(x)$. For Python programmers, it's like `sum(f(x) * dx for x in range(a, b, dx))` as `dx` approaches zero.

Requirements for PDF: $f_X(x) \geq 0$ and $\int_{-\infty}^{\infty} f_X(x)\, dx = 1$.

Note: for a continuous RV, $P(X = x) = 0$ for any single point — probability lives in *intervals*, not points.

### Worked Example 7.7.1 — Discrete: Die as a Random Variable

Define $X$ = value shown by a fair die. The PMF is:

$$p_X(k) = \frac{1}{6} \quad \text{for } k \in \{1, 2, 3, 4, 5, 6\}$$

**Verify:** $\sum_{k=1}^{6} \frac{1}{6} = \frac{6}{6} = 1$. $\checkmark$

We can also define derived RVs: Let $Y = \mathbf{1}[X \text{ is even}]$ (indicator of even outcome):

$$P(Y = 1) = P(X \in \{2,4,6\}) = \frac{3}{6} = \frac{1}{2}, \qquad P(Y = 0) = \frac{1}{2}$$

### Worked Example 7.7.2 — Discrete: Bernoulli and Binomial PMFs

A **Bernoulli** random variable $X \sim \text{Bern}(p)$ models a single trial:

$$P(X=1) = p, \qquad P(X=0) = 1-p$$

In ML: $Y_i = 1$ (correct prediction) or $Y_i = 0$ (wrong prediction) with $p = $ model accuracy.

A **Binomial** $X \sim \text{Bin}(n, p)$ counts successes in $n$ independent Bernoulli trials:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Example: $n = 4$ test examples, $p = 0.7$ accuracy. What is $P(X = 3)$ (exactly 3 correct)?

$$P(X = 3) = \binom{4}{3} (0.7)^3 (0.3)^1 = 4 \times 0.343 \times 0.3 = 4 \times 0.1029 = 0.4116$$

**Verify:** $4 \times 0.343 = 1.372$; $1.372 \times 0.3 = 0.4116$. $\checkmark$

### Worked Example 7.7.3 — Continuous: Uniform Distribution

$X \sim \text{Uniform}(0, 1)$ has PDF:

$$f_X(x) = \begin{cases} 1 & 0 \leq x \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

**Verify:** $\int_0^1 1\, dx = 1$. $\checkmark$ (Area of a rectangle with width 1 and height 1.)

$$P(0.2 \leq X \leq 0.7) = \int_{0.2}^{0.7} 1\, dx = 0.7 - 0.2 = 0.5$$

This represents: "a randomly initialized neural network weight (drawn uniformly) has a 50% chance of landing in the interval $[0.2, 0.7]$."

### Engineer's Angle

```python
import math
from fractions import Fraction

# --- Discrete PMF: die ---
def pmf_die(k):
    return Fraction(1, 6) if 1 <= k <= 6 else Fraction(0)

values = range(1, 7)
total = sum(pmf_die(k) for k in values)
print(f"Die PMF sums to: {total}")  # 1

# --- Binomial PMF ---
def comb(n, k):
    return math.comb(n, k)

def binomial_pmf(k, n, p):
    return comb(n, k) * (p**k) * ((1-p)**(n-k))

n, p = 4, 0.7
prob_3 = binomial_pmf(3, n, p)
print(f"P(X=3) with n=4, p=0.7: {prob_3:.4f}")  # 0.4116

# Verify all probabilities sum to 1
total_bin = sum(binomial_pmf(k, n, p) for k in range(n+1))
print(f"Binomial PMF sums to: {total_bin:.6f}")  # 1.000000

# --- Continuous Uniform: approximate with fine grid ---
def uniform_pdf(x, a=0.0, b=1.0):
    return 1.0 / (b - a) if a <= x <= b else 0.0

# Riemann sum for P(0.2 <= X <= 0.7)
dx = 0.0001
xs = [0.2 + i * dx for i in range(int((0.7 - 0.2) / dx))]
approx_prob = sum(uniform_pdf(x) * dx for x in xs)
print(f"P(0.2 <= X <= 0.7) ≈ {approx_prob:.4f}")  # ≈ 0.5000
```

---

## 7.8 Expected Value and Variance

### Plain English First

The **expected value** is the long-run average of a random variable — if you repeated the experiment millions of times, what would you expect the average to be? It's the "center of gravity" of the probability distribution.

The **variance** measures how spread out the values are around that center. High variance means the random variable jumps around a lot; low variance means it's predictable.

In ML, expected value is the **expected loss** (the quantity we minimize during training), and variance is related to **model uncertainty** and the noise in stochastic gradient descent.

### Formal Notation

**Expected value** for a discrete RV:

$$\mathbb{E}[X] = \sum_{x} x \cdot P(X = x)$$

**Expected value** for a continuous RV:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x)\, dx$$

**Variance:**

$$\text{Var}(X) = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right]$$

**Computational shortcut** (derived below):

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Standard deviation:** $\sigma_X = \sqrt{\text{Var}(X)}$

**Key properties** (Trust these results — each follows from linearity of expectation):
- $\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- If $X$ and $Y$ are independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

**Here's why** the shortcut formula holds:

$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2 - 2\mu X + \mu^2]$$

By linearity of expectation ($\mu = \mathbb{E}[X]$ is a constant):

$$= \mathbb{E}[X^2] - 2\mu\,\mathbb{E}[X] + \mu^2 = \mathbb{E}[X^2] - 2\mu^2 + \mu^2 = \mathbb{E}[X^2] - \mu^2 \quad \square$$

### Worked Example 7.8.1 — Fair Die: E[X] and Var(X)

Let $X$ = outcome of a fair die.

**Step 1 — Expected value:**

$$\mathbb{E}[X] = \sum_{k=1}^{6} k \cdot \frac{1}{6} = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = \frac{21}{6} = \frac{7}{2} = 3.5$$

**Step 2 — Compute $\mathbb{E}[X^2]$:**

$$\mathbb{E}[X^2] = \frac{1}{6}(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) = \frac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6}$$

**Step 3 — Variance via shortcut:**

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{91}{6} - \left(\frac{7}{2}\right)^2 = \frac{91}{6} - \frac{49}{4}$$

Convert to common denominator (12):

$$= \frac{182}{12} - \frac{147}{12} = \frac{35}{12} \approx 2.917$$

**Step 4 — Verify via definitional formula** $\text{Var}(X) = \sum_k P(X=k)(k - \mu)^2$:

| $k$ | $k - 3.5$ | $(k-3.5)^2$ | $\frac{1}{6}(k-3.5)^2$ |
|-----|-----------|-------------|----------------------|
| 1 | $-2.5$ | $6.25$ | $25/24$ |
| 2 | $-1.5$ | $2.25$ | $9/24$ |
| 3 | $-0.5$ | $0.25$ | $1/24$ |
| 4 | $0.5$ | $0.25$ | $1/24$ |
| 5 | $1.5$ | $2.25$ | $9/24$ |
| 6 | $2.5$ | $6.25$ | $25/24$ |
| **Sum** | | | $\mathbf{70/24 = 35/12}$ |

Both methods give $\text{Var}(X) = \frac{35}{12}$. $\checkmark$

$\sigma_X = \sqrt{35/12} \approx \sqrt{2.917} \approx 1.708$

### Worked Example 7.8.2 — Expected Loss

A model makes 4 predictions with losses $\ell_1 = 0.5$, $\ell_2 = 0.2$, $\ell_3 = 0.8$, $\ell_4 = 0.3$, all equally likely. Define $L$ = loss on a random sample:

$$\mathbb{E}[L] = \frac{1}{4}(0.5 + 0.2 + 0.8 + 0.3) = \frac{1.8}{4} = 0.45$$

This is the **empirical risk** — the average loss over a finite sample. Training a model minimizes $\mathbb{E}[L]$ (or an approximation of it over the training set).

### Worked Example 7.8.3 — Continuous: Uniform[0,1]

For $X \sim \text{Uniform}(0, 1)$:

$$\mathbb{E}[X] = \int_0^1 x \cdot 1\, dx = \left[\frac{x^2}{2}\right]_0^1 = \frac{1}{2}$$

$$\mathbb{E}[X^2] = \int_0^1 x^2 \cdot 1\, dx = \left[\frac{x^3}{3}\right]_0^1 = \frac{1}{3}$$

$$\text{Var}(X) = \frac{1}{3} - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{4}{12} - \frac{3}{12} = \frac{1}{12}$$

**Trust this result:** The definite integral $\int_a^b x^n\, dx = \left[\frac{x^{n+1}}{n+1}\right]_a^b$ — plug in $b$ and subtract plugging in $a$.

### Engineer's Angle

Expected value = **average loss** (cross-entropy, MSE). Variance matters for:

1. **Bias-variance tradeoff:** $\text{MSE} = \text{Bias}^2 + \text{Variance}$ — models can fail by being wrong on average OR by being inconsistently wrong.
2. **SGD noise:** Mini-batch gradient estimates have high variance; reducing batch size increases variance of gradient estimates.

```python
from fractions import Fraction

# --- Die expected value and variance ---
p = Fraction(1, 6)
values = [Fraction(k) for k in range(1, 7)]

E_X  = sum(v * p for v in values)
E_X2 = sum(v**2 * p for v in values)
Var  = E_X2 - E_X**2

print(f"E[X]    = {E_X}  = {float(E_X)}")            # 7/2 = 3.5
print(f"E[X²]   = {E_X2} ≈ {float(E_X2):.4f}")      # 91/6 ≈ 15.1667
print(f"Var(X)  = {Var}  ≈ {float(Var):.4f}")        # 35/12 ≈ 2.9167

# Verify via definitional formula
mu = E_X
Var_def = sum(p * (v - mu)**2 for v in values)
print(f"Var(X) definitional = {Var_def} = {float(Var_def):.4f}")
print(f"Both methods match: {Var == Var_def}")

# --- Expected loss ---
losses = [0.5, 0.2, 0.8, 0.3]
E_loss = sum(losses) / len(losses)
print(f"\nE[Loss] = {E_loss}")  # 0.45

# --- Approximate E[X] and Var[X] for Uniform(0,1) ---
dx = 0.00001
xs = [i * dx for i in range(int(1/dx) + 1)]
E_approx  = sum(x * 1.0 * dx for x in xs)
E2_approx = sum(x**2 * 1.0 * dx for x in xs)
Var_approx = E2_approx - E_approx**2
print(f"\nUniform(0,1): E[X] ≈ {E_approx:.4f} (exact: 0.5)")
print(f"Uniform(0,1): Var(X) ≈ {Var_approx:.4f} (exact: {1/12:.4f})")
```

---

## 7.9 Joint, Marginal, and Conditional Distributions

### Plain English First

So far we've worked with one random variable at a time. In ML we almost always have many variables at once: features $x_1, x_2, \ldots, x_n$ and a label $y$. To reason about multiple variables together we need **joint distributions** — the probabilities of all combinations of values.

From the joint you can "zoom out" to a single variable (**marginal distribution**) by averaging over everything else, or "zoom in" to one variable given a fixed value of another (**conditional distribution**).

### Formal Notation

For two discrete RVs $X$ and $Y$, the **joint PMF** is:

$$p_{X,Y}(x, y) = P(X = x, Y = y)$$

The **marginal PMF** of $X$ is obtained by summing over all values of $Y$:

$$p_X(x) = \sum_y p_{X,Y}(x, y)$$

The **conditional PMF** of $X$ given $Y = y$ is:

$$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x, y)}{p_Y(y)} \quad \text{(when } p_Y(y) > 0 \text{)}$$

For continuous RVs, replace sums with integrals:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\, dy$$

$$f_{X \mid Y}(x \mid y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}$$

**Independence** in terms of distributions: $X$ and $Y$ are independent iff:

$$p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y) \quad \text{for all } x, y$$

### Worked Example 7.9.1 — Joint Distribution Table

Let $X$ = model correct? (0=no, 1=yes) and $Y$ = data augmented? (0=no, 1=yes).

Observed joint distribution over many experiments:

$$\begin{array}{c|cc|c}
 & Y=0 & Y=1 & \text{Marginal } P(X=x) \\
\hline
X=0 & 2/10 & 1/10 & 3/10 \\
X=1 & 3/10 & 4/10 & 7/10 \\
\hline
\text{Marginal } P(Y=y) & 5/10 & 5/10 & 1 \\
\end{array}$$

**Step 1 — Verify:** $\frac{2}{10} + \frac{1}{10} + \frac{3}{10} + \frac{4}{10} = \frac{10}{10} = 1$. $\checkmark$

**Step 2 — Marginal of $X$:**
$$P(X=0) = P(X=0, Y=0) + P(X=0, Y=1) = \frac{2}{10} + \frac{1}{10} = \frac{3}{10}$$
$$P(X=1) = \frac{3}{10} + \frac{4}{10} = \frac{7}{10}$$

**Step 3 — Marginal of $Y$:**
$$P(Y=0) = \frac{2}{10} + \frac{3}{10} = \frac{5}{10} = \frac{1}{2}$$
$$P(Y=1) = \frac{1}{10} + \frac{4}{10} = \frac{5}{10} = \frac{1}{2}$$

**Step 4 — Conditional $P(X=1 \mid Y=0)$:**

$$P(X=1 \mid Y=0) = \frac{P(X=1, Y=0)}{P(Y=0)} = \frac{3/10}{5/10} = \frac{3}{5} = 0.6$$

Interpretation: when no augmentation is used, the model is correct 60% of the time.

**Step 5 — Check independence:**

If $X$ and $Y$ were independent: $P(X=1, Y=0) = P(X=1) \cdot P(Y=0) = \frac{7}{10} \times \frac{5}{10} = \frac{35}{100} = \frac{7}{20}$.

But $P(X=1, Y=0) = \frac{3}{10} = \frac{6}{20} \neq \frac{7}{20}$.

So $X$ and $Y$ are **not independent** — augmentation is associated with higher accuracy.

### Worked Example 7.9.2 — Chain Rule of Probability

The chain rule generalizes conditional probability to many variables. For three RVs:

$$P(X, Y, Z) = P(X \mid Y, Z) \cdot P(Y \mid Z) \cdot P(Z)$$

In an autoregressive language model generating tokens $w_1, w_2, w_3, \ldots$:

$$P(w_1, w_2, w_3) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2)$$

This factorization is exact (no independence assumptions needed). Language models are trained to predict each $P(w_t \mid w_1, \ldots, w_{t-1})$.

### Engineer's Angle: P(x, y) Factorizations

Every supervised ML model implicitly or explicitly models the joint distribution $P(\mathbf{x}, y)$, which can be factored in two ways:

$$P(\mathbf{x}, y) = P(y \mid \mathbf{x}) \cdot P(\mathbf{x}) \quad \text{(discriminative decomposition)}$$
$$P(\mathbf{x}, y) = P(\mathbf{x} \mid y) \cdot P(y) \quad \text{(generative decomposition)}$$

Discriminative models (logistic regression, SVMs, neural networks) skip modeling $P(\mathbf{x})$ and learn $P(y \mid \mathbf{x})$ directly. Generative models (Naive Bayes, VAEs) learn $P(\mathbf{x} \mid y)$ and $P(y)$, then use Bayes to get $P(y \mid \mathbf{x})$.

```python
from fractions import Fraction

# Joint distribution table
joint = {
    (0, 0): Fraction(2, 10),
    (0, 1): Fraction(1, 10),
    (1, 0): Fraction(3, 10),
    (1, 1): Fraction(4, 10),
}

# Marginals
def marginal_X(x):
    return sum(v for (xi, yi), v in joint.items() if xi == x)

def marginal_Y(y):
    return sum(v for (xi, yi), v in joint.items() if yi == y)

# Conditional
def conditional_X_given_Y(x, y):
    return joint[(x, y)] / marginal_Y(y)

print("Marginals:")
for x in [0, 1]: print(f"  P(X={x}) = {marginal_X(x)}")
for y in [0, 1]: print(f"  P(Y={y}) = {marginal_Y(y)}")

print("\nConditionals given Y=0:")
for x in [0, 1]:
    print(f"  P(X={x}|Y=0) = {conditional_X_given_Y(x, 0)}")

# Check independence
print("\nIndependence check:")
for x in [0, 1]:
    for y in [0, 1]:
        product = marginal_X(x) * marginal_Y(y)
        actual  = joint[(x, y)]
        print(f"  P(X={x},Y={y}): joint={actual}, product={product}, match={actual==product}")
```

---

## 7.10 Summary Table

| Concept | Formula | ML Application |
|---------|---------|----------------|
| Sample space $\Omega$ | Complete set of outcomes | Output classes of a classifier |
| Event $A \subseteq \Omega$ | Subset of outcomes | "Prediction is correct" |
| Complement | $P(A^c) = 1 - P(A)$ | P(wrong) = 1 - accuracy |
| Inclusion-exclusion | $P(A \cup B) = P(A)+P(B)-P(A\cap B)$ | Union of error types |
| Axiom 2 (normalization) | $P(\Omega) = 1$ | Softmax outputs sum to 1 |
| Conditional probability | $P(A\mid B) = \dfrac{P(A\cap B)}{P(B)}$ | $P(\text{label} \mid \text{features})$ |
| Multiplication rule | $P(A\cap B) = P(A\mid B)\cdot P(B)$ | Chain rule in language models |
| Independence | $P(A\cap B) = P(A)\cdot P(B)$ | Naive Bayes feature assumption |
| Bayes' theorem | $P(A\mid B) = \dfrac{P(B\mid A)\cdot P(A)}{P(B)}$ | Posterior over parameters, spam filter |
| Total probability | $P(B) = \sum_i P(B\mid A_i)\cdot P(A_i)$ | Marginal likelihood in mixture models |
| PMF | $p_X(x) = P(X=x)$ | Discrete class probabilities |
| PDF | $P(a\leq X\leq b)=\int_a^b f_X(x)\,dx$ | Continuous output distributions |
| Expected value | $\mathbb{E}[X] = \sum_x x\,p_X(x)$ | Expected (average) loss |
| Variance shortcut | $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | Model uncertainty, SGD noise |
| Joint PMF | $P(X=x,Y=y)$ | Joint dist. of features and labels |
| Marginal PMF | $p_X(x) = \sum_y p_{X,Y}(x,y)$ | Class-marginal feature distribution |
| Conditional dist. | $p_{X\mid Y}(x\mid y) = p_{X,Y}(x,y)/p_Y(y)$ | Generative model likelihood |

---

## 7.11 Exercises

Work through these before looking at the solutions. Each solution is fully worked out.

---

### Exercise 7.1 [Easy] — Sample Space and Events

Two fair coins are flipped. Define the sample space $\Omega$ explicitly. Let $A$ = "at least one head" and $B$ = "exactly one head".

(a) List $A$, $B$, $A \cap B$, $A \cup B$, $A^c$.  
(b) Compute $P(A)$, $P(B)$, $P(A \cap B)$, $P(A \cup B)$.  
(c) Verify the inclusion-exclusion formula for $P(A \cup B)$.

**Solution:**

(a) $\Omega = \{HH, HT, TH, TT\}$, each with probability $\frac{1}{4}$.

$$A = \{HH, HT, TH\} \quad \text{(at least one H)}$$
$$B = \{HT, TH\} \quad \text{(exactly one H)}$$
$$A \cap B = \{HT, TH\} \quad \text{(at least one H AND exactly one H = exactly one H)}$$
$$A \cup B = \{HH, HT, TH\} = A \quad \text{(since } B \subseteq A\text{)}$$
$$A^c = \{TT\}$$

(b)
$$P(A) = \frac{3}{4}, \quad P(B) = \frac{2}{4} = \frac{1}{2}$$
$$P(A \cap B) = \frac{2}{4} = \frac{1}{2}$$
$$P(A \cup B) = P(A) = \frac{3}{4} \quad \text{(since } B \subseteq A\text{)}$$

(c) Inclusion-exclusion:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B) = \frac{3}{4} + \frac{2}{4} - \frac{2}{4} = \frac{3}{4} \checkmark$$

---

### Exercise 7.2 [Easy] — Conditional Probability

A bag has 3 red and 2 blue marbles. You draw one marble without looking, then draw a second (without replacement).

(a) What is $P(\text{2nd is red} \mid \text{1st is red})$?  
(b) What is $P(\text{both red})$?  
(c) What is $P(\text{2nd is red})$ using the total probability theorem?

**Solution:**

(a) After drawing a red marble first, the bag has 2 red and 2 blue (4 total):
$$P(\text{2nd red} \mid \text{1st red}) = \frac{2}{4} = \frac{1}{2}$$

(b) Multiplication rule:
$$P(\text{both red}) = P(\text{2nd red} \mid \text{1st red}) \cdot P(\text{1st red}) = \frac{2}{4} \times \frac{3}{5} = \frac{6}{20} = \frac{3}{10}$$

**Verify:** $\binom{3}{2} / \binom{5}{2} = 3/10$. $\checkmark$

(c) Partition on the first draw:

$$P(\text{2nd red}) = P(\text{2nd red} \mid \text{1st red}) \cdot P(\text{1st red}) + P(\text{2nd red} \mid \text{1st blue}) \cdot P(\text{1st blue})$$

After drawing a blue first: 3 red, 1 blue (4 total), so $P(\text{2nd red} \mid \text{1st blue}) = \frac{3}{4}$.

$$P(\text{2nd red}) = \frac{2}{4} \times \frac{3}{5} + \frac{3}{4} \times \frac{2}{5} = \frac{6}{20} + \frac{6}{20} = \frac{12}{20} = \frac{3}{5}$$

**Sanity check:** $P(\text{2nd red}) = \frac{3}{5}$ — by symmetry, the second marble is equally likely to be any of the 5 original marbles, so the probability it's red is $\frac{3}{5}$. $\checkmark$

---

### Exercise 7.3 [Medium] — Bayes' Theorem

A factory has three machines: Machine A produces 50% of widgets (2% defective), Machine B produces 30% (5% defective), Machine C produces 20% (10% defective).

A widget is chosen at random and found to be defective. What is the probability it came from Machine C?

**Solution:**

Let $D$ = defective, and let $A$, $B$, $C$ denote the machine of origin.

**Given:**
$$P(A) = 0.5, \quad P(B) = 0.3, \quad P(C) = 0.2$$
$$P(D \mid A) = 0.02, \quad P(D \mid B) = 0.05, \quad P(D \mid C) = 0.10$$

**Step 1 — Total probability** (probability of a random widget being defective):

$$P(D) = P(D \mid A) \cdot P(A) + P(D \mid B) \cdot P(B) + P(D \mid C) \cdot P(C)$$
$$= 0.02 \times 0.5 + 0.05 \times 0.3 + 0.10 \times 0.2$$
$$= 0.010 + 0.015 + 0.020 = 0.045$$

**Step 2 — Bayes' theorem:**

$$P(C \mid D) = \frac{P(D \mid C) \cdot P(C)}{P(D)} = \frac{0.10 \times 0.20}{0.045} = \frac{0.020}{0.045} = \frac{20}{45} = \frac{4}{9} \approx 0.444$$

**Verification (all three machines):**
$$P(A \mid D) = \frac{0.010}{0.045} = \frac{2}{9} \approx 0.222$$
$$P(B \mid D) = \frac{0.015}{0.045} = \frac{3}{9} = \frac{1}{3} \approx 0.333$$
$$P(C \mid D) = \frac{0.020}{0.045} = \frac{4}{9} \approx 0.444$$

Sum: $\frac{2}{9} + \frac{3}{9} + \frac{4}{9} = \frac{9}{9} = 1$. $\checkmark$

Machine C produces only 20% of widgets but accounts for 44% of defects — its high defect rate makes it the most likely culprit.

---

### Exercise 7.4 [Medium] — Expected Value and Variance

A fair four-sided die (values 1, 2, 3, 4) is rolled. Let $X$ = outcome.

(a) Compute $\mathbb{E}[X]$.  
(b) Compute $\text{Var}(X)$ using the shortcut formula. Verify with the definitional formula.  
(c) If $Y = 3X - 2$, compute $\mathbb{E}[Y]$ and $\text{Var}(Y)$ using the linear transformation rules.

**Solution:**

Each outcome has probability $\frac{1}{4}$.

(a)
$$\mathbb{E}[X] = \frac{1}{4}(1 + 2 + 3 + 4) = \frac{10}{4} = \frac{5}{2} = 2.5$$

(b) **Shortcut:**

$$\mathbb{E}[X^2] = \frac{1}{4}(1^2 + 2^2 + 3^2 + 4^2) = \frac{1}{4}(1 + 4 + 9 + 16) = \frac{30}{4} = \frac{15}{2}$$

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{15}{2} - \left(\frac{5}{2}\right)^2 = \frac{15}{2} - \frac{25}{4} = \frac{30}{4} - \frac{25}{4} = \frac{5}{4} = 1.25$$

**Definitional check** (with $\mu = \frac{5}{2}$):

| $k$ | $k - 2.5$ | $(k-2.5)^2$ | $\frac{1}{4}(k-2.5)^2$ |
|-----|-----------|-------------|----------------------|
| 1 | $-1.5$ | $2.25$ | $9/16$ |
| 2 | $-0.5$ | $0.25$ | $1/16$ |
| 3 | $0.5$ | $0.25$ | $1/16$ |
| 4 | $1.5$ | $2.25$ | $9/16$ |
| **Sum** | | | $\mathbf{20/16 = 5/4}$ |

$\checkmark$ Both give $\text{Var}(X) = \frac{5}{4}$.

(c) Linear transformations with $Y = 3X - 2$:

$$\mathbb{E}[Y] = 3\,\mathbb{E}[X] - 2 = 3 \times \frac{5}{2} - 2 = \frac{15}{2} - 2 = \frac{11}{2} = 5.5$$

$$\text{Var}(Y) = 3^2 \cdot \text{Var}(X) = 9 \times \frac{5}{4} = \frac{45}{4} = 11.25$$

---

### Exercise 7.5 [Hard] — Joint Distribution and Naive Bayes

Three words appear in emails: "invoice", "meeting", "sale". The joint distribution of word presence and email type (work vs. promo) is given by the following joint counts over 1,000 emails:

| | Work emails | Promo emails |
|---|---|---|
| Contains "invoice" only | 200 | 10 |
| Contains "meeting" only | 300 | 20 |
| Contains "sale" only | 50 | 300 |
| None of the above | 50 | 70 |
| **Total** | **600** | **400** |

(a) Compute the marginal probabilities $P(\text{work})$ and $P(\text{promo})$.  
(b) Compute the likelihoods $P(\text{"sale"} \mid \text{work})$ and $P(\text{"sale"} \mid \text{promo})$.  
(c) A new email contains the word "sale". Use Bayes' theorem to classify it.  
(d) Now assume you use a Naive Bayes classifier that also observed "meeting" in the email. Compute $P(\text{work} \mid \text{"meeting", "sale"})$ under the Naive Bayes assumption.

**Solution:**

(a) From totals:
$$P(\text{work}) = \frac{600}{1000} = \frac{3}{5}, \qquad P(\text{promo}) = \frac{400}{1000} = \frac{2}{5}$$

(b) Within work emails: "sale" appears in 50 of 600.

$$P(\text{"sale"} \mid \text{work}) = \frac{50}{600} = \frac{1}{12} \approx 0.0833$$

Within promo emails: "sale" appears in 300 of 400.

$$P(\text{"sale"} \mid \text{promo}) = \frac{300}{400} = \frac{3}{4} = 0.75$$

(c) Total probability:

$$P(\text{"sale"}) = P(\text{"sale"} \mid \text{work}) \cdot P(\text{work}) + P(\text{"sale"} \mid \text{promo}) \cdot P(\text{promo})$$
$$= \frac{1}{12} \times \frac{3}{5} + \frac{3}{4} \times \frac{2}{5} = \frac{3}{60} + \frac{6}{20} = \frac{1}{20} + \frac{6}{20} = \frac{7}{20}$$

Bayes:

$$P(\text{work} \mid \text{"sale"}) = \frac{P(\text{"sale"} \mid \text{work}) \cdot P(\text{work})}{P(\text{"sale"})} = \frac{1/12 \times 3/5}{7/20} = \frac{1/20}{7/20} = \frac{1}{7} \approx 0.143$$

$$P(\text{promo} \mid \text{"sale"}) = 1 - \frac{1}{7} = \frac{6}{7} \approx 0.857$$

**Classify as promo.**

(d) Compute $P(\text{"meeting"} \mid \text{work})$ and $P(\text{"meeting"} \mid \text{promo})$:

$$P(\text{"meeting"} \mid \text{work}) = \frac{300}{600} = \frac{1}{2}$$
$$P(\text{"meeting"} \mid \text{promo}) = \frac{20}{400} = \frac{1}{20}$$

Naive Bayes scores (proportional to posterior, assuming independence of words given class):

$$\text{score}(\text{work}) = P(\text{work}) \cdot P(\text{"meeting"} \mid \text{work}) \cdot P(\text{"sale"} \mid \text{work})$$
$$= \frac{3}{5} \times \frac{1}{2} \times \frac{1}{12} = \frac{3}{120} = \frac{1}{40}$$

$$\text{score}(\text{promo}) = P(\text{promo}) \cdot P(\text{"meeting"} \mid \text{promo}) \cdot P(\text{"sale"} \mid \text{promo})$$
$$= \frac{2}{5} \times \frac{1}{20} \times \frac{3}{4} = \frac{6}{400} = \frac{3}{200}$$

Normalize:

$$\text{total} = \frac{1}{40} + \frac{3}{200} = \frac{5}{200} + \frac{3}{200} = \frac{8}{200} = \frac{1}{25}$$

$$P(\text{work} \mid \text{"meeting", "sale"}) = \frac{1/40}{1/25} = \frac{25}{40} = \frac{5}{8} = 0.625$$

$$P(\text{promo} \mid \text{"meeting", "sale"}) = \frac{3/200}{1/25} = \frac{3 \times 25}{200} = \frac{75}{200} = \frac{3}{8} = 0.375$$

**Classify as work** (probability 62.5%). Adding "meeting" flipped the classification from promo to work — the word "meeting" is a very strong work signal ($P(\text{"meeting"} \mid \text{work}) = 50\%$ vs. $P(\text{"meeting"} \mid \text{promo}) = 5\%$).

**Verify:** $\frac{5}{8} + \frac{3}{8} = 1$. $\checkmark$

---

*Next: Chapter 8 — Probability II: Key Distributions, where we build on these foundations to study the distributions that appear everywhere in ML: Gaussian, Bernoulli, Categorical, Poisson, and more.*
