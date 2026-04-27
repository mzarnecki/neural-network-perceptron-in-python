# Intuitive explanation of the single-neuron classifier

Your notebook builds a **binary classifier**: given a point with two coordinates \((x_1, x_2)\), the model predicts whether it belongs to class **0** or **1**.

It starts from the geometric idea of a separating line, then turns that into code with `sigmoid`, `forward_propagation`, `compute_cost`, `backward_propagation`, parameter updates, and final predictions with a `0.5` threshold.

The notebook’s generated dataset uses `X` with shape `(2, m)`, `Y` with shape `(1, m)`, and labels points as class `1` when `x_1 = 1` and `x_2 = 0`, with small random noise added to spread the points out.

## 1. What problem is the perceptron solving?

Imagine a sheet of paper with dots on it:

- blue dots = class 0
- red dots = class 1

Each dot has two numbers:

- \(x_1\): horizontal position
- \(x_2\): vertical position

The job of the model is:

> Can I draw a line so that red points are on one side and blue points are on the other?

That is the meaning of this equation from the notebook:

$$
w_1 x_1 + w_2 x_2 + b = 0
$$

This is just the equation of a line.

### Intuition for each symbol

- \(x_1, x_2\) — the input coordinates of one point
- \(w_1, w_2\) — weights, telling the model how important each coordinate is
- \(b\) — bias, which shifts the line up/down or left/right
- \(z = w_1 x_1 + w_2 x_2 + b\) — a score

Think of \(z\) as:

- positive: “this point looks more like class 1”
- negative: “this point looks more like class 0”
- near zero: “this point is close to the boundary”

So the line itself is the set of points where the score is exactly zero.

## 2. Why do we compute \(z = Wx + b\)?

The notebook writes:

$$
z^{(i)} = w_1 x_1^{(i)} + w_2 x_2^{(i)} + b = W x^{(i)} + b
$$

This is the score for the \(i\)-th example.

### Very intuitive meaning

Suppose:

$$
w_1 = 3, \quad w_2 = -2, \quad b = -1
$$

and one point is:

$$
x = (1,\ 0.2)
$$

Then:

$$
z = 3 \cdot 1 + (-2) \cdot 0.2 - 1 = 3 - 0.4 - 1 = 1.6
$$

Positive score, so the model leans toward class 1.

If another point is:

$$
x = (0,\ 1)
$$

then:

$$
z = 3 \cdot 0 + (-2) \cdot 1 - 1 = -3
$$

Negative score, so the model leans toward class 0.

So before any activation function, the neuron is really just doing:

1. multiply inputs by weights
2. add them up
3. add bias
4. get one score

## 3. Why do we need the sigmoid?

The notebook then applies the sigmoid:

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

and later predicts class 1 if \(a > 0.5\), otherwise class 0.

### What sigmoid does intuitively

Sigmoid takes any real number and squeezes it into the range:

$$
(0, 1)
$$

Examples:

- if \(z = 0\), sigmoid gives \(0.5\)
- if \(z\) is large positive, sigmoid gives something close to \(1\)
- if \(z\) is large negative, sigmoid gives something close to \(0\)

So sigmoid turns the raw score into something that behaves like a probability.

### Why this is useful

The raw score \(z\) can be any number:

- \(-10\)
- \(0.8\)
- \(23\)

That is hard to interpret directly.

But after sigmoid:

- \(-10 ~ 0.000045\)
- \(0.8 ~ 0.69\)
- \(23 ~ 1.0\)

Much easier:

- close to 1 → likely class 1
- close to 0 → likely class 0

### Why threshold 0.5?

Because sigmoid is centered at 0:

$$
\sigma(0) = 0.5
$$

So:

- \(z > 0 &rarr; a > 0.5\)
- \(z < 0 &rarr; a < 0.5\)

That means thresholding at 0.5 is the same as asking which side of the decision line the point is on.

## 4. How is this written for all training examples at once?

The notebook switches from one point to all points together:

$$
Z = W X + b, \qquad A = \sigma(Z)
$$

where `X` stores all training examples as columns.

This part confuses many beginners, so let’s slow down.

### Shapes matter

Your dataset is created with:

- `X` shape: `(2, m)`
- `Y` shape: `(1, m)`

So each **column** of `X` is one point:

$$
X =
\begin{bmatrix}
x_1^{(1)} & x_1^{(2)} & x_1^{(3)} & \cdots \\
x_2^{(1)} & x_2^{(2)} & x_2^{(3)} & \cdots
\end{bmatrix}
$$

If \(m = 50\), then `X` has shape `(2, 50)`.

The weight matrix for one output neuron should have shape:

$$
W \in \mathbb{R}^{1 \times 2}
$$

and bias:

$$
b \in \mathbb{R}^{1 \times 1}
$$

Then:

- `W`: `(1, 2)`
- `X`: `(2, 50)`

so:

- `W X`: `(1, 50)`

That gives one score for each of the 50 examples.

### Tiny example

Let:

$$
W = \begin{bmatrix} 2 & -1 \end{bmatrix}
$$

and:

$$
X =
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1
\end{bmatrix}
$$

Then:

$$
W X =
\begin{bmatrix} 2 & -1 \end{bmatrix}
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1
\end{bmatrix}
=
\begin{bmatrix}
2 & -1 & 1
\end{bmatrix}
$$

If \(b = -0.5\), then:

$$
Z = W X + b =
\begin{bmatrix}
1.5 & -1.5 & 0.5
\end{bmatrix}
$$

After sigmoid:

$$
A = \sigma(Z) \approx
\begin{bmatrix}
0.82 & 0.18 & 0.62
\end{bmatrix}
$$

So the model predicts:

- first point → class 1
- second point → class 0
- third point → class 1

That is exactly what your `forward_propagation` function does:

```python
Z = np.matmul(W, X) + b
A = sigmoid(Z)
```

## 5. What dataset is the notebook actually using?

The notebook creates random points near the corners of the square \((0,0)\), \((0,1)\), \((1,0)\), \((1,1)\), then labels them like this:

$$
y = 1 \text{ if } x_1 = 1 \text{ and } x_2 = 0, \quad \text{otherwise } y = 0
$$

implemented with:

```python
Y = np.logical_and(X[0] == 1, X[1] == 0).astype(int).reshape((1, m))
```

and then small random noise is added so points are not all stacked on top of each other.

### Intuition

This means:

- points near \((1, 0)\) are red / class 1
- points near the other corners are blue / class 0

So the model is trying to isolate the region near \((1,0)\) using a line.

## 6. What is the loss function and why is it needed?

The notebook uses log loss:

$$
\mathcal{L}(W,b) =
\frac{1}{m} \sum_{i=1}^{m}
\left(
-y^{(i)} \log(a^{(i)})
-(1-y^{(i)}) \log(1-a^{(i)})
\right)
$$

This looks scary at first, but the intuition is simple:

> The loss tells us how wrong the model is.

### Case A: true label is 1

Then \(y = 1\), so the formula becomes:

$$
-\log(a)
$$

What happens?

- if \(a = 0.99\), loss is tiny
- if \(a = 0.8\), loss is small
- if \(a = 0.1\), loss is large
- if \(a = 0.01\), loss is huge

So if the true answer is 1, the model is rewarded for predicting a probability close to 1.

### Case B: true label is 0

Then \(y = 0\), so the formula becomes:

$$
-\log(1-a)
$$

Now:

- if \(a = 0.01\), loss is tiny
- if \(a = 0.2\), loss is small
- if \(a = 0.9\), loss is large

So if the true answer is 0, the model is rewarded for predicting a probability close to 0.

### Why log?

Because it punishes **confident wrong answers** very strongly.

That is exactly what we want:

- “I was unsure” is not as bad
- “I was almost certain, and still wrong” is much worse

### Example

Suppose true label is \(y = 1\).

If model predicts:

- \(a = 0.9\), loss \(= -\log(0.9) \approx 0.105\)
- \(a = 0.5\), loss \(= -\log(0.5) \approx 0.693\)
- \(a = 0.1\), loss \(= -\log(0.1) \approx 2.303\)

So lower is better.

## 7. Why do we subtract \(Y\) from \(A\) in backpropagation?

The notebook computes:

```python
dZ = A - Y
dW = 1/m * np.dot(dZ, X.T)
db = 1/m * np.sum(dZ, axis=1, keepdims=True)
```

This matches the matrix derivatives:

$$
\frac{\partial \mathcal{L}}{\partial W}
=
\frac{1}{m}(A-Y)X^T,
\qquad
\frac{\partial \mathcal{L}}{\partial b}
=
\frac{1}{m}(A-Y)\mathbf{1}
$$

### Intuition for \(A - Y\)

For each training example:

$$
dZ = a - y
$$

This is basically the prediction error.

- if \(a = 0.9\) and \(y = 1\), error is \(-0.1\) → prediction is slightly too low
- if \(a = 0.2\) and \(y = 1\), error is \(-0.8\) → badly too low, push harder
- if \(a = 0.8\) and \(y = 0\), error is \(0.8\) → too high, push downward
- if \(a \approx y\), error is near 0

So \(A-Y\) tells the model:

- direction to move
- how strongly to move

## 8. Why is \(dW = \frac{1}{m}(A-Y)X^T\)?

This is easier than it looks.

### First intuition

A weight should change based on:

1. how wrong the model was
2. how much that input contributed

So for one feature:

- big error × big input → bigger weight change
- tiny error or tiny input → smaller change

That is why gradients combine error and input.

### Scalar version

For one example:

$$
\frac{\partial \mathcal{L}}{\partial w_1} = (a-y)x_1
$$

$$
\frac{\partial \mathcal{L}}{\partial w_2} = (a-y)x_2
$$

If the model predicts too high, and \(x_1\) is positive, the gradient for \(w_1\) is positive. Then gradient descent subtracts that, reducing \(w_1\).

If the model predicts too low, the gradient is negative, and subtracting a negative increases the weight.

So gradient descent is basically:

- “if feature contributed in the wrong direction, correct it”
- “if feature should matter more, increase its weight”

### Why transpose \(X^T\)?

Because:

- \(A-Y\) has shape `(1, m)`
- \(X^T\) has shape `(m, 2)`

So:

$$
(1,m) \cdot (m,2) = (1,2)
$$

which matches the shape of \(W\).

That is the matrix version of “aggregate the correction over all training examples.”

## 9. Why is \(db\) just the average of errors?

Bias is a global shift. It does not depend on a specific feature.

So:

$$
db = \frac{1}{m} \sum (a-y)
$$

If the model tends to predict too high overall, \(db\) is positive and gradient descent reduces \(b\).

If the model predicts too low overall, \(db\) is negative and gradient descent increases \(b\).

So bias is like moving the whole decision boundary at once.

## 10. Why do we update with minus?

The notebook updates parameters using:

$$
W = W - \alpha \frac{\partial \mathcal{L}}{\partial W}, \qquad
b = b - \alpha \frac{\partial \mathcal{L}}{\partial b}
$$

where \(\alpha\) is the learning rate.

### Intuition

The gradient points uphill, where loss increases.

But we want to reduce the loss, so we move in the opposite direction.

That is why we subtract.

### Learning rate

\(\alpha\) controls step size:

- too small → learning is slow
- too big → model may jump around and fail to settle

In your notebook, the final training uses many iterations and a learning rate of `0.1`, and the cost falls close to zero after enough iterations.

## 11. How the whole training loop works

Your `nn_model` function does this repeatedly:

1. initialize \(W, b\)
2. compute predictions \(A\)
3. compute cost
4. compute gradients \(dW, db\)
5. update \(W, b\)
6. repeat

This is the core idea of learning.

### Human analogy

It is like learning to throw darts:

- throw
- see how far off you were
- adjust your arm
- throw again
- repeat

The perceptron does the same thing numerically.

## 12. Let’s trace one training example manually

Take one example:

$$
x = \begin{bmatrix}1 \\ 0\end{bmatrix}, \quad y = 1
$$

Suppose current parameters are:

$$
W = \begin{bmatrix}0.2 & -0.1\end{bmatrix}, \quad b = 0
$$

### Step 1: linear score

$$
z = 0.2 \cdot 1 + (-0.1) \cdot 0 + 0 = 0.2
$$

### Step 2: sigmoid

$$
a = \sigma(0.2) \approx 0.55
$$

So model says: probability of class 1 is about 55%.

### Step 3: loss

Since true label is 1:

$$
L = -\log(0.55) \approx 0.598
$$

Not terrible, but not great.

### Step 4: error term

$$
dZ = a-y = 0.55 - 1 = -0.45
$$

Negative means prediction is too low.

### Step 5: gradients

$$
dW = dZ \cdot x^T = -0.45 \cdot \begin{bmatrix}1 & 0\end{bmatrix}
= \begin{bmatrix}-0.45 & 0\end{bmatrix}
$$

$$
db = -0.45
$$

### Step 6: update

If learning rate is \(0.1\):

$$
W_{\text{new}} = W - 0.1 dW
$$

$$
= \begin{bmatrix}0.2 & -0.1\end{bmatrix}
- 0.1 \begin{bmatrix}-0.45 & 0\end{bmatrix}
=
\begin{bmatrix}0.245 & -0.1\end{bmatrix}
$$

$$
b_{\text{new}} = 0 - 0.1(-0.45) = 0.045
$$

So after one step:

- \(w_1\) increased
- \(b\) increased

That makes future predictions for this kind of point more likely to be class 1.

That is learning.

## 13. What does the decision boundary plot mean?

The notebook plots the line:

$$
x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}
$$

This is just the line \(w_1 x_1 + w_2 x_2 + b = 0\) rearranged to solve for \(x_2\).

### Why this line matters

- on the line: model is undecided, \(a = 0.5\)
- one side of the line: predicts class 1
- other side of the line: predicts class 0

As training improves, that line moves into a position that better separates the two classes.

## 14. Why does prediction become `A > 0.5`?

Your `predict` function does:

```python
predictions = A > 0.5
```

That is the final yes/no decision.

### Intuition

Before thresholding:

- `A` is “how much the model believes class 1”

After thresholding:

- above `0.5` → class 1
- below or equal to `0.5` → class 0

So:

- `forward_propagation` gives soft probabilities
- `predict` turns them into hard classes

## 15. One subtle point: is this a perceptron or logistic regression?

The notebook calls it a perceptron, but mathematically the single-neuron model with:

- linear score
- sigmoid activation
- log loss

is very close to what is commonly called **logistic regression** or a **single sigmoid neuron**. That is because it outputs probabilities and is trained with log loss, not the original hard-threshold perceptron update rule.

For learning purposes, that is completely fine. In fact, it is a very good way to understand the bridge between classic perceptrons and neural networks.

## 16. What should you remember from all this?

If you only keep the core picture in your head, keep this:

### The neuron does four conceptual steps

1. **Score the point**

   $$
   z = w_1 x_1 + w_2 x_2 + b
   $$

2. **Turn score into probability**

   $$
   a = \sigma(z)
   $$

3. **Measure how wrong it was**

   with log loss

4. **Adjust weights and bias**

   using gradients so the next prediction is better

That is the whole learning story.

## 17. Super-short intuition for each formula

Here is the compact mental map:

- \(z = W x + b\)  
  “Compute a score”

- \(a = \sigma(z)\)  
  “Convert score into probability-like output”

- \(\mathcal{L}\)  
  “Measure how wrong the prediction is”

- \(A-Y\)  
  “Prediction error”

- \(dW = \frac{1}{m}(A-Y)X^T\)  
  “How each weight should change based on error and inputs”

- \(db = \frac{1}{m} \sum(A-Y)\)  
  “How the global shift should change”

- \(W = W - \alpha dW,\ b = b - \alpha db\)  
  “Move parameters to reduce the error”

## 18. One note about the later notebook section

At the end, your notebook also shows why a **single line is not enough** for more complicated patterns and introduces multiple neurons in a hidden layer.

That is the key limitation of the single perceptron: it can only learn a **linear boundary**. When the positive class is surrounded by negatives, you need extra neurons and layers to build a curved or piecewise boundary.
