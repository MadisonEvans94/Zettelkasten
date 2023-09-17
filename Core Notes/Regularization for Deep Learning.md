#evergreen1 
###### upstream: [[Deep Learning]]

[# Machine Learning Tutorial Python - 17: L1 and L2 Regularization | Lasso, Ridge Regression](https://www.youtube.com/watch?v=VqKq78PVO9g&ab_channel=codebasics)

---


# Regularization

When you're training a deep learning model, your goal is to have a model that not only performs well on the training data, but also generalizes well to new, unseen data (the test data). However, sometimes a model may fit the training data too well and capture the noise along with the underlying pattern, a situation known as [[overfitting]]. When this happens, the model will likely perform poorly on new data because it's essentially "memorized" the training data instead of learning the actual patterns.

## Definition 
**Regularization** is a technique used to prevent this overfitting. The idea is to add a penalty to the loss function, a measure of how well the model fits the data, to discourage the model from learning a too complex model and hence to make it more generalized.

## Why and Where is it used?

Regularization is used during the training process of a deep learning model to encourage the model to keep the weights small. By doing this, the model becomes less sensitive to small changes in the input, thus reducing overfitting. Regularization helps the model generalize better from the training data to unseen data, improving the model's predictive performance on new data.

There are several types of regularization techniques commonly used in deep learning:

Certainly, let's delve into the topic of regularization in machine learning and deep learning. Below is a markdown document that expands on your outline, providing both mathematical and intuitive explanations.

---
Certainly, let's delve deeper into the mathematical aspects of L1 and L2 regularization, breaking down the equations and explaining their implications in layman's terms.

---

## L1 Regularization

### Intuition

L1 regularization encourages sparsity in the model parameters by making some of them exactly zero. This effectively reduces the number of features the model uses, making it simpler.

### Formula

$$
L_{\text{new}} = L + \lambda \sum_{i=1}^{n} |w_i|
$$

#### Mathematical Explanation

In the equation, $( L )$ is the original loss function that the model is trying to minimize. The term $( \lambda \sum_{i=1}^{n} |w_i|)$ is the L1 penalty. Here's what each component does:

- $( \lambda )$: This is the regularization strength. A higher value will make the penalty term more influential, pushing more weights towards zero.
  
- $( \sum_{i=1}^{n} |w_i| )$: This is the sum of the absolute values of the weights. The absolute value makes it equally expensive for the weight to be either positive or negative, focusing solely on the magnitude.

#### Layman's Explanation

Imagine you're shopping for groceries with a limited budget (akin to $( \lambda )$). The L1 term is like saying, "I can only afford to buy a few essential items, so I'll skip the rest." In the context of a model, this means only keeping the most important features and setting the rest to zero.

---

## L2 Regularization

### Intuition

L2 regularization discourages large values in the model parameters but does not enforce sparsity. This helps in preventing overfitting by constraining the model's capacity.

### Formula

$$
L_{\text{new}} = L + \lambda \sum_{i=1}^{n} w_i^2
$$

#### Mathematical Explanation

In this equation, $( L )$ is again the original loss function, and \( \lambda \sum_{i=1}^{n} w_i^2 \) is the L2 penalty. Here's what each component does:

- $( \lambda )$: Similar to L1, this controls the strength of the regularization. A higher value will constrain the weights more.
  
- $( \sum_{i=1}^{n} w_i^2 )$: This is the sum of the squares of the weights. Squaring emphasizes larger weights more than smaller ones, making it expensive for the model to have large weights.

#### Layman's Explanation

Think of L2 regularization as a luxury tax on a sports team's payroll. You can have star players (important features with large weights), but the more you spend on these stars, the higher the "tax" you'll pay. This encourages a more balanced team where no single player's salary (or feature's weight) skyrockets.

---

### Key Differences in Layman's Terms

- **L1 Regularization**: Like shopping on a budget, you can only keep the most essential items (features). The rest you have to leave behind (weights become zero).
  
- **L2 Regularization**: Like managing a sports team's payroll, you're encouraged to distribute the budget (weights) more evenly among players (features) to avoid a luxury tax (penalty).

Both methods add a "cost" to the loss function, making it more expensive for the model to fit the training data too closely, thereby improving generalization.

---

## Dropout Regularization

### What is it and how does it work?

Dropout is a technique where during training, random neurons in a layer are "dropped out" (set to zero) with a probability $( p )$. This prevents the model from becoming overly reliant on any specific neuron.

### Why does regularization work?

Regularization works by constraining the hypothesis space of the model, making it less likely to fit to the noise in the training data. This improves generalization to unseen data.

### Breaking the Principle: "Always try to have similar train and test-time input/output distributions"

Dropout introduces a discrepancy between training and testing phases. During testing, no dropout is applied. To compensate, the outputs (or equivalently, the weights) are scaled by $( p )$.

### Dropout as Ensemble

Dropout can be viewed as training an ensemble of sub-networks. Each forward pass during training uses a different subset of neurons, effectively creating a different "sub-model." During testing, the ensemble's prediction is approximated by scaling the outputs.

---

## Early Stopping

Early stopping involves monitoring the model's performance on a validation set and stopping the training process when the performance starts to degrade. This serves as a form of regularization by preventing the model from overfitting the training data.


> To summarize, *regularization is like a guardrail that prevents the model from overfitting by discouraging it from learning overly complex patterns in the training data that might not generalize well to new data. Different types of regularization methods provide different ways of achieving this goal.*


## Code example 
### Import Libraries

```python 
import numpy as np
import matplotlib.pyplot as plt
```
### Generate Synthetic Data 

```python
# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to the input
X_bias = np.c_[np.ones((100, 1)), X]
```

### Regularized Linear Regression Implementation 

#### L1 Regularization (Lasso)

```python 
def lasso_regression(X, y, lambda_val, epochs=1000, learning_rate=0.01):
    m, n = X.shape
    w = np.random.randn(n, 1)
    
    for epoch in range(epochs):
        gradients = 2 / m * X.T.dot(X.dot(w) - y) + lambda_val * np.sign(w)
        w -= learning_rate * gradients
    
    return w

# Train Lasso model
lambda_val = 0.1
w_lasso = lasso_regression(X_bias, y, lambda_val)
```
#### L2 Regularization (Ridge) 

```python 
def ridge_regression(X, y, lambda_val, epochs=1000, learning_rate=0.01):
    m, n = X.shape
    w = np.random.randn(n, 1)
    
    for epoch in range(epochs):
        gradients = 2 / m * X.T.dot(X.dot(w) - y) + 2 * lambda_val * w
        w -= learning_rate * gradients
    
    return w

# Train Ridge model
lambda_val = 0.1
w_ridge = ridge_regression(X_bias, y, lambda_val)
```

## Plotting the Results 

```python 
# Generate test data
X_test = np.linspace(0, 2, 100).reshape(100, 1)
X_test_bias = np.c_[np.ones((100, 1)), X_test]

# Predict using Lasso and Ridge models
y_pred_lasso = X_test_bias.dot(w_lasso)
y_pred_ridge = X_test_bias.dot(w_ridge)

# Plotting
plt.scatter(X, y, label='Actual data')
plt.plot(X_test, y_pred_lasso, label='Lasso', linewidth=2)
plt.plot(X_test, y_pred_ridge, label='Ridge', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```