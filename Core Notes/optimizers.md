#incubator 
upstream: [[Deep Learning]]

---

**links**: 
- [ml cheatsheet - optimizers](https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html)
---

# Brain Dump: 

- Issues in Optimization 
	- noisy gradient estimates 
	- saddle points 
	- ill conditioned Loss Surface 
- Optimizers: Mathematical Descriptions and High Level Intuitions
	- Vanilla Stochastic Gradient Descent 
		- the vanilla steepest descent implementation 
		- Introducing the concept of momentum 
	- Adagrad 
	- RMSProp 
	- Adam 
- Per-Parameter Learning Rate
	- Idea: have a dynamic learning rate for each weight 
- Second Order Approximation of Loss (Hessian)
- Learning Rate Schedules 
- Condition Number: What is it and how is it relevant? 
	- the ratio of largest and smallest eigen value 
	- tells us how different the curvature is along different dimensions
	- if this is high, SGD will make big steps in some dimensions and small steps in other dimensions 
- Using a subset of the data to calculate the loss at each iteration is an "Unbiased Estimator" 
	- The expectation is equal to the true non-stochastic full batch value (this is only on expectation. In reality we'll have high variance, meaning we'll take very "noisy" steps during gradient descent)


--- 

Certainly, let's dive deep into the topic of optimizers in the context of deep learning. I'll expand on each section of your markdown document, providing both mathematical and intuitive explanations.

---

## Issues in Optimization

### Noisy Gradient Estimates

#### Mathematical Explanation
When using stochastic gradient descent (SGD) or its variants, the gradient is estimated using a subset of the data. This introduces noise into the gradient estimate:

$$
\hat{\nabla} = \nabla + \epsilon
$$

where $(\hat{\nabla})$ is the noisy gradient, $(\nabla)$ is the true gradient, and $(\epsilon)$ is the noise term.

#### Intuition
Noisy gradients can cause the optimizer to oscillate and potentially miss the minimum. This is particularly problematic in high-dimensional spaces.

### Saddle Points

#### Mathematical Explanation
At a saddle point, the Hessian matrix has both positive and negative eigenvalues. Mathematically, a point \(x\) is a saddle point if:

$$
\nabla f(x) = 0, \quad \text{and} \quad \text{Hessian has both positive and negative eigenvalues}
$$

![[Screen Shot 2023-09-17 at 10.53.36 AM.png]]

#### Intuition
Saddle points are problematic because the gradient is zero but it's not a minimum or maximum. Optimizers can get stuck or slow down significantly at saddle points.

![[Screen Shot 2023-09-17 at 10.44.59 AM.png]]

### Ill-Conditioned Loss Surface

#### Mathematical Explanation
A loss surface is ill-conditioned if its condition number $( \kappa )$ is high:

$$
\kappa = \frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}
$$

where \( \lambda_{\text{max}} \) and \( \lambda_{\text{min}} \) are the largest and smallest eigenvalues of the Hessian matrix.

#### Intuition
An ill-conditioned surface has very steep and very shallow regions, making it difficult for optimizers like SGD to navigate efficiently.

---

## Using a Subset of the Data to Calculate the Loss at Each Iteration is an "Unbiased Estimator"

### Mathematical Explanation
An unbiased estimator \( \hat{\theta} \) of a parameter \( \theta \) satisfies:

$$
\mathbb{E}[\hat{\theta}] = \theta
$$

In the context of SGD, the expectation of the stochastic gradient is equal to the true gradient.

### Intuition
Although using a subset introduces noise (high variance), on average, the estimator is correct. The trade-off is between computational efficiency and the quality of the gradient estimate.

---

## Optimizers: Mathematical Descriptions and High Level Intuitions

### Vanilla Stochastic Gradient Descent

#### The Vanilla Steepest Descent Implementation

##### Mathematical Explanation
The update rule is:

$$
w_{t+1} = w_t - \eta \nabla L(w_t)
$$

where \( \eta \) is the learning rate, \( w \) are the parameters, and \( \nabla L \) is the gradient of the loss function.

##### Intuition
The optimizer moves in the direction of steepest descent. It's simple but can be slow and susceptible to the issues mentioned above.

#### Introducing the Concept of Momentum

##### Mathematical Explanation
The update rule with momentum \( \gamma \) is:

$$
v_{t+1} = \gamma v_t + \eta \nabla L(w_t)
$$
$$
w_{t+1} = w_t - v_{t+1}
$$
![[Screen Shot 2023-09-17 at 10.48.51 AM.png]]
![[Screen Shot 2023-09-17 at 10.49.12 AM.png]]
##### Intuition
Momentum helps the optimizer to overcome local minima and noise by adding a fraction of the previous update to the current one.

>when beta is 0 then that's just normal sgd 


### Adagrad

#### Mathematical Explanation
The update rule is:

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla L(w_t)
$$

where \( G_t \) is the diagonal matrix of squared gradients and \( \epsilon \) is a small constant to avoid division by zero.

#### Intuition
Adagrad adapts the learning rate for each parameter based on the historical gradient information. It's good for sparse features but can have a diminishing learning rate.

![[Screen Shot 2023-09-17 at 10.57.04 AM.png]]

### RMSProp

#### Mathematical Explanation
The update rule is:

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2
$$
$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla L(w_t)
$$

#### Intuition
RMSProp also adapts the learning rates but uses a moving average of squared gradients. It solves some of Adagrad's problems like the diminishing learning rate.

### Adam

#### Mathematical Explanation
Adam combines the ideas of momentum and RMSProp:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(w_t)
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) \nabla L(w_t)^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$
$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
w_{t+1} = w_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

#### Intuition
Adam is generally effective across a wide range of problems. It combines the benefits of adaptive learning rates with momentum but can sometimes overshoot the optimum.

---

## Per-Parameter Learning Rate

### Mathematical Explanation
In optimizers like Adagrad, RMSProp, and Adam, each parameter \( w_i \) is updated with its own learning rate \( \eta_i \):


$$
w_{i, t+1} = w_{i, t} - \eta_i \nabla L(w_{i, t})
$$



### Intuition
Having a dynamic learning rate for each weight allows the optimizer to adapt to the importance and scale of each feature, making the optimization process more efficient.

---

## Second Order Approximation of Loss (Hessian)

### Mathematical Explanation
The second-order Taylor expansion of the loss \( L \) around \( w \) is:


$$
L(w + \Delta w) \approx L(w) + \nabla L(w)^T \Delta w + \frac{1}{2} \Delta w^T \text{Hessian} \Delta w
$$


### Intuition
Second-order methods like Newton's method use this approximation to find the minimum more accurately but are computationally expensive for large-scale problems.

---

## Learning Rate Schedules

### Mathematical Explanation
The learning rate \( \eta \) can be updated based on a schedule, such as:

$$
\eta_t = \eta_0 \times \text{decay}^{\frac{t}{\text{decay\_step}}}
$$

### Intuition
Decaying the learning rate can help the optimizer to settle into a minimum by reducing oscillations.

---

## Condition Number:

 What is it and how is it relevant?

### Mathematical Explanation
The condition number \( \kappa \) is defined as:

$$
\kappa = \frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}
$$

### Intuition
A high condition number indicates an ill-conditioned problem. In such cases, SGD will make big steps in some dimensions and small steps in others, making the optimization inefficient.

---


