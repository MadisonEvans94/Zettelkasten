#seed 
upstream: [[Deep Learning]]

---

**video links**: 

---

# Brain Dump: 
- semantic shift vs non-semantic shift ? (Shift types in general)
- what is bootstrapping? 
- data augmentation 
- what are gans? 
- inductive/transductive learning 
- S3VM/TSVM 
- SSGMM
- deeply understanding how semi-supervised learning works 
![[Screen Shot 2023-10-02 at 12.45.46 PM.png]]

--- 



## Types of Error

> Graphics show how model space increases as model complexity increases, moving the green dot closer to the reality green dot 


![[Screen Shot 2023-10-02 at 12.25.47 PM.png]]
![[Screen Shot 2023-10-02 at 12.27.01 PM.png]]
![[Screen Shot 2023-10-02 at 12.27.39 PM.png]]
### Optimization Error 

>Even if your model can perfectly model the world, your algorithm may not be able to find the good weights that model that function 

**Optimization Error** is fundamentally tied to the optimization problem you're trying to solve. In deep learning, the objective is often to minimize a loss function $L(θ)$ with respect to model parameters $θ$. This loss function measures how well your neural network's predictions align with the true data.

Optimization algorithms like [[Stochastic Gradient Descent]]stochastic gradient descent (SGD) aim to find the minimum of this loss function, but they are not always perfect:

1. **Local Minima**: In high-dimensional spaces, although local minima are generally not a problem, saddle points can slow down optimization.
2. **Vanishing/Exploding Gradients**: In deep architectures, gradients can vanish or explode as they propagate back through the layers.
3. **Numerical Instabilities**: Poorly conditioned Hessian matrices, round-off errors, etc., can also affect optimization.
4. **Hyperparameter Sensitivity**: Learning rates, regularization terms, and other hyperparameters can make the optimization process unstable if not set correctly.

In essence, optimization error is the difference between the best possible value of the objective function (if we could solve it exactly) and the value we actually obtain through numerical optimization.

$$Optimization Error=\mathcal{L}(θ_{obtained})−\mathcal{L}(θ_{optimal})$$

### Estimation Error

>Even if we do find the best hypothesis (the best set of weights or parameters for our nn that minimizes training error) that doesn't mean necessarily that we'll be able to generalize

**Estimation error** is a concept that comes from statistical learning theory. It relates to how well the learned model generalizes to new, *unseen* data. You might have a model that fits your training data exceptionally well but performs poorly on new data, indicating it has *overfit*.

1. **Bias-Variance Tradeoff**: A model with high bias (underfitting) has high estimation error on new data because it's too simple. A model with high variance (overfitting) memorizes noise in the training data, also leading to high estimation error.
2. **Data Distribution**: If the training data is not representative of the population, the model will not generalize well.
3. **Model Complexity**: More complex models like deep neural networks are prone to overfitting if not regularized or if trained too long on a limited dataset.

Estimation error can be formally defined as the difference between the expected loss on new data and the best possible expected loss one could achieve (Bayes error).

### Modeling Error

**Modeling Error** represents the shortcomings of the model architecture itself in capturing the underlying data-generating process or function. In other words, modeling error quantifies how well the chosen model family can represent the function that generated the data.

>Given a particular NN architecture, your actual model that represents the real world may not be in that space. Ie, there may be no set of weights that model the real world 

1. **Function Approximation**: In an ideal world, we would like our model to perfectly represent the function that generated the data. However, no model is perfect; neural networks, decision trees, or any other model you pick belongs to a specific class of functions. If the true function falls outside this class, you incur modeling error.

2. **Feature Representation**: Sometimes, the features you have selected may not capture all the nuances of the data. For example, if you're trying to predict a sound signal and you only use frequency features, you might miss out on phase information, thus incurring modeling error.

3. **Assumptions**: Most models come with assumptions. For instance, linear regression assumes a linear relationship between variables. If this assumption is not met, the model will have a high modeling error.

4. **Trade-off with Complexity**: More complex models like deep neural networks may reduce modeling error but at the cost of potentially increasing estimation error (overfitting) and complicating optimization.

### Relationships Between Error Types

- as model complexity increases, so does its capacity to model a problem, thus *reducing* its modeling error 
- increasing model complexity also *increases* estimation error because it's more challenging and requires much more computation power to get the weights right 
- as model complexity increases, optimization error also *increases* because 

> What's the difference between Optimization Error and Estimation Error? 

Alright, imagine you're making the ultimate playlist, okay? You're curating this set of tracks to vibe with your friends, and you want it to be perfect.

#### Optimization Error: The Mixtape Saga

Imagine you have a huge selection of songs and you are trying to formulate the perfect playlist to fit a current mood. However, due to lack of focus or whatever, you miss certain perfect songs that would really elevate the mood. So your playlist is good not perfect though. 

> Tendency to get stuck at local minima 

#### Estimation Error: The Unforeseen Audience

Now, let's say you've got your playlist, and you think it's pretty dope. You've listened to it multiple times, and it feels just right when you're alone. But then you play it at a party, and it doesn't vibe. People are like, "What is this track doing here?" and you realize that what was perfect for solo listening doesn't really generalize to a crowd of different tastes. That's estimation error, dude. You thought you had the universal banger of playlists, but it doesn't generalize well to new audiences (or data).

> Tendency to overfit 
