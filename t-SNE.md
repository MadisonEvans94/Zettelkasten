#incubator 
upstream: [[Deep Learning]], [[Data Visualization]], 

---

**links**: 

---

Brain Dump: 


--- 

# Understanding t-SNE for Data Visualization

t-SNE (t-distributed Stochastic Neighbor Embedding) is a powerful tool for visualizing high-dimensional data by reducing it to two or three dimensions, typically for the purposes of data exploration and visualization. This document provides an in-depth look at the t-SNE algorithm, its purpose, its comparison to [[Principal Component Analysis (PCA)]], and a walkthrough of the algorithm itself.

## High-Level Purpose of t-SNE

t-SNE is designed to capture the local structure of high-dimensional data and to reveal global structure such as the presence of clusters at several scales. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

### Why Use t-SNE?

- **Preservation of Local Structure**: t-SNE is particularly adept at unraveling the local structure of the data and revealing clusters and groupings that are often missed by other dimensionality reduction techniques.
- **Visual Interpretability**: The algorithm is tailored for visualization purposes, yielding more interpretable maps that faithfully represent the important relationships in the data.
- **Flexibility**: t-SNE can handle non-linear dependencies which are common in real-world data.

## t-SNE vs. PCA

While both t-SNE and PCA aim to reduce the dimensionality of data, they differ significantly:

- **Linearity**: PCA is a linear algorithm, which means it cannot capture non-linear relationships as effectively as t-SNE.
- **Variance vs. Probability**: PCA preserves global structure and tends to project data along axes of maximum variance. t-SNE, on the other hand, translates high-dimensional Euclidean distances between points into conditional probabilities representing similarities.
- **Scalability**: PCA is computationally less intensive and scales better with the number of dimensions. t-SNE is more computationally expensive, making it less ideal for very large datasets.
- **Use Cases**: PCA is often used for preliminary analysis and data pre-processing, while t-SNE is primarily used for visually inspecting the data after other preprocessing steps have been performed.

## Walkthrough of the t-SNE Algorithm

>The t-SNE algorithm comprises several steps:

### Step 1: Compute Pairwise Similarity in High-Dimensional Space

For each pair of points $( x_i )$ and $( x_j )$ in the high-dimensional space, we compute the conditional probability $( p_{j|i} )$, which is the probability that $( x_i )$ would pick $( x_j )$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $( x_i )$.

The probabilities are given by:

$$p_{j|i} = \frac{exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} exp(-||x_i - x_k||^2 / 2\sigma_i^2)} $$

where  $\sigma_i$ is the variance of the Gaussian that is centered on datapoint $( x_i )$.

To make the probability distribution symmetrical, we then define the joint probabilities as:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

where $( N )$ is the total number of data points.

### Step 2: Compute Pairwise Similarity in Low-Dimensional Space

In the low-dimensional map, t-SNE uses a similar approach but with a Student-t distribution (with one degree of freedom, which is the same as a Cauchy distribution) to compute the joint probabilities $( q_{ij} )$.

$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||y_k - y_l||^2)^{-1}} $$

where $y_i$ and $y_j$ are the low-dimensional representations of $x_i$ and $x_j$, respectively.

### Step 3: Minimize the Kullback-Leibler Divergence

The Kullback-Leibler (KL) divergence is used to measure the difference between the two probability distributions $( p_{ij} )$ and $( q_{ij} )$.

The cost function $( C )$ is the sum of KL divergences over all pairs of points:

$$ C = \sum_{i \neq j} p_{ij} \log\frac{p_{ij}}{q_{ij}} $$

t-SNE aims to find a low-dimensional data representation that minimizes this cost function. The minimization is typically performed using gradient descent.

### Step 4: Gradient Descent

The positions of the points in the map are optimized iteratively using gradient descent. The gradient of the Kullback-Leibler divergence with respect to the points $( y_i )$ is given by:

$$ \frac{\delta C}{\delta y_i} = 4 \sum_{j}(p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1} $$

This gradient is used to update the positions of the points in the low-dimensional space.

### Step 5: Tuning Parameters

- **Perplexity**: This parameter balances attention between local and global aspects of the data and can be thought of as a smooth measure of the effective number of neighbors.
- **Learning Rate**: The learning rate for the gradient descent. If too high, the map may look like a "ball," if too low, the map may be compressed with few outliers.

### Step 6: Iterate Until Convergence

The algorithm iterates until the positions of the points in the map converge or until a maximum number of iterations is reached.

## Conclusion

t-SNE is a non-linear, probabilistic technique mainly used for exploratory data analysis and visualizing high-dimensional data. While it can be computationally intensive and sensitive to hyperparameter settings, it is exceptionally good at revealing the structure of the data on several scales, making it a popular choice for tasks like identifying clusters in the data. However, the non-convex nature of its cost function means that it can converge to different solutions based on the initial conditions and parameters, so it is advisable to run the algorithm several times or with different perplexities to confirm the stability of the resulting visualization.



