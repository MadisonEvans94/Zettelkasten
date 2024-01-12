
#incubator 
###### upstream: [[dimensionality reduction techniques]]

### Details:

**PCA** is a technique that transforms a high-dimensional dataset into a lower-dimensional 'subspace', while attempting to maintain as much of the original information as possible. This is done by transforming the original dataset's axes to a new set of axes, where each new axis (or 'principal component') is a linear combination of the original dataset's features.

The new axes are chosen in a particular way. The first principal component (PC1) is chosen as the direction in which the data varies the most. The second principal component (PC2) is chosen as the direction orthogonal to PC1 in which the data varies the most, and so on.

How do eigenvectors and eigenvalues come into play?

1.  **Eigenvectors**: In PCA, the eigenvectors of the covariance matrix of the data are the principal components. These eigenvectors define the directions of the new feature space.
    
2.  **Eigenvalues**: The corresponding eigenvalues determine the magnitude of the variance in these directions. A larger eigenvalue means that the data has larger variance in that direction, and hence that direction is a better choice for a principal component.
    

So, the eigenvectors and eigenvalues of the covariance matrix of the data are crucial to PCA, as they directly determine the new set of features that the data is transformed into.

Let's go through the steps involved in PCA to make it clearer:

1.  Standardize the dataset (mean=0, standard deviation=1).
    
2.  Calculate the covariance matrix of the data.
    
3.  Calculate the eigenvectors and eigenvalues of the covariance matrix.
    
4.  Sort the eigenvalues and their corresponding eigenvectors in decreasing order. The eigenvector associated with the largest eigenvalue is the direction in which the data varies the most.
    
5.  Select the first N eigenvectors, where N is the number of dimensions you want to reduce your data to. These N vectors form a new 'basis' for your data.
    
6.  Finally, transform the original data into this new subspace (using matrix multiplication). The transformed data is now your dimensionality-reduced data.
    

In essence, eigenvectors and eigenvalues are the backbone of PCA. They are used to identify the new feature space (principal components) that maximizes variance and hence keeps as much information as possible from the original data while reducing its dimensionality.

### Code Example: 

```python 
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize PCA model
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_X = pca.fit_transform(X)

# Visualize the reduced data
colors = ['navy', 'turquoise', 'darkorange']
target_names = iris.target_names

plt.figure()

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(reduced_X[y == i, 0], reduced_X[y == i, 1], color=color, alpha=0.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.show()

```

In this code, we first load the Iris dataset. Then, we initialize the PCA model with `n_components=2`, meaning we want to reduce the data to 2 dimensions. We then fit the PCA model to the data and transform the data according to this model. This reduced data is then visualized using a scatter plot. Each of the three classes of the Iris dataset is plotted in a different color.

### Visual: 

![[PCA plot.png]]

### Why is PCA2 orthogonal to PCA1? 

You're correct that the second principal component (PCA2) is the direction with the second most variance. However, it's also true that PCA2 is orthogonal to PCA1. This might seem confusing, but let me explain.

Principal components are both orthogonal (perpendicular) to each other and identify the directions of maximum variance. This is possible because PCA1 identifies the direction of most variance, and then PCA2 identifies the direction of most variance that is orthogonal to PCA1.

The reason why orthogonality is a desirable property is because in statistics and data analysis, we often want our features to be independent of each other. If two features are highly correlated, they carry very similar information, and this redundancy is not efficient. Orthogonal vectors are uncorrelated, which implies that each principal component brings new, independent information to the analysis.

This is a result of the procedure used to calculate the principal components (the eigenvectors of the [[covariance matrix]]), which inherently produces orthogonal vectors. Therefore, each subsequent principal component is a direction that accounts for the highest possible variance in the data, under the constraint that it be orthogonal to the preceding components.