#incubator 
###### upstream: [[Linear Algebra]]

[# Eigenvectors and eigenvalues | Chapter 14, Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g&t=128s&ab_channel=3Blue1Brown)

### Definition:

1.  **Eigenvalue**: In linear algebra, an eigenvalue is a scalar associated with a linear system of equations (usually represented as a matrix), which, when multiplied by a particular vector (called an eigenvector), yields that same vector multiplied by the eigenvalue. In other words, it's a value such that if you multiply the matrix by a certain vector, you get the same vector times the eigenvalue.
2. **Eigenvector**: An eigenvector is that particular vector associated with a matrix and an eigenvalue such that when the matrix multiplies the eigenvector, the result is the eigenvector times the eigenvalue. It can be thought of as a vector that maintains its direction, only being scaled by the multiplication.

if you have a matrix A and an eigenvector v of that matrix, then when you multiply A by v, the result is the same as multiplying v by a scalar λ (the eigenvalue). Mathematically, this is expressed as:

*Av = λv*

Where:

-   *A* is the matrix,
-   *v* is the eigenvector, and
-   *λ* is the eigenvalue.

![[Screen Shot 2023-06-15 at 7.58.21 AM.png]]

### What does this mean visually?: 

basically, imagine stretching and skewing a 2d plane via linear combinations. The **eigenvector** is the line on that 2d plane that doesn't change in direction, only in scale. And the **eigenvalue** is the factor by which an eigenvector is squished or stretched. 

![[Pasted image 20230615080033.png]]

### Where else are eigenvectors used?: 

Eigenvectors and eigenvalues have a wide range of applications across various disciplines. Here are some of them:

1.  **Quantum Mechanics**: The Schrödinger equation, one of the fundamental equations in quantum mechanics, is an eigenvalue equation and its solutions provide the possible states a quantum system can be in.
    
2.  **Computer Vision**: Eigenvectors are used in facial recognition technologies. In [[Computer Vision]] technique is called Eigenfaces.
    
3.  **Differential Equations**: Eigenvectors and eigenvalues are used in the solutions of many differential equations.
    
4.  **Data Science**: In addition to [[Principal Component Analysis (PCA)]], they are used in Singular Value Decomposition, Latent Semantic Analysis, and other [[dimensionality reduction techniques]].
    
5.  **Graph Theory**: Eigenvectors and eigenvalues are used in [[Graph Theory]] in the analysis of network structures, including social networks, web page rankings (Google's PageRank algorithm), etc.
    
6.  **Vibration Analysis**: In physics and engineering, the vibration of a system (like a building or a bridge) can be analyzed by looking at the eigenvectors and eigenvalues of the equations of motion. See [[vibration analysis]] for more 
    
7.  **Control Theory**: According to [[Control Theory]] control systems, stability of the system can be analyzed using eigenvalues.
    
8.  **Machine Learning**: They are used in various [[ML]] algorithms, including clustering and linear discriminant analysis.
    
9.  **Finance**: They are used in [[portfolio optimization]] and other statistical analyses.


### Examples (if any): 

