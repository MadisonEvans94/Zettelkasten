#incubator 
###### upstream: [[ML]]

### Details:

1.  **Overfitting**
    
    Overfitting refers to a situation where a model learns the training data too well. When a model is overfit, it has learned the noise and outliers in the training data to the point where it negatively impacts the model's ability to generalize to new, unseen data.
    
    -   Characteristics: High accuracy on training data, poor performance on testing/unseen data.
    -   Reasons: It typically happens when the model is too complex (e.g. too many features or too many layers in deep learning). Another reason could be when the model is trained for too long.
    -   Mitigation: Techniques like cross-validation, regularization, pruning, or early stopping can help prevent overfitting. Also, providing more training data can help, if available.
2.  **Underfitting**
    
    Underfitting is the opposite of overfitting. It occurs when a model is too simple to learn the underlying structure of the data. The model fails to capture important aspects and therefore also lacks in performance on unseen data.
    
    -   Characteristics: Poor performance on both training and testing data.
    -   Reasons: Typically due to a model being too simple (e.g. linear model for non-linear data), or the model has not been trained long enough.
    -   Mitigation: Use a more complex model, engineer better features, or train the model for a longer time. Additionally, reducing the amount of regularization can also help.
3.  **Bias-Variance Tradeoff**
    
    The bias-variance tradeoff is a fundamental concept in machine learning that describes the balancing act between overfitting and underfitting. It helps us understand the performance of the model and what can be done to improve it.
    
    -   **Bias**: Refers to the assumptions made by the model about the underlying data. *High bias can cause underfitting*, where the model is too simple and misses relevant relations between features and target outputs.
    -   **Variance**: Refers to how much the predictions would change if we trained it on a different dataset. High variance can cause overfitting, where the model is too sensitive and models the random noise in the training data.
    -   Tradeoff: The bias-variance tradeoff is the point where we are adding just enough complexity to the model that we minimize the total error. An optimal balance of bias and variance would never overfit or underfit the data.

![[Pasted image 20230614140251.png]]

![[Pasted image 20230614140350.png]]

[[Learning Curves (Validation vs Training)]]
### Examples (if any): 

