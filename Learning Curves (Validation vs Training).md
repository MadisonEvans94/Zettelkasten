#incubator 
###### upstream: [[ML]]

[Learning Curves in Machine Learning](https://www.youtube.com/watch?v=rZ_glsxvIAg&t=631s&ab_channel=WITSolapur-ProfessionalLearningCommunity)

[# Cornell CS 5787: Applied Machine Learning. Lecture 22. Part 1: Learning Curves](https://www.youtube.com/watch?v=lYAV5KNk_TY&t=1109s&ab_channel=VolodymyrKuleshov)

### Details:

Learning curves are great visual tools in machine learning for understanding how well your model is learning from your training data and how it generalizes to unseen data (validation data).

The concept of **learning curves** revolves around plotting training and validation performance as a function of the size of the training set, or alternatively, the training epochs.

Here's an overview of what we typically observe on these curves:

1.  **Training Curve**:
    
    -   The x-axis usually represents the number of training examples or **epochs**, and the y-axis shows the training error.
    -   At the beginning, when there are few training examples or the model is in its early epochs, the model can fit these quite well, resulting in low error, thus starting from the top of the graph.
    -   As more examples are added or more epochs are completed, it becomes harder for the model to fit all these examples perfectly, so the error typically goes up, meaning the curve goes upwards or flattens.
2.  **Validation Curve**:
    
    -   Similarly, the x-axis represents the number of training examples or epochs, but the y-axis in this case shows the validation error.
    -   At the beginning, with fewer training examples or early epochs, the model hasn't learned enough, so it performs poorly on the validation set, starting from the top.
    -   As the model learns from more examples or over more epochs, it generalizes better to unseen data, and the validation error goes down, meaning the curve goes downwards.

### Cross Validation: 

**Cross-validation** is a statistical method used to estimate the skill of machine learning models. It helps to understand how the results of your model will generalize to an independent, unseen data set. It's primarily used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice.

Here is a general overview:

1.  **K-Fold Cross Validation**: The most common form of cross-validation, and it involves the following steps:
    
    -   The data set is divided into k subsets (or 'folds'). For example, if k=5, you'd divide your data set into 5 subsets.
    -   The model is trained on k-1 (4 in our example) of those subsets.
    -   The remaining 1 subset is used to test the model.
    -   The process is repeated k times, with each of the k subsets used exactly once as the test set. The k results from the folds can then be averaged (or otherwise combined) to produce a single estimation.
2.  **Leave One Out Cross Validation (LOOCV)**: A variant of cross-validation where k (number of folds) equals the number of observations in the dataset. Each learning set is created by taking all the samples except one (which serves as the test set).
    
3.  **Stratified K-Fold Cross Validation**: It's a variant of k-Fold which should be used for imbalanced datasets where the minority class instances are less in number. In stratified k-fold, each fold contains roughly the same proportions of the different types of class labels.
    
4.  **Time Series Cross Validation**: A variant for time series data where data is divided in a way to avoid future data 'leaking' into the past. The training set grows over time.
    

Benefits of Cross-Validation:

-   It provides a more accurate measure of model prediction performance, as it does not rely on a single train/test split.
-   It makes efficient use of limited data, as it uses most of the data for fitting, and a subset for validation.
-   It gives your model the opportunity to train on multiple train-test splits, providing insight into how well the model will generalize to unseen data.

Drawbacks:

-   It can be computationally expensive, especially for large datasets and complex models.
-   It may not be appropriate for all kinds of data. For example, in time series data, traditional K-Fold cross-validation can cause data leakage.

### Takeaways: 
- High Bias = underfitting 
- High Variance (inability to converge) = overfitting 
- [[Accuracy Plots vs Loss Plots]]

