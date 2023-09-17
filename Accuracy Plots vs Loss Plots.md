#evergreen1 
###### upstream: 

### Underlying Question: 

*I've seen 2 ways to plot learning curves: 1 way plots epochs against accuracy, and the other plots epochs against loss. What is the difference between each method of plotting? When should I use one method over the other? What is the industry standard method to use? *

### Description: 

![[Pasted image 20230614141822.png]]

The choice between plotting epochs against accuracy or against loss depends largely on what you are specifically interested in monitoring and evaluating in your model during the training process. Let's go over each one:

1.  **Plotting Epochs Against Loss**: The loss function (also called cost function or objective function) is what the model tries to minimize during training. The value of the loss function represents the discrepancy between the predictions of the model and the actual data.
    
    -   Pros: Loss is a direct measure of how well the model's predictions match the data, irrespective of the specific task (classification, regression, etc.).
    -   Cons: While it's informative, it may not directly align with your end goal. For example, you might ultimately care about accuracy or some other business-specific performance measure rather than the mean squared error.
    -   When to use: This is particularly useful when you're interested in the internal optimization of the model. You'll almost always want to monitor this to ensure that training is decreasing the loss over time.
2.  **Plotting Epochs Against Accuracy**: Accuracy is a specific performance measure for classification problems. It represents the proportion of correct predictions over total predictions.
    
    -   Pros: Accuracy is intuitive and easy to interpret. It directly tells you what fraction of the predictions are correct.
    -   Cons: It may not be the best performance measure for imbalanced datasets. It treats all classes equally, so a high accuracy might be misleading if one class is significantly more prevalent. In such cases, other metrics like precision, recall, F1-score, or AUC-ROC may be more informative.
    -   When to use: This is more task-specific and should be used when your primary concern is the model's ability to correctly classify data points, and the classes are approximately balanced.

As for an industry standard, both are very common, and which one to use depends more on the context and the specific objectives of the training. It is often helpful to monitor both during training. This gives you a view of both how well optimization is proceeding (via the loss) and how well the model is achieving the ultimate objective of accurate prediction (via the accuracy or other relevant metrics).