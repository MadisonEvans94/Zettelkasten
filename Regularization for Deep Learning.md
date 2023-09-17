#incubator 
###### upstream: [[Deep Learning]]

[# Machine Learning Tutorial Python - 17: L1 and L2 Regularization | Lasso, Ridge Regression](https://www.youtube.com/watch?v=VqKq78PVO9g&ab_channel=codebasics)

### Description: 

**What is Regularization?**

When you're training a deep learning model, your goal is to have a model that not only performs well on the training data, but also generalizes well to new, unseen data (the test data). However, sometimes a model may fit the training data too well and capture the noise along with the underlying pattern, a situation known as overfitting. When this happens, the model will likely perform poorly on new data because it's essentially "memorized" the training data instead of learning the actual patterns.

Regularization is a technique used to prevent this overfitting. The idea is to add a penalty to the loss function, a measure of how well the model fits the data, to discourage the model from learning a too complex model and hence to make it more generalized.

**Why and Where is it used?**

Regularization is used during the training process of a deep learning model to encourage the model to keep the weights small. By doing this, the model becomes less sensitive to small changes in the input, thus reducing overfitting. Regularization helps the model generalize better from the training data to unseen data, improving the model's predictive performance on new data.

There are several types of regularization techniques commonly used in deep learning:

1.  **L1 and L2 regularization:** These methods add a penalty to the loss function based on the size of the weights. L1 regularization adds a penalty proportional to the absolute value of the weights (which can result in sparse weights), while L2 regularization adds a penalty proportional to the square of the weights.
    
2.  **Dropout:** This is a very popular regularization method for neural networks. During training, dropout randomly "drops out" or deactivates a proportion of the neurons in a layer, meaning they don't contribute to the forward pass or the backward pass for that specific training iteration. This prevents the model from relying too heavily on any one neuron and encourages it to learn more robust, general features.
    
3.  **Early stopping:** During the training process, we can monitor the model's performance on a separate validation set. If we notice that the performance on the validation set is starting to get worse, even though the performance on the training set is still improving (a sign of overfitting), we stop the training. This is known as early stopping.
    

To summarize, *regularization is like a guardrail that prevents the model from overfitting by discouraging it from learning overly complex patterns in the training data that might not generalize well to new data. Different types of regularization methods provide different ways of achieving this goal.*

### Example:

Suppose we have a dataset with 100 features and we're doing a binary classification (classifying data into one of two classes). Here's how you might use L2 regularization and dropout:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

model = Sequential()

# Input layer
model.add(Dense(64, input_dim=100, activation='relu'))

# Hidden layer with L2 Regularization
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))

# Add dropout of rate 0.5 to the next layer
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

```

In the above code:

1.  `kernel_regularizer=l2(0.01)` adds L2 regularization to the weights (kernels) of the second layer, where `0.01` is the regularization factor (also known as lambda).
    
2.  `Dropout(0.5)` applies dropout to the input of the subsequent layer. The rate `0.5` means approximately 50% of neurons in the preceding layer will be turned off during each training epoch.
    

Please note, you will need to adjust the model parameters (like the number of layers, the number of neurons in each layer, the regularization factor, and the dropout rate) based on your specific use case for optimal performance. And of course, you'll need to actually fit your model using your data with `model.fit(x_train, y_train, epochs=10, batch_size=32)`.

Also, always remember to split your data into training and validation (or test) sets and monitor the model's performance on both sets during the training process. This can help you spot if your model is starting to overfit the training data.