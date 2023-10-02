
#evergreen1 
upstream: [[Deep Learning]] 

---

**video links**: 

---

Understanding the terms "Epochs," "Batches," and "Iterations" is crucial for anyone diving into the world of machine learning and deep learning. These terms often appear when discussing the training process of a neural network or any machine learning algorithm. This document aims to provide a comprehensive understanding of these terms, both formally and through abstract analogies.

## Epochs

### Formal Explanation

An epoch is one complete forward and backward pass of all the training examples. In simpler terms, an epoch is one cycle through the full training dataset. Usually, training a neural network takes more than a few epochs.

### Abstract Analogy

Imagine you're studying for an exam by going through a textbook. Reading the textbook from cover to cover once would be equivalent to one epoch. You'll likely read it multiple times (multiple epochs) to fully understand and memorize the material.

---

## Batches

### Formal Explanation

A batch is a subset of the training dataset. It's a chunk of data that you feed into the neural network for training in one iteration. The size of the batch is commonly denoted as "batch size."

### Abstract Analogy

Let's say you're eating a bag of chips while watching a movie. You don't eat all the chips at once; you grab a handful (a batch) at a time. Each handful of chips represents a batch of data that you process (eat) before grabbing the next handful.

### Reason For Using Batches 

>So is the idea to obtain an average loss per batch and use that loss for the weight updates? In other words, if my batch size is 64, then I'll calculate the loss for each of the 64 samples, and that average loss will be what is used for the update as opposed to updating 64 times?

Yes, exactly. The idea behind using mini-batch gradient descent (as opposed to stochastic gradient descent) is to compute the gradient of the loss function with respect to the parameters for each mini-batch. When you have a batch size of 64, you'll forward-propagate all 64 samples through the network, compute the loss for each sample, and then average these losses to get the average loss for the batch. This average loss is then backpropagated through the network to update the weights.

#### Mathematical Explanation

Let \( L_i \) be the loss for the \( i^{th} \) sample in the batch. The average loss \( \bar{L} \) for the batch of size \( N \) (64 in your case) is:

$$
\bar{L} = \frac{1}{N} \sum_{i=1}^{N} L_i
$$

This average loss \( \bar{L} \) is then used to compute the gradients:

$$
\frac{\partial \bar{L}}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial L_i}{\partial w}
$$

The gradients are then used to update the weights $( w )$ of the network.

#### Layman's Explanation

Imagine you're a teacher grading a batch of 64 exams. Instead of changing your grading criteria after each individual exam, you grade all 64, average the scores, and then adjust your expectations (weights) based on this average performance. This way, your adjustments are smoother and less reactive to any one student's performance, making your grading more stable and generalized.

#### Advantages

1. **Computational Efficiency**: Modern hardware architectures are optimized for batch operations, making it more efficient than updating per sample.
  
2. **Generalization**: Averaging over a batch aims to provide a more general update to the model, reducing the noise in the weight updates.

3. **Stability**: The average is less sensitive to outliers, providing a more stable and reliable gradient for weight updates.

By using the average loss per batch for weight updates, you're essentially smoothing out the learning process, making it more stable and efficient.

---

## Iterations

### Formal Explanation

An iteration is one update of the model's weights, usually performed using a subset of the training data (a batch). The number of iterations is the number of batches needed to complete one epoch.

$$
\text{Number of Iterations per Epoch} = \frac{\text{Total Number of Training Samples}}{\text{Batch Size}}
$$

### Abstract Analogy

Continuing with the textbook analogy, an iteration would be like reading a single chapter of the textbook and updating your understanding (or taking notes). You'll go through multiple chapters (iterations) to complete reading the textbook once (one epoch).

### How Many Iterations per Epoch? 

>So let's say that an entire training set is 1000 samples, and a batch is 100 samples. Does that mean that each epoch will have 10 iterations, and by definition, this means 10 times that the weights are updated? Subsequently, If I were to train for 10 epochs, then that means that the model will undergo 100 updates?

Yes, you've got it exactly right. In the scenario you've described:

- With a training set of 1000 samples and a batch size of 100 samples, each epoch would indeed consist of 10 iterations.
  
- Each iteration would involve processing one batch of 100 samples, computing the average loss for that batch, and then updating the model's weights based on that average loss.

- If you were to train the model for 10 epochs, then the model would undergo 100 weight updates in total.

### Mathematical Formalization

The number of iterations \( I \) per epoch is calculated as:

$$
I = \frac{\text{Total Number of Training Samples}}{\text{Batch Size}} = \frac{1000}{100} = 10
$$

For 10 epochs, the total number of weight updates \( U \) would be:

$$
U = I \times \text{Number of Epochs} = 10 \times 10 = 100
$$

### Layman's Analogy

Imagine you're reading a 1000-page book in chunks of 100 pages at a time. Each chunk you read (and summarize in your head) would be an iteration. Reading through all 1000 pages once would be an epoch. If you decide to read the book 10 times to really absorb the material, you'd have gone through 100 iterations (or 100 chunks) in total.

---

## Relationship to a Single Training Sample

### Formal Explanation

A single training sample is the smallest unit of your data that you use to update your model. When you're using a batch size of 1 (often called "online learning" or "stochastic gradient descent"), each training sample becomes its own batch, and each iteration updates the model based on that single training sample.

### Abstract Analogy

Think of a single training sample as a single chip in your bag of chips. If you were to eat one chip at a time, each chip would be like a batch, and eating it would be an iteration. In this case, you'd update your "fullness level" with each individual chip.

---

## Summary

- **Epoch**: One complete cycle through the entire training dataset.
  - **Analogy**: Reading a textbook from cover to cover once.
  
- **Batch**: A subset of the training dataset.
  - **Analogy**: A handful of chips from a bag.
  
- **Iteration**: One update of the model's weights, usually using a batch of data.
  - **Analogy**: Reading a single chapter of a textbook and updating your understanding.

- **Single Training Sample**: The smallest unit of data.
  - **Analogy**: A single chip in the bag.

By understanding these terms and their relationships, you'll have a clearer picture of what happens during the training process of a machine learning model. Whether you're sober or under the influence of marijuana, these concepts should now be as clear as a sunny day!



