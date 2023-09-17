#evergreen1
###### upstream: [[Deep Learning]]

### Description:

Let's start with the forward pass:

1.  **Forward Pass**: During the forward pass, the input sequence is fed into the network one step (timestep) at a time. At each timestep, the RNN takes the current input and the previous hidden state, applies the recurrent weights, and generates an output and a new hidden state. This process repeats for each timestep in the sequence.

Now, let's proceed to the backpropagation:

2.  **Backpropagation (BPTT)**: Like standard backpropagation, BPTT starts with calculating the error of the output. The error is typically the difference between the predicted output and the actual output, determined by the loss function. The BPTT algorithm then propagates this error backward through the network to adjust the weights. However, since the same weights are used at each timestep in the sequence, the error at each timestep is affected by the error at the next timestep. Thus, the error is propagated backward through each timestep in the sequence, which gives this process its name: Backpropagation Through Time.
    
3.  **Compute Gradients**: During the backpropagation, we compute gradients of the loss with respect to the weights of the network. In an RNN, due to shared weights across timesteps, the gradient at each timestep is summed up.
    
4.  **Update Weights**: Once the gradients have been computed, they are used to adjust the weights in the network. This is typically done with an optimization algorithm like Stochastic Gradient Descent (SGD) or variants of it like Adam or RMSprop.
    
5.  **Repeat the Process**: Steps 1-4 are repeated for multiple epochs, or complete passes through the training dataset, until the network's weights are optimized to minimize the loss.

### A Pseudocode Example:

let's walk through an example with a sequence of three timesteps. Let's denote:

-   `X(t)` as the input at timestep `t`,
-   `H(t)` as the hidden state at timestep `t`,
-   `Y(t)` as the output at timestep `t`,
-   `Y_hat(t)` as the predicted output at timestep `t`, and
-   `L(t)` as the loss at timestep `t`.

Let's also assume that the model's weights are `U` (for input-to-hidden), `V` (for hidden-to-output), and `W` (for hidden-to-hidden).
Here's the flow of calculations for the backpropagation:

1.  **Calculate the Loss**: First, the loss is computed for each timestep. This is typically done using a loss function appropriate for the task. For instance, for a regression task, you might use [[Mean Squared Error (MSE)]], which calculates the square of the difference between the predicted output `Y_hat(t)` and the actual output `Y(t)`.
    
2.  **Backpropagate Through Time**: Starting from the final timestep (t=3 in this case), the gradient of the loss with respect to the output `dL/dY_hat(3)` is computed. This value is dependent on the choice of loss function.
    
3.  **Compute Gradients for the Output Layer**: The gradients of the loss with respect to the weights `V` are computed as `dL/dV = dL/dY_hat * dY_hat/dV = dL/dY_hat * H(3)`.
    
4.  **Propagate Error to the Hidden Layer**: The error is then propagated to the hidden layer. The gradient of the loss with respect to the hidden state `dL/dH(3)` is calculated as the sum of the direct effect of `H(3)` on `L(3)` and the effect propagated back from the future. However, since t=3 is the last timestep, there is no future effect. So `dL/dH(3) = dL/dY_hat(3) * dY_hat/dH(3) = dL/dY_hat(3) * V`.
    
5.  **Compute Gradients for the Hidden Layer**: The gradients of the loss with respect to the weights `U` and `W` are computed. For `U`, it's `dL/dU = dL/dH(3) * dH/dU = dL/dH(3) * X(3)`. For `W`, it's `dL/dW = dL/dH(3) * dH/dW = dL/dH(3) * H(2)`.
    
6.  **Repeat for t=2 and t=1**: Steps 2-5 are repeated for t=2 and t=1. Note that when calculating `dL/dH(t)`, the error from the future, `dL/dH(t+1) * dH(t+1)/dH(t) = dL/dH(t+1) * W`, needs to be added.
    
7.  **Weights Update**: Once the gradients for `U`, `V`, and `W` have been summed over all timesteps, the weights are updated using a learning rule, typically a form of Stochastic Gradient Descent (SGD).
    

Remember, in the case of RNNs, the same weights (`U`, `V`, and `W`) are used at every timestep. Hence, we sum up the gradients from all timesteps before performing the weights update.

This process is repeated for each sequence in your training dataset until the model's performance on the loss function is satisfactory.


### Underlying Question(s): 

- *Why are the weight vectors the same for each time step?*: 

Recurrent Neural Networks (RNNs) use the same weights across each timestep due to their very design principle. The key idea behind an RNN is to make use of sequential information by having a "memory" (hidden state) that captures information about what has been seen so far in the sequence.

In an RNN, the same weight parameters are shared across all timesteps. This property is also known as **parameter sharing**. This is a crucial feature because it enables the model to generalize across different lengths of sequences and at different positions within the sequence. Regardless of the input's position in time, the model applies the same transformation (via weights) to each input at each timestep.

This design greatly reduces the complexity of the model. Instead of learning a new set of parameters for every timestep, the model only has to learn one set of parameters. It can thus manage sequences of varying lengths and generalize from the learned sequence patterns, which can occur at any point within the sequence.

