
Topics to Study:
#evergreen1 

###### upstream: [[Deep Learning]]

![[repeating_rnn_modules.png]]

## Computation Graph 

![[RNN computation graph.pdf]]


### Feed Forward Pass 
- Dot product between `X` and input weight matrix `U`
- Dot product between initial hidden state `h_t` and hidden state weight matrix `W`
- Summation between the output of both dot product operations and bias term `b`, resulting in intermediate output `a_t`
- Pass `a_t` into the activation node `g` which uses the [[Hyperbolic Tangent]] `tanh` as activation function, resulting in output `h_t`
- Dot product of `h_t` and dense layer weight matrix `V`, added with bias `C`, outputting `O_t`
- Pass `O_t` to softmax, which outputs the prediction `y_hat_t`
- Use cross entropy loss to calculate `L_t`

For each element in the input sequence, each layer computes the following function:
$$h_t = tanh(x_tW^T_{ih} + b_{ih} + h_{t-1} W^T_{hh} + b_{hh})$$
where $h_t$ is the hidden state at time $t$, $x_t$ is the input at time $t$, and $h_{(t-1)}$ is the hidden state of the previous layer at time $t-1$ or the initial hidden state at time $0$ .

> In summary, we are multiplying the input with a weight matrix, multiplying the hidden state with a weight matrix, adding these two products together with a bias term, passing this through a hyperbolic tangent activation function, passing this output through a dense layer, computing softmax, and then finding the cross entropy loss

---

## Pytorch 

PyTorch offers a built-in RNN module, `torch.nn.RNN`, which is easy to use and highly optimized.

#### Parameters:

1. **input_size**: The number of expected features in the input `x`.
2. **hidden_size**: The number of features in the hidden state `h`.
3. **num_layers**: - Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two RNNs together to form a [[Stacked RNN]], with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
4. **nonlinearity**: The non-linearity to use. Can be either 'tanh' (default) or 'relu'.
5. **bias**: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: `True`.
6. **batch_first**: If `True`, then the input and output tensors are provided as (batch, seq, feature). Default: `False`.
7. **dropout**: Fraction of the inputs to be zeroed (default is 0). Useful for regularization.
8. **bidirectional**: If `True`, becomes a bidirectional RNN. Default: `False`.

#### Output:

If we provide an input of shape `(seq_len, batch, input_size)`, it returns:

1. **output**: tensor of shape `(seq_len, batch, num_directions * hidden_size)`.
2. **hn**: tensor of shape `(num_layers * num_directions, batch, hidden_size)`.

#### Example: 

> Here's how you can create and use a simple RNN using the built-in RNN module in PyTorch:

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleRNN, self).__init__()
        
        # Define the RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        # Initial hidden state set to zero
        h0 = torch.zeros(1, x.size(0), hidden_dim)
        
        # RNN forward pass
        out, _ = self.rnn(x, h0)
        
        # Pass the last output of the RNN to the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out

# Test the SimpleRNN
input_dim = 10
hidden_dim = 20
output_dim = 5
batch_size = 8
sequence_length = 3

# Dummy input
x = torch.randn(batch_size, sequence_length, input_dim)

model = SimpleRNN(input_dim, hidden_dim, output_dim)
output = model(x)

print("Output shape:", output.shape)
```

In the above example, we've added a fully connected (linear) layer after the RNN to map the hidden state to the desired output dimension. Note that we've used the `batch_first=True` argument, so the input tensor has shape `(batch_size, sequence_length, input_dim)`.