
Topics to Study:
#incubator 

###### upstream: [[Deep Learning]]

## Computation Graph 

![[RNN computation graph.pdf]]

### Feed Forward Pass 
-  Dot product between `X` and input weight matrix `U`
- Dot product between initial hidden state `h_t` and hidden state weight matrix `W`
- Summation between the output of both dot product operations and bias term `b`, resulting in intermediate output `a_t`
- Pass `a_t` into the activation node `g` which uses the [[Hyperbolic Tangent]] `tanh` as activation function, resulting in output `h_t`
- Dot product of `h_t` and dense layer weight matrix `V`, added with bias `C`, outputting `O_t`
- Pass `O_t` to softmax, which outputs the prediction `y_hat_t`
- Use cross entropy loss to calculate `L_t`

> In summary, we are multiplying the input with a weight matrix, multiplying the hidden state with a weight matrix, adding these two products together with a bias term, passing this through a hyperbolic tangent activation function, passing this output through a dense layer, computing softmax, and then finding the cross entropy loss
## Vanilla Implementation Using Pytorch 

```python 
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        
        # Define the weights and biases for the input to hidden state transformation
        self.U = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.b = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Define the weights and biases for the hidden state to next hidden state transformation
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
        # Define the weights and biases for the hidden state to output transformation
        self.V = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.c = nn.Parameter(torch.randn(1, output_dim))
        
    def forward(self, x, h_prev):
        # Compute the current hidden state
        a = torch.mm(x, self.U) + torch.mm(h_prev, self.W) + self.b
        h = torch.tanh(a)
        
        # Compute the output
        o = torch.mm(h, self.V) + self.c
        y_hat = F.softmax(o, dim=1)
        
        return y_hat, h

# Test the VanillaRNN
input_dim = 10
hidden_dim = 20
output_dim = 5
batch_size = 8

# Dummy input and initial hidden state
x = torch.randn(batch_size, input_dim)
h_prev = torch.zeros(batch_size, hidden_dim)

model = VanillaRNN(input_dim, hidden_dim, output_dim)
y_hat, h = model(x, h_prev)

print("Output shape:", y_hat.shape)
print("Hidden state shape:", h.shape)

```