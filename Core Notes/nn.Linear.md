#incubator 
upstream: [[Pytorch]]

---

**links**: 

[pytorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

---

# Brain Dump: 


--- 


## Overview

Applies a linear transformation to the incoming data: 

$$y = xA^T + b$$

When you instantiate a layer using `nn.Linear`, it automatically creates:

1. A weight tensor (`weight`): This tensor has dimensions `[out_features, in_features]`.
2. A bias tensor (`bias`): This tensor has dimensions `[out_features]`.

Both the `weight` and `bias` tensors are initialized with values from a uniform distribution by default (you can change the initialization if needed). Moreover, these tensors are registered as trainable parameters of the model, meaning they will be updated during back-propagation when training the neural network.
### Parameters

**in_features** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.12)")) – size of each input sample

**out_features** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.12)")) – size of each output sample

**bias** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.12)")) – If set to `False`, the layer will not learn an additive bias. Default: `True`


> A densely connected layer that has an input size of 4 and output size of 3 would look like the following: 

![[Densely Connected Layer.png]]

### Variables

**weight** ([_torch.Tensor_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) – the learnable weights of the module of shape (`out_features`, `in_features`). The values are initialized from $U(-\sqrt{k}, \sqrt{k})$ where k = 1/`in_features`

**bias** – the learnable bias of the module of shape (out_features)(out_features). If `bias` is `True`, the values are initialized from

### Example 

1. Input Layer: 5 input nodes.
2. Hidden Layer: 3 nodes.
3. Output Layer: 2 nodes (let's assume a binary classification task).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        # First layer: 5 input nodes -> 3 hidden nodes
        self.fc1 = nn.Linear(in_features=5, out_features=3)

        # Second layer: 3 hidden nodes -> 2 output nodes
        self.fc2 = nn.Linear(in_features=3, out_features=2)

    def forward(self, x):
        # Pass input through the first layer and apply ReLU activation
        x = F.relu(self.fc1(x))

        # Pass through the second layer
        x = self.fc2(x)

        # Apply softmax for classification
        x = F.softmax(x, dim=1)

        return x

# Instantiate the network
net = SimpleNN()

# Create a dummy input tensor of size (batch_size=1, input_features=5)
input_tensor = torch.randn(1, 5)

# Forward pass
output = net(input_tensor)
print(output)
```

- We've defined a simple neural network with 2 layers using `nn.Linear`.
- We used the ReLU activation function for the hidden layer.
- The output layer uses a softmax activation, which is typical for classification tasks to get probabilities for each class.

You can further train this network using a dataset and an appropriate loss function (like cross-entropy for classification tasks).

You can access the weight tensors directly, if you'd like. For instance, given an `nn.Linear` layer named `fc1`:

- `fc1.weight` would give you the weight tensor.
- `fc1.bias` would give you the bias tensor.

You can also check the shape of these tensors, which can help in understanding the connections and transformations the layer is set up to perform. For instance, `fc1.weight.shape` would return a torch.Size indicating the dimensions of the weight tensor.