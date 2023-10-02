#seed 
upstream:

---

**video links**: 

---

# Brain Dump: 




--- 

Firstly, you need to import the necessary libraries and set the **hyperparameters**. 

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
```

## Components Breakdown

- **Model (`Net`)**: This is the neural network you're training. It's defined as a class that inherits from `nn.Module`.

> see [[Pytorch nn.Module]] for more

- **Loss Function (`criterion`)**: This function computes the difference between the predicted and true labels. It's a measure of how well your model is performing.
> see [[Pytorch Criterion]] for more

- **Optimizer (`optimizer`)**: This is the algorithm you'll use to update your model's parameters based on the computed loss. Optimizers like SGD, Adam, or RMSprop are commonly used.
> see [[Pytorch Optimizers]] for more

- **Data Loaders (`trainloader`)**: These are PyTorch utilities for loading data in mini-batches, which are crucial for training large datasets and optimizing the model effectively. 
> see [[Pytorch Data Loaders]] for more 


## Process Flow 

- **Forward Pass**: In this step, the model computes an output based on the input data.

- **Loss Calculation**: The loss function computes how far off our predictions are from the actual labels.

- **Backward Pass (`loss.backward()`)**: This computes the gradient of the loss with respect to the model parameters. Essentially, it's how much the model parameters contributed to the loss.

- **Optimizer Step (`optimizer.step()`)**: The optimizer updates the model parameters based on the gradients computed during the backward pass.

- **Zero the Gradients (`optimizer.zero_grad()`)**: Before computing the gradients for the next mini-batch, the existing gradients are zeroed out to ensure that gradients don't accumulate.

