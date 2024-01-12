#evergreen1 
###### upstream: [[Deep Learning]]

[Blog: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---


## How it Works

All [[RNN (Recurrent Neural Network)]] have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. 

> Note: the diagram doesn't show it, but the combined hidden state and input needs to pass through a linear layer before going through the tanh activation. see [[Hidden State vs Output in RNN]]

![[repeating_rnn_modules.png]]

**LSTM** networks also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![[repeating_lstm_model.png]]


![[diagram_components.png]]

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

![[longterm_rail.png]]

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

![[Screen Shot 2023-10-24 at 4.21.15 PM.png]]

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

An LSTM has three of these gates, to protect and control the cell state.

- Three gates: [[Forget Gate, Input Gate, and Output Gate]]
- Two Rails: Long-term and short-term memory 

### Step By Step Walkthrough: 

#### Forget Gate 

> determines how much of the old cell state to keep. It can erase parts of the cell state that are no longer needed.

First we need to decide which information to throw away from the cell state. This is made by a sigmoid layer called the *forget gate*. It takes $h_{t-1}$ and $x_{t}$ and outputs a number between 0 and 1 for each value in cell state $C_{t-1}$ . A 1 represents "completely keep this", and a 0 means "completely forget this". 

![[forget_gate.png]]

$$f_t = {\sigma}(W_f {\cdot}[h_{t-1}, x_t] + b_f)$$
#### Input Gate 

> determines how much of the new input will be added to the cell state. Essentially, it's like a valve deciding how much new information to store in the cell.

Next, we need to figure out how much of the new information we're going to keep and add to the cell state. A sigmoid layer called the *input gate* determines which values to update. A $tanh$ layer creates a vector of new candidate values $\tilde{C}$ that could be added to the state. 

![[input_gate.png]]

$$i_t = {\sigma}(W_i{\cdot}[h_{t-1}, x_t] + b_i$$
$$\tilde{C}_t = tanh(W_C{\cdot}[h_{t-1}, x_t] + b_C$$

It’s now time to update the old cell state, $C_{t−1}$, into the new cell state $C_t$. The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by $f_t$, forgetting the things we decided to forget earlier. Then we add $i_t  {\ast}  \tilde{C}_t$. This is the new candidate values, scaled by how much we decided to update each state value 

![[input_gate_combination.png]]

#### Output Gate 

> determines what part of the cell state will be exposed as the output for the next time step

Finally we need to decide what we're going to output. This will be a filtered version of our cell state. To do this, we run a sigmoid layer called the *output gate* which decides what parts of the cell state we're going to output. Then we put the cell state through tanh to squash the values between -1 and 1, and multiply it by the output of the sigmoid gate. 

![[output_gate.png]]

$$o_t = {\sigma} (W_o[h_{t-1}, x_t] + b_o) $$
$$h_t = o_t{\ast}tanh(C_t)$$

Certainly! I'll add a "Weights" section to your notes that describes the weight matrices involved in an LSTM cell. You can insert this section after the "Step By Step Walkthrough" and before any conclusion or summary you may have.

---

### Weights

In an LSTM cell, the behavior of gates and the cell state update is governed by a series of weight matrices and bias vectors. These weights are parameters learned during the training process and are fundamental to the LSTM's ability to store and manage information over time.

There are four main weight matrices (and their corresponding biases) for each of the gates and the cell state, which interact with the input $( x_t )$ and the previous hidden state $( h_{t-1} )$.  The weight matrices and biases are as follows:

#### Forget Gate Weights
- $( W_f )$: Weight matrix for the forget gate, applied to the input and previous hidden state.
- $( b_f )$: Bias for the forget gate.

#### Input Gate Weights
- $( W_i )$: Weight matrix for the input gate, applied to the input and previous hidden state.
- $( b_i )$: Bias for the input gate.

#### Cell State Weights
- $( W_C )$: Weight matrix for creating the cell state candidate, applied to the input and previous hidden state.
- $( b_C )$: Bias for the cell state candidate.

#### Output Gate Weights
- $( W_o )$: Weight matrix for the output gate, applied to the input and previous hidden state.
- $( b_o )$: Bias for the output gate.

Each gate has two sets of weights and biases: one that interacts with the input \( x_t \) and another that interacts with the previous hidden state \( h_{t-1} \). For computational efficiency, these weights are often concatenated into larger matrices:

- $( W_x = [W_{xf}, W_{xi}, W_{xo}, W_{xc}] )$: Concatenated weight matrix for the input.
- $( W_h = [W_{hf}, W_{hi}, W_{ho}, W_{hc}] )$: Concatenated weight matrix for the previous hidden state.
- $( b = [b_f, b_i, b_o, b_C] )$: Concatenated bias vector.

During the forward pass, the LSTM cell calculates the following vectors:

- $( f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) )$: The forget gate vector.
- $( i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) )$: The input gate vector.
- $( \tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) )$: The cell state candidate vector.
- $( o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) )$: The output gate vector.

These calculations result in the LSTM cell's ability to maintain a long-term memory, decide what information to retain or discard, and what to output as the current hidden state $( h_t )$ and the current cell state $( c_t )$.

---

## Example Pytorch Implementation 

### **1. Import Necessary Libraries**
We begin by importing the necessary libraries. For our LSTM implementation, we'll need PyTorch and its neural network module.

```python
import torch
import torch.nn as nn
```

---

### **2. Define the Vanilla LSTM Model**
Our VanillaLSTM class will inherit from PyTorch's `nn.Module`. Within this class, we will define the necessary components for the LSTM gates and the forward pass logic.

```python
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaLSTM, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		
		# i_t: input gate
		
		self.Wii = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.bii = nn.Parameter(torch.Tensor(hidden_size))	
		self.Whi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))	
		self.bhi = nn.Parameter(torch.Tensor(hidden_size))
		
		# f_t: forget gate
		self.Wif = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.bif = nn.Parameter(torch.Tensor(hidden_size))
		self.Whf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.bhf = nn.Parameter(torch.Tensor(hidden_size))
		
		# g_t: cell gate
		self.Wig = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.big = nn.Parameter(torch.Tensor(hidden_size))
		self.Whg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.bhg = nn.Parameter(torch.Tensor(hidden_size))
		
		# o_t: output gate
		self.Wio = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.bio = nn.Parameter(torch.Tensor(hidden_size))
		self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.bho = nn.Parameter(torch.Tensor(hidden_size))
		
		self.init_hidden()
```


> The naming convention in the LSTM code for weights and biases follows this pattern:

1. Start with `W` for weights.
2. The next letter denotes the source of the input:
   - `i` for input (`x_t`)
   - `h` for previous hidden state (`h_{t-1}`)
3. The final letter specifies the gate:
   - `i` for input gate
   - `f` for forget gate
   - `g` for cell gate (candidate cell state)
   - `o` for output gate

For biases, the naming is similar, but without the starting `W`.

Examples:
- `Wii`: Weight matrix for input (`x_t`) to input gate.
- `Whi`: Weight matrix for hidden state (`h_{t-1}`) to input gate.
- `Wif`: Weight matrix for input (`x_t`) to forget gate.
- `bii`: Bias for input (`x_t`) affecting the input gate.
- `bhi`: Bias for hidden state (`h_{t-1}`) affecting the input gate.

This convention helps distinguish the purpose and source of each weight and bias in the LSTM architecture.


---

### **3. Forward Pass**
In the forward method, we define how the LSTM computes its output given an input `x` and the previous states `(h_prev, C_prev)`.


```python 
# The @ denotes matrix multiplication in PyTorch. Shorthand for a dense layer 

for t in range(seq_size):

    # Extract the input for the current timestep.
    x_t = x[:, t, :]

    # Input gate: Compute using dense layer outputs of `x_t` and `h_t`, then apply sigmoid.
    i_t = torch.sigmoid(x_t @ self.Wii + self.bii +
                        h_t @ self.Whi + self.bhi)

    # Cell update (often referred to as the "gate gate"): Compute using dense layer outputs of `x_t` and `h_t`, then apply tanh.
    g_t = torch.tanh(x_t @ self.Wig + self.big +
                     h_t @ self.Whg + self.bhg)

    # Forget gate: Compute using dense layer outputs of `x_t` and `h_t`, then apply sigmoid.
    f_t = torch.sigmoid(x_t @ self.Wif + self.bif +
                        h_t @ self.Whf + self.bhf)

    # Output gate: Compute using dense layer outputs of `x_t` and `h_t`, then apply sigmoid.
    o_t = torch.sigmoid(x_t @ self.Wio + self.bio +
                        h_t @ self.Who + self.bho)

    # UPDATE THE CELL STATE
    c_t = f_t * c_t + i_t * g_t
    
    # UPDATE THE HIDDEN STATE
    h_t = o_t * torch.tanh(c_t)

return (h_t, c_t)

```

> If there's one thing you get out of this, it's the following: 

```python 
	# UPDATE THE CELL STATE
    c_t = f_t * c_t + i_t * g_t
    
    # UPDATE THE HIDDEN STATE
    h_t = o_t * torch.tanh(c_t)
```**

---

### **4. Example Usage**
Finally, let's instantiate our VanillaLSTM model and test it with some random inputs to verify that everything is working correctly.

```python
# Example usage:
input_size = 10
hidden_size = 20

model = VanillaLSTM(input_size, hidden_size)
x = torch.randn(1, input_size)
h_prev = torch.randn(1, hidden_size)
C_prev = torch.randn(1, hidden_size)

h_t, C_t = model(x, (h_prev, C_prev))
print("New Hidden State:", h_t)
print("New Cell State:", C_t)
```

Remember, this is a basic LSTM implementation to demonstrate the core concepts. PyTorch's in-built `nn.LSTM` module provides a more optimized and feature-rich implementation.

---
## Why have an output gate when we have a modified cell state?


1. **Decoupling Memory and Output**: LSTMs separate the concepts of memory (cell state) and the output (hidden state). While the cell state spans over many time steps and captures long-term dependencies, the hidden state is more about capturing short-term dependencies. An LSTM might want to remember information for a long time but only expose it when it's relevant. For example, in language modeling, an LSTM might remember a subject mentioned at the start of a sentence, but it might only want to use that information when a verb appears later in the sentence. Between those two points, the network might not want to reveal that it's "thinking" about the subject.

2. **Controlled Information Flow**: By using the output gate, LSTMs have an extra layer of control over the information flow. They might decide that some information, even though stored in the cell state, isn't immediately relevant for the current output.

3. **Flexibility**: In some situations, an LSTM might decide that even though it has updated its memory with new information, it doesn't want to reflect that in its outputs yet. It's an additional knob for the network to tune, giving it more flexibility.

4. **Architectural Symmetry**: From a design perspective, having an output gate provides a symmetry to the LSTM architecture. You have three gates each controlling input, memory, and output, giving the LSTM balanced control over all aspects of its operation.

To put it more intuitively, imagine your brain as an LSTM. The **cell state** is your long-term memory. The **input and forget gates** decide what new information to store and what old stuff to forget. Now, just because you learned something new or remembered an old fact doesn't mean you're always talking about it. The **output gate** is like your decision on what current thoughts or memories to share in a conversation. Even if you remember it, you might not mention it unless it's relevant.

--- 
## Difference Between Cell State and Hidden State 

The difference between the cell state and the hidden state in an LSTM is one of the foundational aspects that sets LSTMs apart from simpler recurrent neural networks (RNNs). Here's a breakdown of their differences:

1. **Functionality and Role**:
    - **Cell State $( C_t )$**: 
        - Acts as the "long-term memory" of the LSTM. It carries information across many time steps, capturing long-term dependencies.
        - Is modified by the gates (especially the forget and input gates) to add or remove information.
    - **Hidden State $(h_t )$**: 
        - Acts as the "short-term memory" or the "working memory" of the LSTM. 
        - It's the output of the LSTM cell for the current timestep, which will be used as one of the inputs for the next timestep and can also be fed to other layers in a neural network (e.g., another LSTM layer or a dense layer).
  
2. **How They're Updated**:
    - **Cell State**: 
        - The forget gate decides what information to throw away or keep.
        - The input gate decides which values in the state should be updated based on the new input.
    - **Hidden State**: 
        - Is derived from the cell state but is filtered through the output gate. This means the information in the cell state isn't directly exposed; it's regulated by the output gate.

3. **Visualization**: 
    - If you visualize an LSTM cell, the cell state is often represented as a horizontal line running through the top of the cell, emphasizing its role in carrying information across timesteps. The hidden state, on the other hand, emerges from the LSTM cell at each timestep and is passed to the next timestep.

4. **Intuition**: 
    - Think of the **cell state** as the long-term "knowledge" or "memory" of the network, which gets updated as new information comes in, with some old information possibly getting discarded.
    - The **hidden state**, meanwhile, is more of a transient state that represents the current or immediate state of the LSTM, conveying the most recent outputs and the immediate context.

In essence, the combination of cell state and hidden state allows LSTMs to maintain a more nuanced and controlled memory mechanism, capable of learning and remembering over long sequences and deciding which information to propagate forward. This capability helps mitigate the vanishing gradient problem seen in traditional RNNs and allows LSTMs to model long-term dependencies in data more effectively.