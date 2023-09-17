#evergreen1  
###### upstream: 

[Multi Layer RNN Video](https://www.youtube.com/watch?v=NvBAJVppaEM&ab_channel=NeilRhodes)

### Understanding the Structure:

In a sense, a multi-layer **RNN** can be viewed as passing the data through the same RNN cell multiple times. However, there are some fundamental differences between simply running the output through the same RNN cell multiple times and stacking multiple layers in an RNN.

**Reasoning for Layering**: 

When you're increasing the number of layers, each layer will have its own weights and hidden state. This allows each layer to potentially learn to represent different features or abstractions from the data. The lower layers often learn to represent more low-level, direct features of the data, while the higher layers learn to represent more abstract, high-level features. This is because each layer gets its input from the layer below it, so it has a chance to build upon the representations learned by the lower layers.

On the other hand, if you were to simply cycle the output through the same RNN cell multiple times, you're not really increasing the complexity or depth of the network. The cell has the same weights each time, so it's likely to just end up representing the same features each time.

To draw a parallel with non-recurrent (feedforward) neural networks, you could think of a single layer RNN with a loop as a fully connected layer with a self-loop, while a multi-layer RNN would be like a deep fully connected network. Both structures allow the network to process each input multiple times, but the multi-layer network allows for more complex representations and can potentially learn to model more complex functions.

It's also worth noting that in practice, training deep multi-layer RNNs can be quite challenging due to issues like vanishing and exploding gradients, which can make it difficult for the network to learn long-range dependencies. Techniques like LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit) cells, as well as methods like gradient clipping, are often used to help mitigate these issues.


### Examples (if any): 

ok so let's say we're processing the phrase "*I like food*" in a multi layered RNN.

This code snippet assumes that you want to create a 2-layer LSTM-based RNN for text classification:

```python 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Assuming that your vocabulary size is 10000, embedding dimension is 100, and maximum length of a sentence is 50
vocab_size = 10000
embed_dim = 100
max_length = 50

model = Sequential()

# Embedding layer
model.add(Embedding(vocab_size, embed_dim, input_length=max_length))

# First LSTM layer
model.add(LSTM(64, return_sequences=True))

# Second LSTM layer
model.add(LSTM(64))

# Dense layer for classification. 
# This assumes a binary classification problem. For multi-class problems, you may need to use a 'softmax' activation function and adjust the number of units accordingly.
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

```

This is a very simple model, but it should illustrate the basic concept of a multi-layer RNN in Keras. Please note that in practice, you would likely need to adjust the hyperparameters (like the number of layers, the number of units in each layer, the type of RNN cell, etc.) based on the specifics of your problem.

Also note that the LSTM layers are configured with the `return_sequences` argument set to `True`, which means they will return their full sequence of outputs (a 3D tensor) rather than just their output at the last timestep (a 2D tensor). This is necessary for stacking LSTM layers. The last LSTM layer or before the Dense layer `return_sequences` is set to `False` or it's left to its default value which is `False`.

The Dense layer at the end would then make the final prediction (for example, a binary prediction if you're doing sentiment analysis) based on the output from the second LSTM layer.

### Diagraming: 

let's represent a 2-layer RNN using simple ASCII art. We'll assume a sequence length of 3 for the input, i.e., processing the words "I", "like", and "food" from your example.

First, let's represent the overall flow:

```zsh
INPUT SEQUENCE --> EMBEDDING --> LAYER 1 (RNN) --> LAYER 2 (RNN) --> OUTPUT
```

Now, let's show this in more detail:

```zsh
INPUT SEQUENCE (I, like, food)
   |
   V
EMBEDDING LAYER
   |
   V
LAYER 1 (RNN) --> Hidden state 1
   |
   V
LAYER 2 (RNN) --> Hidden state 2
   |
   V
OUTPUT (classification, sequence, etc.)

```

Each step (word in the sequence) passes through each layer:

```zsh
  Word 1    Word 2   Word 3
     |        |       |
     V        V       V
  Embedding Embedding Embedding
     |        |       |
     V        V       V
  LSTM 1 --> LSTM 1 --> LSTM 1   -> Hidden State 1
     |        |       |
     V        V       V
  LSTM 2 --> LSTM 2 --> LSTM 2   -> Hidden State 2
     |        |       |
     V        V       V
Output   Output   Output

```

The "->" arrows represent the hidden state being passed to the next time step within the same layer.

Remember, in a RNN, data is processed sequentially, so these steps are done one after another (not all at once). The output at each step depends on the current input and the hidden state from the previous step, which provides the 'memory' of what has been processed so far. Each layer has its own hidden state, which is passed on to the next step within that layer. The output at each step (except the last one) is passed up to the next layer immediately.

As for what the layers represent, here's a general idea:

-   **Embedding layer**: This layer transforms each word from a one-hot encoded vector of size `vocab_size` into a dense vector of size `embed_dim`. It essentially represents each word as a point in a `embed_dim`-dimensional space, such that words with similar meanings are closer together in this space.
    
-   **LSTM 1 (first layer of LSTM)**: The first LSTM layer typically learns to understand lower-level structure in the data. In the case of text, it might learn things like syntax, common word groupings, and other simple structures.
    
-   **LSTM 2 (second layer of LSTM)**: The second LSTM layer typically learns to understand higher-level structures, building on what LSTM 1 has detected. It might learn to detect more complex patterns that span multiple words or even entire sentences.