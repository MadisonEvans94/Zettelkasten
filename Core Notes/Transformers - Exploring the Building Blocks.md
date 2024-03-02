#seed 

upstream: [[Transformers]], [[Deep Learning]]

---
brain dump: 

Here's how you would calculate the dimensions of the weight matrices:

- `d_k` (dimension of keys) = `E / nhead` = `4 / 2` = 2
- `d_v` (dimension of values) = `E / nhead` = `4 / 2` = 2

So, for each head, you would have:

- Weight Matrix for Q (W^Q) of each head: `(E, d_k)` = `(4, 2)`
- Weight Matrix for K (W^K) of each head: `(E, d_k)` = `(4, 2)`
- Weight Matrix for V (W^V) of each head: `(E, d_v)` = `(4, 2)`

---


**links**: 

---

- [x] Positional Encoding 
- [x] Multi-Head Attention
- [ ] Add & Norm 
- [ ] Skip Layers 
- [ ] Feed Forward
- [ ] Cross Attention
- [ ] Outputs Shifted Right 
- [ ] Masked Multi-Head Attention

---

## Boilerplate Reference 

```python
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, 
                 d_model: int = 512, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1, 
                 activation: str = "relu", 
                 custom_encoder: nn.Module = None, 
                 custom_decoder: nn.Module = None):
        super(Transformer, self).__init__()
        
        # Initialize model parameters (like dimensions, heads, layers, dropout, activation)
        self.d_model = d_model
        self.nhead = nhead
        # ... (other attributes)
        # If custom encoder or decoder are provided, use them; otherwise create new ones
        self.encoder = custom_encoder if custom_encoder is not None else nn.Module() # Replace with actual encoder
        self.decoder = custom_decoder if custom_decoder is not None else nn.Module() # Replace with actual decoder

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Define the forward pass for the transformer
        # - Apply masks and padding
        # - Pass through the encoder
        # - Pass through the decoder
        # - Return the output
        pass

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        # Define the encoding part of the transformer
        # - Apply source mask and padding mask
        # - Pass through each encoding layer
        # - Return the memory (output of the encoder)
        pass

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Define the decoding part of the transformer
        # - Apply target mask, memory mask, and padding masks
        # - Pass through each decoding layer
        # - Return the output
        pass

    # Implement methods for generating masks and padding
    def generate_square_subsequent_mask(self, sz):
        # Create and return a mask for the sequences
        pass
    
    def generate_source_key_padding_mask(self, sz):
        # Create and return a padding mask for the source sequences
        pass
    
    # Add any additional helper functions or methods required for the transformer
    # - Functions for initializing weights
    # - Any custom layers or components
    # - Helper methods for loading/saving model
    # ... (other helper functions)

# This skeleton outlines the class structure without implementing the actual functionality,
# providing placeholders where the actual logic needs to be written.

```

---

## Architecture Diagram 

![[Transformers Hand Drawing.pdf]]

---

## Tensor Shapes 

Visualizing the flow of data through a Transformer architecture can be complex due to the many operations and transformations applied to the tensors. Here's a breakdown of the key tensors and their shapes within a standard Transformer model:

1. **Input Tokens Tensor**:
- Shape: `(N, M)`
- Description: A batch of sequences with `N` representing the batch size and `M` the sequence length. Each element is an integer representing a token.
![[Screen Shot 2023-11-11 at 3.31.37 PM.png]]


2. **Token Embedding Tensor**:
- Shape: `(N, M, E)`
- Description: The embedded representation of the input tokens where each token is mapped to an `E` dimensional embedding.

![[Screen Shot 2023-11-11 at 3.31.56 PM.png]]

3. **Positional Encoding Tensor**:
- Shape: `(M, E)`
- Description: Positional encodings added to the embedded tokens, with the same dimension as the token embeddings. This is broadcasted to the entire batch.

![[Screen Shot 2023-11-11 at 3.32.17 PM.png]]
4. **Summed Embedding Tensor**:
- Shape: `(N, M, E)`
- Description: The sum of the token embeddings and the positional encodings, providing position-aware embeddings for the input sequences.

![[Screen Shot 2023-11-11 at 3.32.35 PM.png]]
5. **Query (Q), Key (K), and Value (V) Tensors** for each head in Multi-Head Attention:
- Shape: `(N, M, E/nhead)` each
- Description: The embedded tokens are linearly projected to create `Q`, `K`, and `V` for each attention head (`nhead` is the number of heads).

![[Screen Shot 2023-11-11 at 3.32.55 PM.png]]

![[Screen Shot 2023-11-11 at 3.33.06 PM.png]]
6. **Scaled Dot-Product Attention Scores Tensor**:
- Shape: `(N, nhead, M, M)`
- Description: The attention scores computed for each head, where the self-attention mechanism computes a score for each key-query pair.

7. **Attention Output Tensor** for each head:
- Shape: `(N, M, E/nhead)` each
- Description: The result of the attention score applied to the `V` tensor for each head.

![[Screen Shot 2023-11-11 at 3.33.54 PM.png]]
8. **Concatenated Attention Output Tensor**:
- Shape: `(N, M, E)`
- Description: All attention head outputs are concatenated back to form a single output tensor per input sequence.

9. **Feed-Forward Network Input Tensor**:
- Shape: `(N, M, E)`
- Description: The input for the position-wise feed-forward network, usually the same as the concatenated attention output tensor.

10. **Feed-Forward Network Output Tensor**:
- Shape: `(N, M, E)`
- Description: The output of the feed-forward network, which is then passed through a nonlinear activation function.

11. **Encoder Output Tensor**:
- Shape: `(N, M, E)`
- Description: The final output of the encoder stack, which serves as the `K` and `V` in the decoder's multi-head attention mechanisms.

12. **Target Sequence Embedding Tensor** (in the decoder):
- Shape: `(N, M_target, E)`
- Description: The embedded representation of the target sequences, where `M_target` is the target sequence length.

13. **Decoder Output Tensor**:
- Shape: `(N, M_target, E)`
- Description: The final output of the decoder stack, which is then passed to a final linear layer and softmax to predict the output tokens.

14. **Final Output Probabilities Tensor**:
- Shape: `(N, M_target, vocab_size)`
- Description: The output probabilities for each token in the target sequence, where `vocab_size` is the size of the vocabulary.

These are the primary tensors and their shapes that you would encounter in a Transformer model. Note that for simplicity, some details like layer normalization, dropout, and residual connections are not explicitly represented here, but they are present in the architecture, applied to appropriate tensors as per the model's design.

---

## Transformer Components Breakdown

### Positional Encoding

In natural language processing, the order of tokens in a sequence is crucial for understanding the content and context. Transformers use two primary methods to incorporate sequence order information: **positional encoding** and **positional embedding**.

#### Positional Encoding

Positional encoding is a deterministic algorithm typically implemented using **sine** and **cosine** functions of different frequencies. The functions are designed so that each position in the sequence generates a unique encoding, which is then added to the token embeddings. This process infuses order information without the need for additional training parameters. The advantage of this method is that it can handle sequences of any length and the model can infer relationships based on the mathematical properties of the encodings. However, since it's not learned, it may not capture more complex positional relationships as effectively as a learned system might.

#### Positional Embedding

Positional embedding, on the other hand, treats positions as tokens and learns an embedding for each position in a sequence during the training process. This is done through layers like `nn.Embedding` in PyTorch, which map each position to a high-dimensional space. The learned embeddings are then added to the token embeddings. Since this method learns the positional relationships from the data, it can potentially capture more complex patterns specific to the task at hand. The tradeoff is that the model becomes limited by the maximum sequence length for which it was trained, and adding more positions requires additional training.

#### High-Level Purpose

Both methods aim to provide the model with awareness of the order of tokens, which is not inherently present in the token embeddings alone. The high-level purpose is to ensure that the model can take into account the sequence of words or tokens, which is essential for understanding the meaning in tasks such as language translation, text summarization, or question-answering.

#### Tradeoffs

Positional encoding is more flexible and generalizable for different sequence lengths, but it may not capture positional relationships as effectively as positional embedding. Positional embedding, while potentially more powerful, requires more parameters and is fixed to the sequence lengths seen during training. Choosing between the two methods involves considering the nature of the task, the variability of sequence lengths in the data, and the computational resources available for training.

### Input Embedding

Before delving into the intricacies of the Transformer's attention mechanisms, it's crucial to understand the role of input embedding, which serves as the entry point for data into the model.

#### Structure and Function

Input embedding is the process of converting tokens, which can be words or subword units, into vectors of a specified dimension (`d_model`). This vector representation is necessary because models do not understand text or discrete tokens; they operate on numerical data. The embedding layer thus acts as a lookup table, where each unique token is assigned a high-dimensional vector. The vectors are learned during the training process, allowing the model to capture semantic and syntactic information about each token.

#### Type of Inputs

The inputs to the embedding layer are typically integer IDs that correspond to tokens. These integers are neither one-hot encoded vectors, which would be highly inefficient in terms of space, nor are they simple dictionary keys. Instead, these integers serve as indices that the model uses to retrieve the corresponding vectors from the embedding matrix. This matrix has a size of `vocab_size x d_model`, where `vocab_size` is the number of unique tokens in the vocabulary.

#### Why Input Embedding is Needed

Input embedding is essential for a few reasons:

1. **Dimensionality Reduction**: It provides a dense representation of tokens, as opposed to sparse one-hot vectors, which would be computationally prohibitive for a large vocabulary.
   
2. **Semantic Information**: Embeddings capture the meaning of tokens in a way that raw indices cannot. Tokens with similar meanings are often closer in the embedding space, enabling the model to generalize well over unseen data.

3. **Training Efficiency**: Learning embeddings as part of the model allows for simultaneous optimization of token representations along with the model's parameters, leading to more efficient training.

The resulting embedded vectors are then combined with the positional encodings or embeddings to retain the sequence order before being passed through the subsequent layers of the Transformer model. This combination ensures that the model not only understands what the tokens are but also their context within the sequence, which is pivotal for tasks involving sequence modeling.

### Multi-Head Attention

The Multi-Head Attention mechanism in Transformers allows the model to jointly attend to information from different representation subspaces at different positions. It does this by creating multiple sets of attention weights (heads), each with its own set of linear transformations.

#### Matrices in Attention

The core idea of attention involves three matrices: queries (Q), keys (K), and values (V). These are derived from the input embeddings, typically through linear transformations. For an input sentence like "I like apples", each word would be embedded into a vector and then transformed into Q, K, and V matrices:

- **Q (Queries):** Reflects what to look for in other parts of the input sequence.
- **K (Keys):** Represents the parts of the sequence to be attended to.
- **V (Values):** Contains the actual content from the input sequence.

Each word in the sequence is passed through these matrices to calculate the attention weights. The dimensions of these matrices depend on the number of attention heads and the size of the attention keys, queries, and values (`d_k`, `d_q`, `d_v`).

#### Example with "I like apples"

Let's assume each word in "I like apples" is embedded into a vector of size `d_model`. These vectors are then linearly transformed into Q, K, and V vectors of size `d_k`, `d_q`, and `d_v` respectively for each head.

- For simplicity, let's assume `d_model = 6` and we have 2 heads, so `d_k = d_q = d_v = 3`.
- Each word now results in two sets of Q, K, and V vectors, one for each head.

Here's a simplified illustration:

```
Input Embeddings: [I_emb, like_emb, apples_emb]
Transformations: I_emb * W^Q -> I_q1, I_emb * W^K -> I_k1, I_emb * W^V -> I_v1 (for head 1)
                 I_emb * W^Q -> I_q2, I_emb * W^K -> I_k2, I_emb * W^V -> I_v2 (for head 2)
                 ... (similar transformations for 'like' and 'apples')
```

`W^Q`, `W^K`, and `W^V` are the weight matrices that project the embeddings into Q, K, and V spaces for each head.

#### Conceptual Representation of Weight Matrices

- **Weight Matrices (`W^Q`, `W^K`, `W^V`):** These matrices can be thought of as lenses focusing on different aspects of the embedding information. For example, one might focus on the syntactic role, while another might focus on semantic content.

#### Role of Multiple Heads

Having multiple heads allows the model to capture different types of dependencies in the data:

- One head might attend to the syntactic structure, understanding the grammatical relationships between words.
- Another head might capture semantic relationships, focusing on the meaning conveyed by the sequence.

#### Induced Roles

After training, these heads often specialize to perform different tasks. While it's not explicitly enforced during training, empirical observations show that different heads indeed tend to learn to attend to different kinds of patterns in the data. This specialization allows the model to be more expressive and to capture a richer understanding of the data.

In essence, Multi-Head Attention allows the Transformer to process the input sequence in a parallel and richly interconnected manner, leading to a more nuanced understanding of the input data.


---



### Positional Encoding

In natural language processing, the order of tokens in a sequence is crucial for understanding the content and context. Transformers use two primary methods to incorporate sequence order information: **positional encoding** and **positional embedding**.

#### Positional Encoding

Positional encoding is a deterministic algorithm typically implemented using **sine** and **cosine** functions of different frequencies. The functions are designed so that each position in the sequence generates a unique encoding, which is then added to the token embeddings. This process infuses order information without the need for additional training parameters. The advantage of this method is that it can handle sequences of any length and the model can infer relationships based on the mathematical properties of the encodings. However, since it's not learned, it may not capture more complex positional relationships as effectively as a learned system might.

#### Positional Embedding

Positional embedding, on the other hand, treats positions as tokens and learns an embedding for each position in a sequence during the training process. This is done through layers like `nn.Embedding` in PyTorch, which map each position to a high-dimensional space. The learned embeddings are then added to the token embeddings. Since this method learns the positional relationships from the data, it can potentially capture more complex patterns specific to the task at hand. The tradeoff is that the model becomes limited by the maximum sequence length for which it was trained, and adding more positions requires additional training.

#### High-Level Purpose

Both methods aim to provide the model with awareness of the order of tokens, which is not inherently present in the token embeddings alone. The high-level purpose is to ensure that the model can take into account the sequence of words or tokens, which is essential for understanding the meaning in tasks such as language translation, text summarization, or question-answering.

#### Tradeoffs

Positional encoding is more flexible and generalizable for different sequence lengths, but it may not capture positional relationships as effectively as positional embedding. Positional embedding, while potentially more powerful, requires more parameters and is fixed to the sequence lengths seen during training. Choosing between the two methods involves considering the nature of the task, the variability of sequence lengths in the data, and the computational resources available for training.", give me a subsection that shows a pytorch example of setting a 