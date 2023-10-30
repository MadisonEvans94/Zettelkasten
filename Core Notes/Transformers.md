
#incubator 
upstream: [[Deep Learning]]

---

**links**: 

[Transformer: Concepts, Building Blocks, Attention, Sample Implementation in PyTorch](https://www.youtube.com/watch?v=6PmIoCnqcFU&t=1s&ab_channel=RowelAtienza)

---

## Feed Forward Walkthrough

Let's take a sentence such as the following: *"I like to eat apples"*

for the feed forward process, we will first tokenize this sentence into $N$ tokens

```python 
sentence = ["I", "like", "to", "eat", "apples"]
```

These tokens will each be passed to an embedding layer in order to get a new vector representation of fixed length $M$, creating an $N$ x $M$ matrix. 

Additionally, we have 3 weight matrices: $W_q$, $W_k$, and $W_v$

Each of these weight matrices will have a shape of $M$ x $M$

Multiplying the embedding matrix with each of the weight matrices gives 3 new tensors: Query, Key, and Value. Each of size $N$ x $M$

$$
Q = X \cdot W_Q 
$$
$$K = X \cdot W_K $$
$$V = X \cdot W_V$$

For self attention, we conduct the following arithmetic: 

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V 
$$

The portion "$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$" produces an output of size $N$ x $N$. This is a similarity matrix that looks like the following: 

$$
\begin{array}{c|ccccc}
\text{Tokens} & \text{"I"} & \text{"like"} & \text{"to"} & \text{"eat"} & \text{"apples"} \\
\hline
\text{"I"} & 1.0 & 0.2 & 0.1 & 0.1 & 0.05 \\
\text{"like"} & 0.2 & 1.0 & 0.4 & 0.3 & 0.1 \\
\text{"to"} & 0.1 & 0.4 & 1.0 & 0.5 & 0.2 \\
\text{"eat"} & 0.1 & 0.3 & 0.5 & 1.0 & 0.6 \\
\text{"apples"} & 0.05 & 0.1 & 0.2 & 0.6 & 1.0 \\
\end{array}
$$

> Note: In a real-world scenario, after computing this matrix, you'd scale it by dividing with $( \sqrt{d_k} )$ and then apply the soft-max function to get the attention weights. These weights would then be used to compute the weighted sum of the Value matrix to get the self-attention output.


When we multiply this output with $V$, we get a new output of shape $N$ x $M$. 

This output, known as the self-attention output, is then fed through a Multi-Layer Perceptron (MLP), also referred to as the feed-forward neural network within the Transformer block.




---
## Simple Diagram 

![[Screen Shot 2023-10-27 at 12.48.45 PM.png]]

---

## Contextual Influence in Self-Attention

The self-attention mechanism in transformers provides a powerful way to adjust word embeddings based on their surrounding context. One way to understand its impact is to observe the magnitude of change in a word's embedding from its initial value in the value matrix (input) to its position in the output matrix after attention.

### Understanding the Magnitude of Change

- **Minimal Change**:
  - If there's minimal change in a word's embedding, it suggests that the word's meaning remains largely consistent. 
  - It isn't heavily influenced by the surrounding words.
  - This is often the case for words with a clear, stable meaning in that particular sentence.

- **Significant Change**:
  - A pronounced shift in the embedding indicates that the word's meaning or relevance is being heavily adjusted based on the surrounding context.
  - This can happen with words that are ambiguous or whose interpretation is context-dependent.


