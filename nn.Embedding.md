#seed 
upstream:

---

**video links**: 

---

### Introduction

`nn.Embedding` is a layer provided by PyTorch in its neural network module. It's primarily used for converting sparse categorical data (like word indices) into dense vector representations (embeddings).

### Usage

```python
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
```

- **`num_embeddings`**: Total number of unique embeddings in input (e.g., vocabulary size in NLP tasks).
- **`embedding_dim`**: Dimensionality of the embedding vector (e.g., 100, 300 for word embeddings).

### Key Points

1. **Initialization**: By default, the embedding weights are initialized randomly but can be fine-tuned during training.
2. **Pre-trained Embeddings**: You can also load pre-trained embeddings (like [[GloVe]], [[FastText]]) and optionally fine-tune them.
3. **Padding**: `nn.Embedding` has a `padding_idx` argument that can be used to specify an index which won't be updated during training, commonly used for [[NLP padding]].
  
    ```python
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    ```

4. **Sparse Updates**: PyTorch provides an option to perform sparse updates to the embedding matrix, which can be memory efficient for certain applications. This is enabled by setting `sparse=True`.
  
### Example

Suppose we have a vocabulary of size 5 and want embedding vectors of size 3:

```python
vocab_size = 5
embedding_size = 3
embedding_layer = nn.Embedding(vocab_size, embedding_size)

# Input a tensor of size 2 containing vocabulary indices
input_data = torch.tensor([1, 4])
embedded_data = embedding_layer(input_data)
print(embedded_data) # Tensor of size [2, 3]
```

### Additional Notes

- Embedding layers can be seen as a simple look-up table. Given an index, it returns the embedding associated with that index.
- They are crucial for tasks like NLP where inputs can be large vocabularies, and dense representations (embeddings) capture semantic meanings.


