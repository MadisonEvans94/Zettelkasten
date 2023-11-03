#incubator 
###### upstream: [[Deep Learning]], [[LSTM]]

## Definition: 

**Seq2Seq**, is a family of machine learning approaches used for natural language processing. Applications include language translation, image captioning, conversational models, and text summarization.

---
## Architecture Overview 

![[seq2seq diagram.png]]

Here's a breakdown of the components and their functions:

### Encoder

   - The encoder processes the input sequence (in this case, "Are you free tomorrow?").
   - Each word (or *token*) of the input sequence is passed through a series of recurrent units, which could be traditional [[RNN (Recurrent Neural Network)]], [[LSTM]], or gated recurrent units (GRUs). In this diagram, they are LSTM cells with the "tanh" activation function.
   - As each word is processed, the encoder updates its internal state. By the end of the sequence, the encoder produces a "thought vector," which is a dense representation of the entire input sequence. This thought vector aims to capture the meaning or context of the input sequence.
  
### Decoder

   - The decoder uses the thought vector from the encoder to generate the output sequence. Here, the output sequence is "Yes, what's up?"
   - The process starts with a special `<START>` token.
   - Similar to the encoder, the decoder has its own series of recurrent units. For each step, the decoder produces an output word based on its current internal state and the thought vector.
   - The output word becomes the input to the next step in the decoder.
   - The process continues until a special `<END>` token is produced, signaling the end of the output sequence.

### Key Points:
- The encoder processes the input sequence and compresses its information into a thought vector.
- The decoder takes this thought vector and produces an output sequence.
- The entire process allows the model to transform one sequence (like a sentence in one language) into another sequence (like a translated sentence in another language).

---
## A Walkthrough of the Forward Pass

Let's walk through the forward pass of a fully trained sequence-to-sequence (seq2seq) model step by step. I'll use the example in the provided diagram to make it easier to follow.

**Encoder**: 
- The input sequence is "*Are you free tomorrow?*".
- The word "*Are*" is fed into the first cell of the encoder. This cell updates its internal state based on this input.
![[repeating_rnn_modules.png]]
- Next, the word "*you*" is input to the second cell. This cell considers both its input ("you") and the previous cell's state to update its own state.
- This process continues for the words "*free*" and "*tomorrow?*"
- By the end of the sequence, the encoder has accumulated information from the entire input sentence and produces a thought vector. This vector is a dense representation of the input sequence and is supposed to capture its essence or meaning.

**Decoder**
- The decoder starts with the `<START>` token. Along with the thought vector, this token is fed into the first cell of the decoder.
- Based on the thought vector and the `<START>` token, the first cell produces an output word. In this case, the word is "*Yes*,". This word is also treated as the input for the next step.
- The word "*Yes*," and the thought vector are now input to the second cell of the decoder. It processes these inputs and produces the next word, "what's".
- Similarly, "*what's*" and the thought vector are fed into the third cell, resulting in the word "*up?*".
- The process would continue until the decoder produces an `<END>` token, signaling the end of the output sequence. In this example, it seems the process stops after "up?", but in other situations, it might continue until that end token is produced.

- **Important Note:** In each step of the decoder, the produced word is based on:
  - The previous word (or the `<START>` token for the first step).
  - The thought vector from the encoder.
  - The internal state of the current cell, which accumulates information from all previous steps.

### Key Takeaway:
The encoder processes the input sequence and captures its information into a thought vector. The decoder then uses this thought vector to generate an output sequence, word by word, while considering the previously generated words. The thought vector acts as a bridge between the encoder and decoder, ensuring the decoder has knowledge of the entire input sequence as it generates the output.

---

## A Walkthrough of the Forward Pass During Training

In typical seq2seq tasks, the encoder and decoder are trained jointly from scratch. This is because the encoder's purpose is to produce a "thought vector" or context that is meaningful for the decoder to produce the correct output sequence. Training them together ensures that the encoder produces representations that the decoder can best use, and vice versa.

Let's walk through the training process of a seq2seq model using the example sentences ("Are you free tomorrow?" → "Yes, what's up?"). 

In the initial stages of training, the outputs of the model will likely be gibberish due to random weight initialization
### Forward Pass:

1. **Encoder**:
   - The sentence "*Are you free tomorrow?*" is tokenized into words: `["Are", "you", "free", "tomorrow?"]`.
   - Each word is then converted into a vector, typically using embeddings.
   - These word vectors are fed into the encoder one-by-one.
   - The final state of the encoder after processing the entire input sequence is the "thought vector".

2. **Decoder** (with Teacher Forcing):
   - The `<START>` token is input to the decoder along with the thought vector.
   - It attempts to generate the response, but if we're in the early iterations of training, it may end up with gibberish such as "*Tomato dogs another France*" because of its untrained state

3. **Output & Loss Calculation**:
- The output sequence from the decoder, "*Tomato dogs another France*", is compared to the target sequence, "*Yes, what's up?*".
- For each time step (word position) in the decoder's output:
    - The model predicts a probability distribution over the entire vocabulary. In the early stages, this distribution is likely almost random.
    - Cross-entropy loss is calculated for that time step, comparing the predicted distribution to a one-hot encoded vector of the actual word in the target sequence.
- The total loss is the sum (or average) of the losses at each time step.

### Backward Pass (Backpropagation):

1. **Compute Gradients**:
   - Starting from the loss, calculate the gradient of the error with respect to each weight in the model. This tells us the direction and magnitude of change required for each weight to reduce the error.
   - For this, the chain rule of calculus is applied iteratively from the loss backward through all operations in the network.

2. **Update the Weights**:
   - Using an optimizer (like SGD, Adam, etc.), adjust each weight based on its gradient. The learning rate defines the size of the step taken in the direction of the gradient.
   - For example, if a particular weight in the model was responsible for a large portion of the error, it would be adjusted more significantly than a weight that contributed very little.

3. **Reset States**:
   - At the end of backpropagation, the internal states of the RNN cells (in both the encoder and decoder) are reset. This is because, during training, we want the model to treat each sentence pair independently. 

### Iterative Process:

- This forward and backward pass is done for each sentence pair in the dataset.
- Multiple passes (epochs) are made over the entire dataset to refine the model's weights and reduce the overall error.
- Over time, as the model sees more examples and adjusts its weights, it becomes better at producing accurate translations or responses.

>The beauty of seq2seq models lies in their ability to handle sequences of varying lengths and learn intricate patterns in sequential data. As the model is trained on more data and for more epochs, its predictions should align more closely with the expected outputs.

## Incorporating Attention 

In the standard seq2seq model, the encoder compresses the entire input sentence into a fixed-size "thought vector", which the decoder then uses to generate the output. This can be limiting, especially for long sentences.

Attention allows the decoder to "focus" on different parts of the input sentence at each step of its output generation, dynamically giving more weight to the relevant parts of the input.

### How Attention Works with Our Example:

**1. Input Processing**:

- Like before, we start by converting the sentence "*Are you free tomorrow?*" into embeddings.
- These embeddings are then passed through the encoder, producing a sequence of encoder hidden states. Each state can be thought of as a representation of the input up to that point.

**2. Initialization**:

- The decoder is initialized with a start token and the last hidden state of the encoder, as usual.

**3. Attention Calculation**: For each word the decoder produces:

- **Score Calculation**: The current decoder hidden state is compared to all encoder hidden states to produce a set of scores. The exact manner of comparison varies. A common method is the dot product between the decoder hidden state and each of the encoder hidden states.

- **Softmax**: These scores are then passed through a softmax function to produce the attention weights. These weights sum to one and indicate the model's focus on the input sequence. For instance, when generating the word "Yes" in response, the attention weights might be higher for the word "free" and "tomorrow" in the input sequence, implying a relationship between being "free" and affirming with "Yes".

- **Context Vector**: This weighted sum of encoder hidden states, given by the attention weights, results in what's called a context vector. The context vector can be thought of as an aggregated representation of the input sequence, weighted by importance (or attention) concerning the current decoding time step.

**4. Decoder's Step with Attention**:

- The decoder's RNN takes in the context vector and its previous hidden state to produce the current hidden state.

- The context vector is usually concatenated with the decoder's input embedding for that time step. This combined vector is then used to generate the output for the current time step.

- For instance, when generating the response "Yes, what's up?", the context vector for producing "Yes" might be heavily influenced by the words "free" and "tomorrow" from the input sentence, allowing the decoder to produce a more contextually relevant word.

**5. Continue Decoding**:

- This process repeats for each word in the output sequence until an `<END>` token is generated or some max length is reached.

### Why is Attention Useful?

1. **Handling Long Sequences**: Attention helps mitigate the information bottleneck in the encoder's fixed-size context vector, especially for long input sequences.

2. **Interpretable Results**: The attention weights can be visualized, offering insights into which parts of the input the model deems important when producing a particular word in the output.

3. **Improved Performance**: On many tasks, especially translation, attention mechanisms have significantly improved the performance of seq2seq models.


### Wrapping Up:

With attention, the model doesn't solely rely on the encoder's final state to gather information about the entire input sentence. Instead, it can "refer back" to the entire input sequence, making it more flexible and powerful in capturing the nuances of languages and producing more accurate and contextually relevant outputs.