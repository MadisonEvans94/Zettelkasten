#incubator 
###### upstream: [[Word2Vec]]

### Origin of Thought:
To create the training data necessary for Word2Vec models from thousands of documents of raw text, you need to follow a few steps. These steps involve pre-processing the text data and organizing it into a suitable format.

### Underlying Question: 
- What steps should I take for pre-processing data? 

### Solution/Reasoning: 
Here's a general overview of the steps you might take:

1.  **Text Preprocessing:** Start with cleaning and normalizing the text. This often includes:
    -   Lowercasing: This ensures that the same word in different cases is not treated as different words.
    -   Removing punctuation: Punctuation can interfere with recognizing the same word in different contexts.
    -   Removing stop words: These are common words like 'and', 'the', 'is', etc. that do not provide much semantic meaning.
    -   [[Tokenization]]: This is the process of splitting text into individual words (tokens).
2.  **Create Word Pairs:** Once your text is cleaned and tokenized, the next step is to generate your training data. For each word in each document, take the surrounding words as its context. The number of words you take is determined by the window size. For example, if you choose a window size of 2, then for each word, you will take two words before it and two words after it as context. Each pair of target word and context word is a training example.
    
3.  **Convert Words to Indices:** In order to use words as input to a model, we often convert them into numerical form. One common way to do this is to create a vocabulary of all unique words in your data and assign each word a unique index. You can then represent each word by its index in the vocabulary.
    

At this point, you have your training data: for each target word (represented by its index), you have a set of context words (also represented by their indices).

When training the model, you'd use techniques like negative sampling or softmax regression to update the word vectors. The aim is to make the vectors of words that appear in similar contexts closer to each other. This way, after training, the model will output a unique vector for each word that captures its semantic meaning based on its context in the training data.

Keep in mind that these are high-level steps and the exact implementation might differ slightly based on the specific requirements of your task, the nature of your text data, and the specifics of the model or library you're using.

### Examples (if any): 

