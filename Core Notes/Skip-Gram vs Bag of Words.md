#incubator 
###### upstream: [[NLP]]

### Origin of Thought:
- understanding the two approaches and how they differ 

### Underlying Question: 
- how can we define each? 
- how are they different 

### Solution/Reasoning: 

- Both **skip-gram** and **bag of words** are used to learn word embeddings, which are a form of vector representations for words 

- **Skip Gram**: 
	- The model was popularized by the [[Word2Vec]] algorithm developed by Tomas Mikolov at Google in 2013 
	- The problem skip-gram solves is representing words in a way that captures their meanings, semantic relationships, and the contexts in which they appear. This is a significant problem in NLP because computers are inherently bad at understanding language; they treat words as discrete symbols that are devoid of meaning.
	- The skip-gram model does this by training a [[neural network]] to predict the context words given a target word. For instance, in the sentence "The cat sat on the mat", if "sat" is our target word, the context words could be "The", "cat", "on", "the", "mat". 
	- By learning to predict the context, the model learns representations for the words that reflect their meanings. Words with similar meanings will have similar vector representations, allowing mathematical operations in the form of [[linear combinations]] that reflect semantic relationships (like 'king' - 'man' + 'woman' = 'queen').

- **Bag of Words**: 
	- Bag of Words is a simpler, more classic technique in NLP. It is used for document classification, sentiment analysis, spam filtering, and other tasks *where the order of words isn't very important*.
	- The problem BoW solves is converting text data, which is unstructured, into a structured form that can be understood by machine learning algorithms.
	- BoW represents a document as a "bag" (or multiset) of its words, disregarding grammar and word order but keeping track of frequency. That is, it converts text into a matrix where each unique word is a column in the matrix, each document is a row, and each cell represents the frequency of a word in a document.
	- So, for example, the sentence "The cat sat on the mat" and "The mat sat on the cat" would have the same representation under a BoW model, despite the different meanings of the two sentences.

- **How they differ**: 
	1. **Context Sensitivity**: Skip-gram is sensitive to the context in which a word appears, while BoW completely disregards the order of words.
	2. **Semantics**: Skip-gram learns vector representations that capture the semantic meanings of words, whereas BoW simply treats words as discrete symbols.
	3. **Complexity**: Skip-gram is generally more computationally complex due to the training of the neural network, while BoW is simpler but can result in high-dimensional data when the vocabulary size is large.
	4. **Use Cases**: Skip-gram is often used in tasks that require understanding the meanings of words (e.g., named entity recognition, part-of-speech tagging, etc.), while BoW is often used in document classification tasks where the occurrence of certain words is more important than their order.
	5. **Dimensionality**: BoW representations are usually high-dimensional with the size of the vocabulary, and they are sparse (most entries are zero). Skip-gram, on the other hand, creates dense vectors of a predefined size (e.g., 50, 100, or 300 dimensions), which can lead to better performance in many tasks.


### Examples (if any): 

**Bag of Words example with Python:**

Let's start with a simple implementation of the Bag of Words model. In this example, I'll use the `CountVectorizer` from `sklearn.feature_extraction.text`.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Assume we have the following documents
documents = [
    'The cat sat on the mat',
    'The dog sat on the log',
    'Cats and dogs are great'
]

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer to the documents and transform the documents into vectors
X = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Print the feature names
print("Feature names: ", feature_names)

# Print the BoW representation for each document
print("BoW representation for each document: ")
print(X.toarray())

```

The output will be a matrix where each row represents a document and each column represents a unique word in the corpus. The value at each cell will be the frequency of the word in the corresponding document. In this example, the feature names are the unique words found in the documents. The array following the feature names is the Bag of Words representation of the documents. Each row of the array corresponds to a document, and each column corresponds to a word. The numbers represent the frequency of the word in the document.

```python
Feature names:  ['and', 'are', 'cat', 'cats', 'dog', 'dogs', 'great', 'log', 'mat', 'on', 'sat', 'the']
BoW representation for each document: 
[[0 0 1 0 0 0 0 0 1 1 1 2]
 [0 0 0 0 1 0 0 1 0 1 1 2]
 [1 1 0 1 0 1 1 0 0 0 0 0]]

```

**Skip-gram example with Python:**

A full implementation of the Skip-gram model is a bit more complex and beyond the scope of a short example as it involves implementing and training a neural network. However, I'll show you how to generate skip-grams from a text using `keras.preprocessing.sequence`:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

# Assume we have the following text
text = "The cat sat on the mat"

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}
sequences = tokenizer.texts_to_sequences([text])[0]

# Generate skip-grams
pairs, labels = skipgrams(sequences, len(word2id))

# Print the skip-grams
for i in range(len(pairs)):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
        id2word[pairs[i][0]], pairs[i][0], 
        id2word[pairs[i][1]], pairs[i][1], 
        labels[i]
    ))

```

In this script, `skipgrams` generates pairs of words where a pair consists of a target word and a context word. A label of `1` indicates that a context word is within the window size of the target word, while a label of `0` indicates a negative sample.

Remember, in practice, you would train a neural network to learn to predict the context word from the target word, or vice versa, which is the main idea of Word2Vec's Skip-gram model.

In this example, the pairs of words are generated with their respective labels. The pairs of words with a label of `1` are true skip-grams (i.e., context words that are within the window size of the target word), while the pairs with a label of `0` are negative samples (i.e., words that are not context words).

Please note that the exact output may vary due to the randomness involved in generating negative samples.

```python
(mat (1), the (2)) -> 1
(on (4), cat (5)) -> 1
(the (2), sat (3)) -> 1
(cat (5), mat (1)) -> 0
(mat (1), sat (3)) -> 1
(the (2), mat (1)) -> 1
(sat (3), on (4)) -> 1
(the (2), on (4)) -> 1
(cat (5), the (2)) -> 1
(mat (1), on (4)) -> 1
(sat (3), the (2)) -> 1
(sat (3), mat (1)) -> 0

```


Here, the output `(mat (1), the (2)) -> 1` means that 'the' is a context word for 'mat' within the given window size, and it's a positive sample. Meanwhile, `(cat (5), mat (1)) -> 0` means that 'mat' is not a context word for 'cat' within the window size, and it's a negative sample.

### Explain as if I'm 12: 

let's break down **skip-gram** a bit more using [[Icecrem Line Analogy]]

First, let's consider the sentence "The cat sat on the mat." Imagine each of these words is a person in a line waiting for an ice-cream truck. They're all standing in a particular order. "The" is first in line, then "cat", then "sat", and so on.

**What is Skip-gram?**

Skip-gram is like a game of "telephone". The person in the middle (the target word) wants to pass a message to a few people next to them in line (the context words). This game helps everyone (the words) understand who is close to whom in line (the sentence).

For example, if "cat" (our target word) is passing the message, the people standing close to "cat" could be "The", "sat", "on", and "the". These are the context words.

The goal of the skip-gram model in Word2Vec is to learn to predict these context words given a target word. It's like teaching our "cat" person to recognize who usually stands near them in the ice-cream line.

**What is a Window?**

The "window" is simply how many people on either side of our target person ("cat" in this case) that we want to consider as potential message receivers.

If the window size is 1, "cat" would only pass the message to "The" and "sat" because they're the closest. If the window size is 2, "cat" could pass the message to "The", "sat", "on", and "the" because they're within 2 spots in the line.

**Skip-gram in Action**

Now, let's put it all together. When we train the skip-gram model, we're teaching it to recognize who usually stands next to who in various lines for ice cream (sentences in our text). Over time, the model gets pretty good at understanding the relationships between words (people in line). It learns things like "cat" and "mat" often stand close together, so they probably have something in common.

And this is how Word2Vec's Skip-gram model works. It learns to understand the relationships between words based on their co-occurrences in the text, and it uses this understanding to create a numerical representation (word embedding) that reflects these relationships.