#incubator 
###### upstream: [[NLP]]

### Origin of Thought:
- saw them listed on the same slide of lecture so wanted to know what the concepts are and if they have any relationship to each other 

### Underlying Question: 
- first off, what is negative sampling? 
- what is softmax? 
- are these two related? If so, how? 

### Solution/Reasoning: 

Let's start of by explaining each of these by using the [[Icecrem Line Analogy]]

**Negative Sampling**: 

Remember how in the skip-gram model, each word (or person in our line) tries to pass a message to their nearby friends (context words)? Well, that's what we call "positive sampling" — these are the real or positive examples of who usually stands next to each other in line.

But to learn well, the model also needs to understand who _doesn't_ usually stand together. It's like understanding who's not a usual friend of our person in the line. This is where "negative sampling" comes in.

**Negative Sampling** is when we randomly pick a few people from the line who aren't standing close to our target person and pretend that they were. So, if we're still focusing on "cat", we might select "mat" as a negative sample if our window size is small enough that "mat" is outside it.

We then tell the model, "Hey, these folks don't usually stand next to our target person." This helps the model to understand both who _does_ and who _doesn't_ tend to stand next to each other in different lines (sentences).

To put it simply, "negative sampling" is about learning from the opposite — who doesn't belong or fit together. This strategy helps make the model more accurate in its predictions by strengthening its understanding of the relationships between words.

Just like how knowing both who your friends are and who they aren't can give a good picture of your social circle, positive samples (nearby words) and negative samples (randomly chosen distant words) together give the skip-gram model a clearer picture of the linguistic context of words. It's like knowing your allies annnnd knowing your opps 


Now... 

To understand **Naive Soft-max**, consider the following: 

Remember the game of "telephone" in the ice-cream line? Well, in that game, every person (word) tries to guess who they are going to pass the message to (the context words). They could pass the message to the person standing right next to them, two people away, three people away, and so on, depending on the size of the window.

In a perfect world, our person in line should have a clear idea about who usually stands near them in line (sentences), based on their past experience standing in different lines (training data). But, of course, they don't know this from the beginning. They need to learn this through playing the telephone game multiple times (training the model).

Now, let's imagine that our person is "cat", and they are trying to guess who they should pass the message to. Maybe they think it's most likely that they will pass it to "sat", somewhat likely to "the", less likely to "on", and so on. This is what the model is trying to predict: the probabilities of each word being a context word for "cat".

**Naive Softmax**

Naive Softmax is a method to compute these probabilities. The word "softmax" comes from a mathematical function that takes in a bunch of numbers and squishes them so that they all fall between 0 and 1 and their total sums up to 1, making them proper probabilities.

For each word in the vocabulary, the naive softmax computes a score (which can be any real number) based on the current understanding of the model, and then it uses the softmax function to convert these scores into probabilities.

So, in our ice-cream line analogy, the naive softmax is like a voting system where every person in line (all the words in the vocabulary) gets to cast a vote about how likely they are to receive the message from "cat". The votes are then tallied and converted into probabilities.

The problem with naive softmax, however, is that it's computationally expensive when the vocabulary size (the line for ice-cream) is very large, because it involves computing and normalizing scores for every single word in the vocabulary, even though only a few of them are actual context words. This is why techniques like negative sampling are used to make this process more manageable.

I hope this makes things clearer! Let me know if you have any other questions.


### How are These Concepts Related: 

Naive Softmax and Negative Sampling are two different techniques used to handle the output layer of the Skip-gram model in Word2Vec.

**Naive Softmax** is a straightforward approach where, given an input word, the model computes the probabilities for every word in the vocabulary being a context word. It does so by giving each word a score and then applying the softmax function to convert these scores into probabilities. The goal during training is to adjust the model's parameters to maximize the probability of the actual context words.

However, the naive softmax has a significant computational cost, especially when the vocabulary is large. For each training example, we need to compute and normalize scores for every single word in the vocabulary, which is computationally intensive.

To tackle this issue, a technique called **Negative Sampling** is used. Instead of trying to predict all words in the vocabulary, Negative Sampling simplifies the problem by taking a sample of words.

Negative Sampling changes the task to a binary classification problem for each context word. It takes one positive example (a true context word) and a few negative examples (words randomly sampled from the vocabulary, which are assumed to not be context words). The goal then is to adjust the model's parameters to maximize the model's ability to distinguish the positive examples from the negatives.

So, Naive Softmax and Negative Sampling are both strategies to handle the output of the Skip-gram model, but they approach the problem in different ways. While Naive Softmax considers all words in the vocabulary for each prediction, Negative Sampling simplifies the task by only considering a sample of negative examples along with the positive ones.

In practice, Negative Sampling tends to be used more commonly due to its efficiency, especially when dealing with large vocabularies.

