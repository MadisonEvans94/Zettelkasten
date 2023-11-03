#incubator 
upstream: [[Deep Learning]]

---

**links**: [word2vec tutorial](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

---
brain dump 

- [ ] what is $J= max(0, 1-s + s_c)$ ? 



---


## High Level Insight 

The idea is to train a simple dense layer neural network on a "fake" task, and the weights we discover will be the **word embeddings** 

Given a word in the middle of a sentence (input) we will pick a close neighboring word at random, and the network will tell us the probability for every word in our vocabulary of being the “nearby word” that we chose.

When I say "nearby", there is actually a "window size" parameter to the algorithm. A typical window size might be 5, meaning 5 words behind and 5 words ahead (10 in total).

![[Skip Gram Example.png]]

The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.

## Algorithm Details 

### One Hot Encoded Inputs 

To implement, we will represent are vocabulary in the form of one-hot encoded vectors. For a vocabulary of 10,000 words, each word will be a one-hot encoded vector with a length of 10,000. 

![[Word2Vec NN Architecture.png]]

When _training_ this network on word pairs, the input is a one-hot vector representing the input word and the training output _is also a one-hot vector_ representing the output word. But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution (i.e., a bunch of floating point values, _not_ a one-hot vector).

### The Hidden Layer

For our example, we’re going to say that we’re learning word vectors with 300 features. So the hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron).

![[Word2Vec Architecture Continued.png]]

If you look at the _rows_ of this weight matrix, these are actually what will be our word vectors! In other words, if we want to find the embedding of the first word, it's just the first row of our weight matrix after training. So the end goal of all of this is really just to learn this hidden layer weight matrix – the output layer we’ll just toss when we’re done!

If you multiply a 1 x 10,000 one-hot vector by a 10,000 x 300 matrix, it will effectively just _select_ the matrix row corresponding to the “1”. This means that the hidden layer of this model is really just operating as a lookup table. The output of the hidden layer is just the “word vector” for the input word.

### The Output Layer 

The output layer is a [[softmax]] regression classifier. Each output neuron (one per word in our vocabulary) will produce an output between 0 and 1, and the sum of all these output values will add up to 1.

![[Word2Vec Softmax Diagram.png]]

>Note that neural network does not know anything about the offset of the output word relative to the input word. It _does not_ learn a different set of probabilities for the word before the input versus the word after. To understand the implication, let's say that in our training corpus, _every single occurrence_ of the word 'York' is preceded by the word 'New'. That is, at least according to the training data, there is a 100% probability that 'New' will be in the vicinity of 'York'. However, if we take the 10 words in the vicinity of 'York' and randomly pick one of them, the probability of it being 'New' _is not_ 100%; you may have picked one of the other words in the vicinity.

## Intuition 

If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. And one way for the network to output similar context predictions for these two words is if _the word vectors are similar_. So, if two words have similar contexts, then our network is motivated to learn similar word vectors for these two words! Ta da!

And what does it mean for two words to have similar contexts? I think you could expect that synonyms like “intelligent” and “smart” would have very similar contexts. Or that words that are related, like “engine” and “transmission”, would probably have similar contexts as well.

## Modifications

One of the fallbacks of the word2vec algorithm is that it is a huge model... And so training can be difficult and costly. 

In the previous example, we had word vectors with `300` components, and a vocabulary of `10,000` words. Recall that the neural network had two weight matrices–a hidden layer and output layer. Both of these layers would have a weight matrix with `300` x `10,000` = `3 million` weights each. Millions of weights times billions of training samples means that training this model is going to be a beast.

The authors of Word2Vec addressed these issues in their second [paper](http://arxiv.org/pdf/1310.4546.pdf) with the following two innovations:

1. Subsampling frequent words to decrease the number of training examples.
2. Modifying the optimization objective with a technique they called “**Negative Sampling**”, which causes each training sample to update only a small percentage of the model’s weights.

Subsampling frequent words and applying Negative Sampling not only reduced the compute burden of the training process, but also improved the quality of their resulting word vectors as well.

### Subsampling Frequent Words

To explain this strategy, let's look at the word *the*. There are two “problems” with common words like “the”:

1. When looking at word pairs, (“fox”, “the”) doesn’t tell us much about the meaning of “fox”. “the” appears in the context of pretty much every word.
2. We will have many more samples of (“the”, …) than we need to learn a good vector for “the”.

Word2Vec implements a “**subsampling**” scheme to address this. For each word we encounter in our training text, there is a chance that we will effectively delete it from the text. The probability that we cut the word is related to the word’s frequency.

If we have a window size of `10`, and we remove a specific instance of “the” from our text:

1. As we train on the remaining words, “the” will not appear in any of their context windows.
2. We’ll have 10 fewer training samples where “the” is the input word.

Note how these two effects help address the two problems stated above.

#### Sampling Rate 

$w_i$ is the word, $z(w_i)$ is the fraction of the total words in the corpus that are that word. For example, if the word "peanut" occurs `1000` times in a `1 billion` word corpus, then z('peanut') = `1E-6`. There is also a parameter in the code named ‘sample’ which controls how much subsampling occurs, and the default value is `0.001`. Smaller values of ‘sample’ mean words are less likely to be kept.

$P(w_i)$ is the probability of keeping the word: 
$$ P(w_i) = (\sqrt{\frac{z(w_i)}{.001}}+1) \cdot\frac{0.001}{z(w_i)}$$![[Sub Sampling Plot for Word2Vec.png]]

#### Stoner Analogy 

>Alright, so like, imagine you're chilling out, right? And you've got this massive bowl of your favorite snacks. Now, you're not gonna eat every single one, 'cause that's just too much, and some of them are like, the not-so-awesome pieces you get tired of. You want to keep it interesting and get the best taste experience. So you pick out a mix, but mainly the really good stuff.

>Now, let's bring that vibe to word2vec. Word2vec is this super cool tool that turns words into these magical little points in space. It's like giving every word its own chill spot in a huge galaxy of meaning. Words that are like each other, maybe they're synonyms or used in the same kind of sentences, end up hanging out close together. It's all about context and the company they keep, you know?

>But here's the thing, in language, like in your snack bowl, there are some words that show up all the time. They're like the plain chips or something. Words like "the," "is," and "and." If you focus too much on them, they're going to overwhelm the subtle flavors. You won't get the rich taste of the rare, interesting words that add spice to language.

>So, what word2vec does is this thing called "subsampling." It's like it takes a look at the bowl and goes, "Man, we've got way too many plain chips in here. Let's give the rare snacks more space to shine." It starts to skip over some of the plain chips, not all, but just enough so that you get a balanced taste in your language snack mix.

>By doing this, word2vec makes sure that the rare words get a fair chance to show off their unique flavors and how they relate to other words. It's about finding the perfect balance so that the model can get a good feel for the full spectrum of vibes that words can have.

Picture this: You've got a giant bag of mixed snacks - some are your basic pretzels (common words), and some are like exotic, flavored ones (rare words).

So $w_i$ is a specific type of snack. And $z(w_i)$ is how often you're pulling that snack out of the bag. If pretzels are 50% of your mix, their $z(w_i)$ would be `0.5`, because half the time, you're grabbing a pretzel.

Now, back to that trippy formula:

$$( P(w_i) = (\sqrt{\frac{z(w_i)}{0.001}} + 1) \cdot \frac{0.001}{z(w_i)} )$$

Imagine we're dealing with those super common pretzels, and we want to keep our snack experience exciting. So we use this formula to decide if we're gonna eat this particular pretzel we just picked or if we're gonna put it back and try for something else.

1. $( \sqrt{\frac{z(w_i)}{0.001}} )$ - This is like saying, *"How much more often am I picking pretzels over something super rare?"* If pretzels are super common, this square root part gets big.

2. Adding 1 to that square root - It's like saying, "Even though pretzels are common, I don't want to totally ignore them. I'll give them at least a little chance."

3. Then, by multiplying it with $( \frac{0.001}{z(w_i)} )$ - You're adjusting the whole thing. Since pretzels are so common (let's say $z(w_i)$ is high), this part makes the final number smaller, which means you're less likely to eat that pretzel. It's like telling yourself, "I've had so many pretzels already; let's see what else is in there."

After crunching all these numbers, $P(w_i)$ gives you a probability, a chance that you'll actually eat the pretzel you just picked. If it's a low chance, you might put it back and reach in for something different, something more thrilling for your taste buds.

So, with this formula, word2vec is sort of doing what you're doing with snacks but with words. It's deciding on the fly: *"Should I keep using this word to learn about language, or should I skip it this time and look for something less common?"* It helps the model not to get bored with "pretzels" and keeps an eye out for the "flavored chips" that make the snack mix (language) interesting.

### Negative Sampling 

**Negative sampling** is a technique used to efficiently train the skip-gram neural network of the word2vec model. Typically, adjusting the neural network to predict words would involve updating a vast number of weights for each training sample, which is computationally intensive given a large vocabulary. Negative sampling simplifies this by altering only a small subset of weights. For example, when learning the word pair ("fox", "quick"), instead of updating all output neurons, it selects a small number of 'negative' words (e.g., 5) and updates their weights to predict a 0, along with the 'positive' word ("quick") to predict a 1. This significantly reduces the number of weights updated (e.g., 1,800 out of 3 million in the output layer).

Negative words are chosen based on a unigram distribution, where words are raised to the 3/4 power, allowing for a balanced chance of selecting frequent and infrequent words. This approach, along with considering word pairs or 'phrases' as single entities (like "Boston Globe"), enhances the model's capability to capture nuanced meanings and results in a more sophisticated understanding of language context within a tractable computational framework.

#### Stoner Analogy 

Imagine you're trying to figure out what combinations of snacks go well together, like a flavor mastermind. You've got your favorites that you know pair up nicely, like cheese and crackers (let's call these your "positive samples"). But you're also kinda curious about what doesn't go well together, like gummy bears and mustard (these are your "negative samples").

Now, negative sampling in word2vec is like deliberately trying some weird snack combos to confirm they're not as good as your go-to pairs. Why? Because understanding what doesn't match helps you appreciate and recognize the good combos even more.

Let's break it down with the formula:

 $$P(w_i) = \frac{f(w_i)}{\sum_{j=0}^n(f(w_j))}$$

Imagine $f(w_i)$ is the amount of a specific snack you have, like pretzels. The bottom part, $\sum_{j=0}^n(f(w_j))$, is the total amount of all snacks combined.

1. $f(w_i)$ is how much of snack $i$ you've got.
2. The sum on the bottom $\sum$ is like adding up all the snacks in the bag.
3. $P(w_i)$ is the chance you'll pick out snack $i$ when you reach in without looking.

So, in the vast universe of snacking, $P(w_i)$ is like the likelihood of grabbing a particular snack randomly from the bag.

Here's where the negative sampling groove comes in: Instead of always trying every possible weird combo, which would be like eating every type of snack with every other type (and would take forever), you just try a few. You grab your cheese (your target word) and instead of trying it with every other snack (every other word in your vocab), you just try it with a small, random set of snacks (a small subset of all words) to see if it's a hit or a miss.

This way, you're teaching the word2vec model about what doesn't go together without having to taste-test every single bad combo. It’s efficient, like being smart with your snack choices. You still learn a lot about your snacks (words) and what makes a great party mix (word context) without overdoing it.

So, negative sampling helps the model learn from a practical number of bad matches (negative samples) so that it gets really good at predicting the awesome ones (positive samples). It's like fine-tuning your snack palate, but for words and their contexts.


## Intrinsic vs Extrinsic Evaluation

Intrinsic and extrinsic are terms used to describe different evaluation methods for word embeddings, which are vector representations of words.

### Intrinsic 

Intrinsic evaluation methods assess word embeddings based on how well they capture linguistic properties like semantic similarity, syntactic relationships, and analogy resolution. This evaluation is done using task-specific benchmarks that are designed to test particular aspects of word meaning or grammar.

A common intrinsic task is to measure how well the embeddings capture synonymy by checking if words that are close in the vector space are actually synonyms according to a thesaurus. Another task might be analogy solving, where the model is asked to complete analogies like *"king is to queen as man is to ---?"* using the vector arithmetic properties of the embeddings (i.e., by finding a word whose vector is closest to the result of the equation *"king - man + woman"*).

Intrinsic evaluations are usually quicker to perform and more focused than extrinsic ones. They are useful for getting a fast sense of the quality of the embeddings and for comparing different embedding models or algorithms. However, they might not always reflect how the embeddings will perform on real-world tasks.

### Extrinsic

Extrinsic evaluation methods measure the performance of word embeddings in downstream tasks – actual applications such as text classification, sentiment analysis, machine translation, or information retrieval. In this case, the word embeddings are a component of a larger system, and their quality is judged based on how much they improve the performance of that system.

For example, in sentiment analysis, word embeddings might be used to understand the sentiment expressed in sentences or documents. The embeddings are extrinsically evaluated based on how accurately the model can classify the sentiment when using these embeddings.

Extrinsic evaluations are usually more time-consuming and computationally expensive, as they require training and evaluating a full system rather than a simple task-specific benchmark. However, they provide a clear picture of how the embeddings will perform in practical applications.

### Summary 

- **Intrinsic Evaluations:** Quick, task-specific tests to probe the linguistic properties of embeddings.
- **Extrinsic Evaluations:** Assessment of embedding performance in real-world applications.

Choosing between intrinsic and extrinsic evaluation methods depends on what you want to know about the embeddings. If you are interested in their linguistic properties and how they capture meaning, go with intrinsic evaluation. If you want to know how they will help in an application, use extrinsic evaluation.