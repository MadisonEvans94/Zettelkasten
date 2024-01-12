#seed 
upstream:


---

**links**: 

[Foundations of NLP Explained Visually: Beam Search, How It Works](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24)

---

Brain Dump: 

--- 




# Beam Search

Beam Search makes two improvements over Greedy Search.

- With Greedy Search, we took just the single best word at each position. In contrast, Beam Search expands this and takes the best ’N’ words.
- With Greedy Search, we considered each position in isolation. Once we had identified the best word for that position, we did not examine what came before it (ie. in the previous position), or after it. In contrast, Beam Search picks the ’N’ best _sequences_ so far and considers the probabilities of the combination of all of the preceding words along with the word in the current position.

In other words, it is casting the “light beam of its search” a little more broadly than Greedy Search, and this is what gives it its name. The hyperparameter ’N’ is known as the Beam width.

Intuitively it makes sense that this gives us better results over Greedy Search. Because, what we are really interested in is the best complete sentence, and we might miss that if we picked only the best individual word in each position.

# Beam Search — What it does

Let’s take a simple example with a Beam width of 2, and using characters to keep it simple.

![[Beam Search Diagram.png]]

Certainly, here is a section explaining how beam search scores sequences:

---

## Scoring Sequences in Beam Search

Beam search selects the top 'N' sequences by evaluating their scores at each step in the sequence generation process. The score of a sequence is determined by the *cumulative probability of the sequence so far*. This scoring mechanism is crucial as it allows the algorithm to balance between longer sequences with lower probability words and shorter sequences with higher probability words. Here's how scoring typically works:

### Probability of a Sequence
The probability of a sequence in a language model is the product of the probabilities of each word given the previous words. However, because multiplying many probabilities (which are less than 1) can lead to underflow, beam search operates in the log space. The log probabilities of the individual words are summed to get the score of the entire sequence.

### Length Normalization
Simply multiplying probabilities favors shorter sequences over longer ones because additional multiplication by a fraction (the probability of the next word) decreases the overall score. To address this, length normalization can be applied, which divides the cumulative log probability by a function of the sequence length, often just the length itself or some power of the length.

### Scoring Function
The scoring function for a sequence \( S \) of length \( L \) with words \( w_1, w_2, ..., w_L \) typically looks like this:

\[ \text{Score}(S) = \frac{1}{L^\alpha} \sum_{i=1}^{L} \log P(w_i | w_1, w_2, ..., w_{i-1}) \]

where \( P(w_i | w_1, w_2, ..., w_{i-1}) \) is the conditional probability of word \( w_i \) given the preceding words in the sequence, and \( \alpha \) is a hyperparameter that controls the length normalization.

### Beam Search Steps
At each step of the beam search:
1. The algorithm extends the top 'N' sequences from the previous step by one additional word, creating a set of candidate sequences.
2. It calculates the score for each candidate sequence using the scoring function.
3. It selects the 'N' sequences with the highest scores to keep for the next step.

### End-of-Sequence Token
Once the end-of-sequence token is generated for a sequence, it is removed from the pool of actively extended sequences but is kept in the list of completed sequences if its score is among the top 'N'.

### Final Selection
After processing up to a maximum sequence length or when a stopping criterion is met (like no change in top sequences), beam search selects the sequence with the highest overall score from the set of completed sequences.

In summary, the scoring of sequences "so far" in beam search is a crucial component that guides the algorithm to select the most promising sequences. It takes into account not only the raw probabilities of words but also adjusts for sequence length, preventing a bias toward shorter sequences. The length normalization factor and the careful consideration of cumulative probabilities are what enable beam search to outperform greedy methods in generating higher-quality sequences.