This is the first major topic to study that's directly related to BERT and GPT. RNNs are a type of artificial neural network designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or the spoken word.

Topics to Study:
#incubator 

###### upstream: [[Deep Learning]]

### Diagram:

![[IMG_0744.jpg]]

- In the above diagram, each time step has an input `x` and that input is multiplied by a weight matrix before being added to the weighted output of the previous step.  
- Notice how a RNN has 3 weight matricies 
	- W1 is the weight matrix that the input gets passed through 
	- W2 is the weight matrix that the recurrent output is passed through prior to getting added to the current input 
	- W3 is the weight matrix that the recurrent output is passed through in order to provide the `y` prediction for that time step 
- Backpropagation for RNN is referred to as [[Back Propogation Through Time]]

### Thought Bucket: 

- RNN's suffer from the **vanishing gradient problem**, which basically makes them explode or converge to zero change 
- Understanding RNN's are essential for understanding [[LSTM]], which is necessary for understanding [[Transformers]]
- Understanding [[Multilayer RNNs]]

### Solution/Reasoning: 



### Examples (if any): 

