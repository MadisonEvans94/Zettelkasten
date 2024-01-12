#evergreen1 
###### upstream: [[LSTM]]

### Definitions:


1.  **Forget Gate (Deciding what to forget)**: The LSTM cell determines what proportion of the existing long-term memory (cell state) to retain. This is decided based on the current input and the previous short-term memory (hidden state).
    
2.  **Input Gate (Deciding what to update)**: The LSTM cell first decides on a candidate memory based on the current input and the previous short-term memory. Then, it uses the input gate to decide what proportion of this candidate memory should be used to update the long-term memory.
    
3.  **Output Gate (Deciding what to output)**: The LSTM cell calculates the new short-term memory (also the output of the cell) based on the updated long-term memory and the current input. The proportion of the long-term memory to be used for this calculation is decided by the output gate.

**Main Takeaway**: For LSTM cells, everytime there is a sigmoid function, that is where the *gate* is 

### Diagram:![[LSTM gates analogous diagram.jpeg]]

- **tanh** is the function that calculates the actual memory value while the **sigmoid** function determines what percentage to attenuate 
- **sigmoid** function can thought of as *notches* as depicted in diagram above. These notches determine how much contribution it's input has 
	- for example, an input like the character `Q` would probably have an input gate value close to 1 because the next letter is almost certainly going to be `u`