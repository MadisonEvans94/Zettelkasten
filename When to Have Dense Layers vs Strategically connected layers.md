#incubator 
###### upstream: [[Deep Learning]]
https://www.youtube.com/watch?v=70A3uYfM1qA&ab_channel=IntuitiveML

### Definitions:

1.  **Densely Connected Networks (Fully Connected Layers)**: Each neuron in one layer is connected to every neuron in the next layer.
    
2.  **Strategically Connected Layers (Sparse Networks)**: Connections between layers are not fully formed and there is a deliberate architecture behind which neurons connect to which.


### When to use each: 

The type of connectivity you use heavily depends on the kind of problem you're trying to solve.

**Densely connected layers** are great for problems where global patterns in data are useful. They work well in tasks where the relation between all features needs to be considered. For example, in image classification tasks, fully connected layers are often used at the end of Convolutional Neural Networks (CNNs) after feature extraction to classify the image based on the features.

However, densely connected layers have their drawbacks. They can increase the computational cost significantly, especially for large input data, due to the high number of parameters. They also have a higher risk of *overfitting*, which can be mitigated with regularization methods and dropout.

**Strategically connected layers** or sparse networks are often seen in CNNs where local spatial hierarchies are more important. For example, in image processing tasks, CNNs make use of local spatial coherence in images, assuming that pixels close to each other are more related than those far apart. Hence, each neuron in a layer doesn't need to be connected to all neurons in the previous layer. Instead, it only needs to be connected to a local region in the input volume (its [[receptive field]]), leading to fewer parameters and less computational cost.

Specific use cases where sparse networks are beneficial include image and video processing tasks, natural language processing (for example, in transformers), and time series analysis.


### Solution/Reasoning: 


### Examples (if any): 

