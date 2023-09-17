#incubator 
###### upstream: [[Linear Algebra]]

### Definition:
a **tensor** is a multi-dimensional array of numbers (which can be seen as a generalization of scalars, vectors, and matrices) that can be used in various mathematical operations, including transformations that project inputs onto different vector spaces.

For example, in the context of neural networks and [[Deep Learning]], the weights of a fully-connected (dense) layer can be represented as a 2D tensor, or a matrix. When an input vector (a 1D tensor) is multiplied by this weight matrix, it's essentially being projected into a new vector space. This projection allows the network to learn complex patterns in the input data.

The concept of tensors extends to higher dimensions as well. For instance, the weights of a convolutional layer in a Convolutional Neural Network (CNN) can be represented as a 4D tensor, with dimensions corresponding to the height and width of the filters, the number of input channels, and the number of output channels. When this 4D tensor is convolved with an input tensor (such as a 3D tensor representing an image), it's projecting the input into a different, possibly higher-dimensional space, allowing the network to detect spatial hierarchies in the data.

In this way, tensors provide a flexible and efficient way to represent high-dimensional data and transformations, which is essential for many machine learning tasks.

