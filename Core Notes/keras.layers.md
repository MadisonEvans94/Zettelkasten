#seed 
###### upstream: [[Software Development]]

### Details: 

The `layers` module of Keras is a central part of the library. It provides the fundamental building blocks for designing and implementing deep learning models. Each layer in Keras encapsulates a state (the layer's "weights") and a transformation from inputs to outputs.

The core data structure of Keras is a **model**, which is a way to organize layers. The simplest type of model is the `Sequential` model, a linear stack of layers. For more complex architectures, you should use the Keras functional API, which allows [[building arbitrary graphs of layers]].

The layers have several common methods and properties:

1.  **Common Methods:**
    
    -   `layer.weights`: Returns the weights of the layer as a list of TensorFlow tensors.
    -   `layer.get_weights()`: Returns the weights of the layer as a list of NumPy arrays.
    -   `layer.set_weights(weights)`: Sets the weights of the layer from a list of [[NumPy]] arrays.
    -   `layer.get_config()`: Returns a dictionary containing the configuration of the layer. The layer can be reinstantiated from its config via `layer = Layer.from_config(config)` or `layer = Layer(**config)`.
    
1.  **Common Properties:**
    
    -   `layer.input`: The input tensor(s) of a layer.
    -   `layer.output`: The output tensor(s) of a layer.
    -   `layer.input_shape`: The shape of the input received by the layer during the last call to `call()`, including the batch size.
    -   `layer.output_shape`: The shape of the output computed by the layer during the last call to `call()`, including the batch size.

The `layers` module in Keras contains numerous classes, each providing a different type of layer functionality. Some of the more commonly used classes include:

-   **Dense**: This is a fully connected layer where each input node is connected to each output node.
-   **Activation**: Applies an activation function to the output.
-   **Dropout**: Randomly sets a fraction of the input units to 0, which helps prevent overfitting.
-   **Flatten**: Flattens the input into a one-dimensional array.
-   **Conv2D**: Applies a 2D convolution over an input signal composed of several input planes.
-   **MaxPooling2D**: Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window for each channel of the input.
-   **BatchNormalization**: Applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

Each of these classes can take various arguments to modify the behavior of the layer. When you construct a model in Keras, you are essentially connecting these layers to one another, specifying how the data should flow through the network and what transformations should be applied. After defining your model architecture, Keras handles much of the lower-level computation for you, allowing you to focus on the high-level structure of the model.

*does `Dense`, `Conv2D`, and `MaxPooling` have any methods unique to them that aren't directly inherited from `Layer`?*

The `Dense`, `Conv2D`, and `MaxPooling2D` classes in Keras primarily use the methods defined in the `Layer` base class. They don't define new public methods that would be used by a developer. However, they do override some of the base class methods to provide the functionalities needed for their specific types of layers.

For example, the `Dense` class overrides the `call()` method from the `Layer` base class to provide the functionality of a dense (fully connected) layer. Similarly, the `Conv2D` and `MaxPooling2D` classes override the `call()` method to provide the functionality of a 2D convolutional layer and a 2D max pooling layer, respectively.

These classes also override the `build()` method from the `Layer` base class. This method is called the first time the layer is used, and it's where the layer's weights are initialized. The `Dense`, `Conv2D`, and `MaxPooling2D` classes each have their own `build()` method to initialize their weights in the way that's appropriate for their type of layer.

It's important to note that while these methods are technically part of the classes' APIs, they are not typically called directly by developers. Instead, they are called internally by Keras when the model is trained. As a developer, you generally don't need to worry about these methods - you just need to create the layers and add them to your model, and Keras takes care of the rest.