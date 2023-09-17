#evergreen1 
###### upstream: [[Deep Learning]]

[But What is A Convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA&t=1103s&ab_channel=3Blue1Brown)
[CNN Visualized](https://www.youtube.com/watch?v=f0t-OCG79-U&t=2s&ab_channel=IsraelVicars)

### Brain Dump 

- `(k1xk2+1)` parameters
- full vs valid convolution
- cross correlation and it's relationship to convolution
- feature map/activation map
- edge detection kernel operation example with coordinate system 
- Y=I-k+1
- weight sharing 

---

### Underlying Question: 
- What is a convolutional neural network, how does it work, and what is it best at solving

### Details: 

Convolutional Neural Networks (CNNs) are a class of deep learning models that have revolutionized computer vision tasks, such as image classification, object detection, and many more. The main idea behind CNNs is that they can automatically and adaptively learn spatial hierarchies of features from the data.

### Structure of a Convolutional Neural Network:

A typical CNN has three types of layers: convolutional layers, pooling layers, and fully connected layers.

1.  **Convolutional layers:** These are the main building blocks of CNNs. In this layer, a set of learnable filters (or [[Kernel]]s) slide over the input image (or feature map from the previous layer) and perform convolution operations to produce new feature maps. These feature maps highlight the regions in the image where features learned by the filters have been detected.

2.  **Pooling layers (downsampling):** These layers are used to reduce the spatial dimensions (width and height) of the input volume. This helps to reduce computational cost, memory usage, and also to prevent overfitting. The most common type of pooling is max pooling, which extracts the maximum value from the region it is applied to. To understand this better, check out [[Understanding Pooling Layers]]

3.  **Fully connected layers:** These layers are typically at the end of the network. They take the high-level features learned by the convolutional layers and the pooled feature maps and use them to classify the image into a label (for example, identifying whether an image contains a cat, a dog, or another object).


### Key Concepts in Convolutional Neural Networks:

1.  **Local receptive fields:** Unlike traditional fully connected layers in neural networks, where each neuron is connected to all neurons in the previous layer, in CNNs, each neuron is only connected to a small region of the input volume. The spatial extent of this connectivity is a hyper-parameter called [[the receptive field of the neuron]].

2.  **Shared weights and biases:** In CNNs, each filter is defined by a set of weights and a bias. The key feature of CNNs is that these weights and bias are shared across the whole input. This means that all the neurons in a feature map share the same parameters.

3.  **Translation invariance:** Since the same filters (with the same weights and biases) are used across the whole image, CNNs have the property of [[Translation Invariance]], meaning that they can recognize patterns no matter where they occur in the image.


### Use Cases of Convolutional Neural Networks:

Convolutional Neural Networks have been immensely successful in many computer vision tasks such as:

-   Image and video classification
-   Object detection in images and videos
-   Image synthesis and enhancement
-   Facial recognition
-   Medical imaging analysis

CNNs, due to their capability to automatically and adaptively learn spatial hierarchies of features, have been a game-changer in the field of computer vision and continue to be at the core of most state-of-the-art computer vision systems.


### Examples (if any): 

