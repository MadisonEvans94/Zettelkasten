#seed 
upstream:

---

**video links**: 

---

# Brain Dump: 

https://www.doc.ic.ac.uk/~bkainz/teaching/DL/notes/equivariance.pdf

--- 

## Equivariance vs. Invariance in CNNs

### Definitions:

1. **Equivariance**: 
- When a transformation is applied to the input of a function, the output changes in a predictable manner corresponding to that transformation.
- In other words, if you shift or transform the input, the output will shift or transform in a similar way.

2. **Invariance**: 
- When a transformation is applied to the input of a function, the output remains unchanged or is resistant to that transformation.
- It means that no matter how you shift or transform the input, the output will stay the same.

### In the Context of CNNs:

1. **Equivariance**:
- CNNs are naturally equivariant to **translation**.
- If you move an object in an image, the feature maps in the initial layers will also move but the same features will still be detected.
- This property is due to the shared weights and the sliding window mechanism of convolutions.

2. **Invariance**:
- Invariance is typically desired in higher layers of the CNN, especially in tasks like classification.
- Pooling layers (e.g., max pooling) in CNNs introduce a form of translation invariance. Even if an object slightly shifts in the image, the pooled output remains similar.
- Invariance can also be achieved through data augmentation (e.g., rotating, scaling, and cropping images during training). This ensures the CNN learns features that are invariant to these transformations.

### Importance:

1. **Equivariance**:
- It ensures that the same features are detected irrespective of their position in the image.
- Maintaining spatial information in the initial layers is crucial for tasks like object detection or segmentation.

2. **Invariance**:
- Important for tasks where the exact position or orientation of a feature is not crucial, but its presence is (e.g., image classification).
- Ensures that the model generalizes well across various transformations of the input.

### Visualization:

Imagine an image of a cat sitting in the center. 

- If the cat moves to the right (shifted input), and we pass the image through the initial layers of a CNN, the detected features (like the cat's eyes or ears) will also shift to the right in the feature maps (shifted output) – this demonstrates **equivariance**.

- If we're trying to classify the image as containing a cat or not, despite where the cat is positioned (center, left, or right), our final prediction should ideally remain "cat" – this demonstrates **invariance**.
