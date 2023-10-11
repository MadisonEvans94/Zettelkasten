#seed 
upstream:

---

**video links**: 

---

# Brain Dump: 


--- 

## Step-by-Step Pseudocode for Grad-CAM

![[Pasted image 20231010092709.png]]

### Step 1: Initial Setup
- Choose the convolutional layer in your CNN that you'll use for Grad-CAM. Typically, this is one of the last convolutional layers in the network.
- Select the input image you want to interpret.
- Determine the class label you're interested in. This could be the class the model predicts, or any other class you're curious about.

### Step 2: Forward Pass
- Perform a forward pass of the input image through the CNN.
- At the chosen convolutional layer, capture and store the intermediate feature maps. These feature maps will be multi-dimensional arrays containing spatial information of different features learned by the network.

### Step 3: Compute Class Score
- Complete the forward pass all the way to the output layer to get the class scores (logits).
- Identify the score corresponding to the class of interest. This score will serve as the basis for computing the loss that we'll backpropagate.

### Step 4: Backward Pass
- Initialize gradients to zero throughout the network.
- Perform a backward pass to compute the gradients of the selected class score with respect to the stored feature maps. These gradients will capture how much each spatial location in the feature maps impacts the class score.

### Step 5: Global Average Pooling of Gradients
- For each feature map, average its gradients over all spatial locations. This results in a single scalar for each feature map, and these scalars form a weight vector.

### Step 6: Compute the Weighted Combination of Feature Maps
- Multiply each feature map from the stored activations by its corresponding weight from the weight vector.
- Sum these weighted feature maps along the feature dimension. The result is a 2D spatial map where each location represents the importance of that location in identifying the class of interest.

### Step 7: ReLU Activation
- Apply the ReLU function to this 2D spatial map to eliminate negative values. Negative values are not important for the class of interest.

### Step 8: Generate Heatmap
- Resize this 2D spatial map to the size of the input image. 
- Overlay it onto the original image to see which parts of the image are important for the selected class.

By following this step-by-step procedure, you'll generate a heatmap that emphasizes the regions in the input image that are most important for the selected class, according to your CNN model.

