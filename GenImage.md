
## Brain Dump 

- best generator: 
- best detector: 
- In Table 2, training on Stable Diffusion V1.4 achieves the best results
- For example, we respectively train eight ResNet-50 models using eight generators and average their evaluation results on Midjourney, yielding an accuracy of 59.0%. The evaluation is also performed on the other generators, such as Stable Diffusion V1.4, which leads to another fifty-six evaluation results. All the sixty-four testing results are then averaged to achieve 66.9%.
- For ImageNet 1000 class image classification, the backbone models tend to focus more on the classification of the content of the image.For GenImage, the focus is more on the pattern of discriminating between real and fake images.
- We train them from scratch for binary classification. In the cross-generator image classification task, the best result is achieved by Swin-T, while the other two methods are close in accuracy. Improving Transformer-based methods is a promising direction in our dataset.
- CNNSpot proposes that the recognition performance can be improved by augmenting the training data: (1) Images are blurred with σ ∼ Uniform[0,3] with 50% probability. (2) Images are JPEG-ed with 50% probability
- GAN often causes unique artifacts in the up-sampling component. Spec show that such artifacts are manifested as replications of spectra in the frequency domain. Thus, Spec takes spectrum instead of pixel as its input for classification.
- F3Net [26] contains two branches, namely Frequency-aware Image Decomposition (FAD) and Local Frequency Statistics (LFS). FAD studies manipulation patterns through frequency-aware image decomposition. LFS extracts local frequency statistics.

In this project, we aim to explore the performance of the Swin-T model on the GenImage dataset for detecting AI-generated images. Our motivation for focusing on the Swin-T model stems from its superior performance in the original research paper, where it outperformed other architectures like ResNet-50 and DeiT-S in terms of cross-generator accuracy. Given the increasing challenges in distinguishing AI-generated images from real ones, especially across different generative models, we believe that further optimizing Swin-T could lead to significant advancements in this field. As part of our strategy to enhance model performance, we're turning to frequency-domain data augmentation. The original paper showcased the benefits of data augmentation techniques like blurring and JPEG compression, which inspired us to explore the frequency domain. Transforming the images into the frequency domain allows us to manipulate spectral components, offering another avenue for model robustness and potentially improving the model's ability to generalize across different generators.

- 1. **Fourier Transform Image Preprocessing**:

   To apply Fourier Transform as an image preprocessing step, you can follow these general steps:

   - **Image Preparation**: Start with your input image dataset.
   - **Convert to Grayscale**: Convert the images to grayscale if they are not already in grayscale. Fourier Transform is typically applied to single-channel images.
   - **Apply Fourier Transform**: Use libraries like OpenCV or NumPy in Python to perform the 2D Fourier Transform on the grayscale images. You can use functions like `cv2.dft` in OpenCV to transform your images from the spatial domain to the frequency domain.
   - **Frequency Domain Manipulation**: In the frequency domain, you can manipulate the magnitude and phase components. This might involve filtering, scaling, or introducing noise to specific frequency components.
   - **Inverse Fourier Transform**: After making desired changes in the frequency domain, apply the Inverse Fourier Transform to convert the images back to the spatial domain.

   It's important to note that the exact details of the transformations you apply will depend on the specific goals of your data augmentation.

2. **Combining Transformed Image with Input Image**:

   To create an augmented dataset by combining the transformed image with the input image, you generally have two options:

   - **Element-Wise Combination**: You can combine the magnitude and phase components of the transformed image with those of the original image in the frequency domain. The magnitude and phase components are essentially the amplitude and phase information for each frequency. You can apply various strategies for combining these components, such as taking weighted averages.

   - **Combine in Spatial Domain**: You can convert the combined magnitude and phase components back to the spatial domain using the Inverse Fourier Transform. This will give you an augmented image in the spatial domain. You can then combine this spatial domain image with the original image using techniques like alpha blending, overlaying, or simply adding or subtracting pixel values.

   Whether you need to update your model's architecture depends on how you plan to use the augmented dataset during training. If your model can accept two images (original and augmented) as input and has the capability to process and learn from the combined data, you may not need significant architecture updates. However, if your model is designed for a single input image, you may need to adapt it to handle two-input scenarios.

   For combining the augmented image with the input image, it's essential to ensure that the resulting dataset is properly formatted and fed into your training pipeline. You may need to modify your data loading and preprocessing steps to handle the augmented data appropriately. Additionally, consider updating your loss functions or training strategies to account for the augmented data.

   The specific implementation details will depend on your deep learning framework and the architecture of your Swin-T model. It's essential to carefully plan and test the integration of augmented data into your training pipeline to ensure it has the desired impact on model performance.
Certainly! Frequency analysis for 2D images is a concept from signal processing and image analysis that helps us understand the frequency components present in an image. It's a fundamental technique for analyzing and processing images in the frequency domain, which is particularly useful for tasks like data augmentation and feature extraction. Here's a breakdown of the key concepts:

1. **Frequency Components in Images**:
   - **High-Frequency Components**: These correspond to rapid changes in intensity and are associated with fine details, edges, and textures in the image. High-frequency components represent the local variations.
   - **Low-Frequency Components**: These correspond to slow changes in intensity and represent the overall trends or structure in the image. Low-frequency components capture global variations.

2. **Fourier Transform**:
   - Fourier Transform is a mathematical transformation that converts an image from its spatial domain (pixel values) into the frequency domain. It decomposes an image into its constituent sinusoidal components.
   - In the context of 2D images, the Fourier Transform generates a representation of the image in terms of amplitude (magnitude) and phase information at different spatial frequencies. The amplitude represents how much of a particular frequency is present in the image.
   - For an image, you get a 2D representation in the frequency domain, where the axes correspond to the frequency components (horizontal and vertical frequencies). This is often visualized as a spectrum.

3. **Frequency Analysis Intuition**:
   - Think of an image as a composition of various patterns and textures. High-frequency components represent the fine details, like the edges of objects, while low-frequency components represent the broader structures.
   - By analyzing and manipulating these frequency components, you can emphasize or suppress certain patterns in the image.

4. **Applications**:
   - **Image Enhancement**: Frequency analysis is used for tasks like image sharpening, where you boost the high-frequency components to make edges and details more prominent.
   - **Filtering**: You can filter out specific frequency ranges to remove noise or enhance certain features.
   - **Data Augmentation**: Introducing controlled variations in the frequency domain can create diverse data for training deep learning models, as you can add noise or perturbations consistent with real-world variations.

5. **Frequency Domain Manipulation**:
   - When applying data augmentation using frequency analysis, you can alter the amplitude or phase of specific frequency components to generate new images.
   - For example, you might attenuate high-frequency components to create smoother images or add noise to simulate variations consistent with different image generators.

6. **Fourier Transform in Practice**:
   - In practice, software libraries like OpenCV and NumPy in Python provide functions for applying Fourier Transforms to images.
   - The transformed image can be modified in the frequency domain (amplitude and phase) and then converted back to the spatial domain using the Inverse Fourier Transform.

In the context of your project, frequency analysis can be a powerful tool for generating diverse and realistic augmented data. By manipulating the frequency components of images, you can introduce controlled variations that are consistent with real-world variations across different generators. This can potentially improve the model's ability to detect AI-generated fake images, even when tested on images from generators not seen during training.

---

## Gameplan: 

- **Replicate Paper's Experiment**: Train 8 different Swin-T models on 8 different image generators and establish a baseline performance.
- **Initial Testing**: Evaluate the baseline models on cross-generator accuracy.
- **Frequency Domain Augmentation**: Implement data augmentation techniques in the frequency domain to improve cross-generator accuracy.
	- Convert images to frequency domain
	- Apply transformations
	- Convert them back to spatial domain

---

## Research Paper Reference 

![[GenImage_paper.pdf]]

---
## The Dataset 

### Summary 

- Total images in GenImage: 2,681,167 (1,331,167 real, 1,350,000 fake).
	- Real images have 1,281,167 for training and 50,000 for testing.
	- Each of the 1000 ImageNet classes has 1350 generated images (1300 images are for training and 50 for testing.)
- Uses eight generative models for image generation
- Each generator produces roughly equal images per class (162 for training and 6 for testing), except for Stable Diffusion V1.5 (166 for training and 8 for testing).
- Real images are unique to each subset (e.g., Stable Diffusion V1.4 subset) and are not reused.
- Image generation strives for balance, avoiding biases from imbalanced generator outputs.
- Image generation templates: "photo of class", with "class" being ImageNet labels.
	- Wukong uses Chinese translations for better quality.
	- ADM and BIGGAN use pre-trained models on ImageNet.

---

## Image Generators

### Diffusion Models 

- [Midjourney (V5)](https://docs.midjourney.com/docs/model-versions)
- [Wukong](https://arxiv.org/abs/2202.06767)
- [Stable Diffusion (V1.4)](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [Stable Diffusion (V1.5)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [ADM](https://openreview.net/pdf?id=AAWuCvzaVt)
- [GLIDE](https://paperswithcode.com/method/glide)
- [VQDM](https://arxiv.org/abs/2111.14822)

### GAN Models 

- [Big-GAN](https://paperswithcode.com/method/biggan)

---
## Detectors 

### Backbone Model 

- These are basic models used for binary classification to detect real versus fake images.
- Examples include [*Resnet-50*](https://datagen.tech/guides/computer-vision/resnet-50/) and [*DeiT-S*](https://arxiv.org/abs/2012.12877) & [*Swin-T*](https://arxiv.org/abs/2103.14030) (both based on Transformers).
- They are considered baseline methods since they don't have specific designs tailored to fake image detection.
### Fake Face Detector 

- These models are specifically designed for detecting forged face images.
- [*F3Net*](https://arxiv.org/abs/1911.11445) analyzes frequency components and differences in frequency statistics between real and fake face images.
- [*GramNet*](https://arxiv.org/abs/2002.00133) leverages global texture features to enhance the robustness and generalizability of fake face detection.
- While effective for face images, their performance may not be as strong for non-face images.
### General Fake Image Detector 

- Designed to classify a broad range of images, not just faces.
- [*Spec*](https://arxiv.org/pdf/1907.06515.pdf) uses the frequency spectrum as input and identifies GAN-induced artifacts in real images without needing specific GAN models to produce training data.
- [*CNNSpot*](https://symposium.cshlp.org/content/82/57.full.pdf) employs ResNet-50 for binary classification and includes special pre-processing, post-processing, and data augmentation techniques.
- Existing methods need enhancement, especially when dealing with images generated by a mix of GANs and diffusion models.

---
## Task 1: Cross-Generator Image Classification 

### Summary 

- The study used the *ResNet-50* model to evaluate the *GenImage* dataset.
- GenImage has 8 subsets from different image generators.
- Training and testing on the same subset often achieved >98.5% accuracy; some subsets even reached 99.9%.
- Performance dropped when trained on one generator and tested on another (e.g., 54.9% accuracy between *Stable Diffusion V1.4* and *Midjourney*).
- Authors proposed training on a single generator and testing on multiple to test general fake detection.
- Models, both CNNs like *ResNet* and Transformers like *DeiT-S* and *Swin-T*, had similar results.
- *CNNSpot* and *Spec* performed well on their datasets but not as well on GenImage.
- *F3Net* and *GramNet* introduced unique detection techniques.
- A basic *ResNet-50* binary classification was more effective than other methods.

> Main takeaway: There's a need for a specially designed model to effectively detect fake images in the GenImage dataset.


--- 
## Task 2: Degraded Image Classification
### Summary 

- Explored detectors' robustness against image degradation issues like low resolution, compression, and noise.
- Trained detectors on *Stable Diffusion V1.4* subset, then degraded testing set images through downsampling, JPEG compression, and Gaussian Blurring.
- Baseline models like *ResNet-50*, *DeiT-S*, and *Swin-T* performed similarly, but struggled with very low resolutions and JPEG compression.
- *CNNSpot* was resilient to JPEG compression and Gaussian blurring due to its training preprocessing.
- Detectors' performance on degraded images offers insights on their real-world applicability.

---
## Additional GenImage Dataset Analysis & Characteristics 

### Summary

- Analyzed GenImage dataset's characteristics and its effectiveness.
- **Effect of Increasing the Number of Images:** Larger datasets improve classification model performance.
- **Frequency Analysis:** Studied artifacts in images using Fourier transform; diffusion model images resemble real images more than GAN.
- **Image Class Generalization:** Training on a subset of classes can generalize to other classes, with more classes leading to better performance.
- **Generator Correlation Analysis:** Cross-generator performance better between similar generators (e.g., *Stable Diffusion V1.4* and *Stable Diffusion V1.5*).
- **Image Content Generalization:** *ResNet-50* trained on GenImage can detect fake face and art images with high accuracy (>95%).

---

## Potential Improvements with Transfer Learning

For AI-generated image detection, leveraging transfer learning can potentially enhance the generalization capabilities of binary classifiers. The core intuition behind using transfer learning in this context is that knowledge gained while learning one generator can aid performance on other generators. However, care must be taken to balance the trade-off between leveraging prior knowledge and adapting to new data.

### 1. Select a Pre-trained Model:
Start with a model that's already been trained on a specific image generator. This pre-trained model will serve as a solid foundation due to its prior knowledge.

### 2. Data Collection and Augmentation:
- Gather datasets from other image generators. Ensure diversity in the datasets to enhance generalization.
- Augment the data by introducing variations such as rotations, translations, and other forms of distortions. This can help the model become invariant to these changes.

### 3. Fine-tuning:
- Instead of training the model from scratch, use the pre-trained model and further train it (fine-tune) on datasets from other generators.
- It might be beneficial to fine-tune only the top layers of the model while keeping the initial layers frozen. The rationale is that initial layers capture generic features, while the top layers are more specialized.

### 4. Regularization:
To prevent overfitting during fine-tuning, employ techniques such as dropout, early stopping, or weight decay. This ensures the model generalizes well to images from unseen generators.

### 5. Evaluation and Iteration:
- Continuously evaluate the model's performance on a diverse set of generators.
- If the model's performance on a particular generator isn't satisfactory, consider collecting more data or adjusting the fine-tuning process for that generator.

### Challenges:
- **Data Imbalance**: Some generators might produce images that are rare or unique, leading to class imbalances.
- **Overfitting**: While fine-tuning, the model might overfit to the new data and lose its capability to generalize on the original data.
- **Complexity**: Introducing data from multiple generators can increase the complexity of the training process.

### Best Practices:
- **Curriculum Learning**: Start fine-tuning with easier datasets and gradually introduce more complex datasets. This can help in smoother convergence.
- **Model Ensembling**: Train multiple models and ensemble their predictions. This can potentially improve accuracy and robustness.
- **Continuous Evaluation**: Regularly test the model on unseen data to ensure it maintains a good generalization capability.



