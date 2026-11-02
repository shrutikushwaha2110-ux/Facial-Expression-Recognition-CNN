# Reading Faces: Facial Expression Recognition Using CNNs

**Group:** 3  
**Institution:** Atria University  
**Course:** Neural Network and Convolutional Neural Network Sprint

**Team Members:**
- Ashmita
- Revathi
- Meghna K
- Shruti (Team Leader)

**GitHub Repository:** [Add your public GitHub repository link]

---

# 1. Abstract

Facial Expression Recognition is a computer vision task that involves identifying facial expressions from images. This project aims to develop a Convolutional Neural Network (CNN) capable of classifying facial images into seven emotion categories using the FER-2013 dataset.

The project involves dataset exploration, image preprocessing, CNN development, model training, and performance evaluation. The dataset contains approximately 35,887 grayscale facial images belonging to seven emotion categories: angry, disgust, fear, happy, sad, surprise, and neutral.

The work begins with Exploratory Data Analysis (EDA) to understand the dataset, identify class imbalance, and examine sample images. The images are then preprocessed and prepared for CNN model development. Different model architectures and training techniques will be investigated to improve classification performance.

**Note:** Update this abstract after completing model training and experiments. Include the actual results and best-performing model.

# 2. Introduction

Facial expressions are an important form of non-verbal communication. People use facial movements to express different emotions such as happiness, sadness, anger, fear, and surprise.

Facial Expression Recognition uses computer vision and deep learning techniques to identify these expressions automatically from images.

However, recognizing facial expressions is challenging because of variations in lighting, facial features, image quality, expression intensity, and similarities between different emotions.

In this project, we explore the use of Convolutional Neural Networks (CNNs) to classify facial images into seven emotion categories using the FER-2013 dataset.

# 3. Problem Statement

The objective of this project is to develop a CNN-based system that can classify a 48 × 48 grayscale facial image into one of seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The project also investigates the challenges of facial expression classification, including class imbalance, noisy labels, and similarities between facial expressions.

# 4. Project Objectives

The main objectives are:

1. Understand and explore the FER-2013 dataset.
2. Perform Exploratory Data Analysis to understand the distribution and characteristics of the images.
3. Preprocess the images and prepare the dataset for CNN training.
4. Develop a custom CNN architecture for facial expression classification.
5. Experiment with different model configurations and training techniques.
6. Compare a custom CNN with a transfer learning approach.
7. Evaluate model performance using appropriate metrics.
8. Analyze model errors and understand its limitations.

# 5. Dataset Description

## 5.1 Dataset Overview

The project uses the FER-2013 (Facial Expression Recognition 2013) dataset.

**Dataset source:** https://www.kaggle.com/datasets/msambare/fer2013

The dataset contains approximately 35,887 grayscale facial images. Each image has a resolution of 48 × 48 pixels.

The images are organized into seven emotion categories.

| Emotion | Description |
|---|---|
| Angry | Facial expression showing anger |
| Disgust | Facial expression showing disgust |
| Fear | Facial expression showing fear |
| Happy | Facial expression showing happiness |
| Sad | Facial expression showing sadness |
| Surprise | Facial expression showing surprise |
| Neutral | Facial expression without a strong visible expression |

## 5.2 Dataset Distribution

The original dataset contains:

| Dataset | Number of Images |
|---|---:|
| Training | 28,709 |
| Testing | 7,178 |
| **Total** | **35,887** |

The original training dataset was further divided into training and validation sets.

## 5.3 Dataset Integrity Verification

All 35,887 images were checked for image dimensions, formats, and file integrity.

The following observations were made:

- All images had a resolution of 48 × 48 pixels.
- All images were stored in grayscale format.
- No images with incorrect dimensions were found.
- No corrupted images were detected.

Therefore, no image resizing or removal of corrupted images was required.

# 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the characteristics and distribution of the FER-2013 dataset before model development.

## 6.1 Class Distribution

The number of images in each emotion category was examined to understand the distribution of the dataset.

**Training dataset class distribution:**

| Emotion | Number of Images |
|---|---:|
| Angry | 3,995 |
| Disgust | 436 |
| Fear | 4,097 |
| Happy | 7,215 |
| Sad | 4,830 |
| Surprise | 3,171 |
| Neutral | 4,965 |
| **Total** | **28,709** |

**Observations:**

- The happy category contains the highest number of images.
- The disgust category contains the lowest number of images.
- The number of images is not equally distributed across all seven categories.
- The dataset therefore exhibits class imbalance.

**Inference:**

Class imbalance may affect model performance because the model receives more examples of certain emotions than others. Therefore, accuracy alone may not fully represent performance across all emotion categories. Metrics such as precision, recall, and macro F1-score are also important.

**Figure 1: Class Distribution**

[Insert the class distribution bar chart here.]

## 6.2 Sample Images from Each Emotion Category

Sample images were displayed from all seven emotion categories to visually understand the dataset.

Five sample images were selected from each category.

**Observations:**

- Images contain variations in lighting and facial features.
- The intensity of facial expressions differs between images.
- Some emotion categories have visually similar expressions.
- These variations make facial expression classification challenging.

**Figure 2: Sample Images from Seven Emotion Categories**

[Insert the sample image grid here.]

## 6.3 Average Face for Each Emotion

A pixel-wise average face was calculated for each emotion category using all images belonging to that category in the original training dataset.

The average face was obtained by calculating the mean pixel value at each corresponding position across the images in a category.

**Observations:**

- The resulting average faces appear blurred.
- This happens because different facial features and expressions are combined through pixel-wise averaging.
- The average faces show general visual patterns within each emotion category.
- They do not represent an actual individual face.

**Figure 3: Average Face for Each Emotion**

[Insert the seven average face visualizations here.]

# 7. Data Preprocessing

Data preprocessing was performed to prepare the images and labels for CNN model development.

## 7.1 Training and Validation Split

The original training dataset contained 28,709 images.

It was divided into training and validation sets using an 80:20 ratio.

A stratified split was used to maintain approximately the same proportion of each emotion category in both sets.

| Dataset | Number of Images |
|---|---:|
| Training | 22,967 |
| Validation | 5,742 |
| Testing | 7,178 |

The original testing dataset was kept separate and was not used during training or validation.

## 7.2 Stratified Class Distribution

The resulting training and validation distributions are:

| Emotion | Training | Validation |
|---|---:|---:|
| Angry | 3,196 | 799 |
| Disgust | 349 | 87 |
| Fear | 3,277 | 820 |
| Happy | 5,772 | 1,443 |
| Sad | 3,864 | 966 |
| Surprise | 2,537 | 634 |
| Neutral | 3,972 | 993 |
| **Total** | **22,967** | **5,742** |

Stratification ensures that the training and validation sets maintain approximately the same class proportions as the original training dataset.

**Figure 4: Training and Validation Class Distribution**

[Insert the grouped bar chart here.]

## 7.3 Image Loading and Grayscale Conversion

The images were loaded using the Pillow library and converted to grayscale format.

Each image was represented as a numerical array of pixel values.

The original image dimensions were 48 × 48 pixels.

## 7.4 Image Normalization

The original pixel values ranged from 0 to 255.

To normalize the images, each pixel value was divided by 255.

This converted the pixel values into a range between 0 and 1.

Normalization helps provide consistent input values for CNN training.

## 7.5 Channel Dimension

A channel dimension was added to each image to make the data compatible with the expected CNN input format.

The final image shape became:

**48 × 48 × 1**

Here:
- 48 represents image height.
- 48 represents image width.
- 1 represents the grayscale channel.

## 7.6 Label Encoding

The seven emotion categories were converted into numerical labels so that they could be used during model training.

| Emotion | Encoded Label |
|---|---:|
| Angry | 0 |
| Disgust | 1 |
| Fear | 2 |
| Happy | 3 |
| Sad | 4 |
| Surprise | 5 |
| Neutral | 6 |

## 7.7 Final Preprocessed Dataset

After preprocessing, the final dataset had the following shapes:

| Dataset | Image Shape | Label Shape |
|---|---|---|
| Training | (22,967, 48, 48, 1) | (22,967,) |
| Validation | (5,742, 48, 48, 1) | (5,742,) |
| Testing | (7,178, 48, 48, 1) | (7,178,) |

All image-label counts and image dimensions were verified.

The pixel values were also checked to ensure that they were within the range of 0 to 1.

The preprocessed dataset was saved as a compressed NumPy file named:

`fer2013_preprocessed.npz`

This file contains the training, validation, and testing images along with their encoded labels. It is ready to be used for CNN model development.

# 8. Methodology

**To be completed after the CNN and transfer learning implementations.**

This section will describe:

- Custom CNN architecture.
- Convolutional layers, Batch Normalization, Max Pooling, and Dropout.
- Loss function and optimizer.
- Training configuration and hyperparameters.
- Data augmentation.
- Class imbalance handling.
- Transfer learning architecture.

[Insert the CNN architecture diagram here.]

# 9. Experiments and Results

**To be completed after model experiments.**

This section will include:

- Baseline CNN results.
- Network depth experiments.
- Batch Normalization experiments.
- Dropout experiments.
- Optimizer comparison.
- Data augmentation comparison.
- Class-weighted loss versus oversampling.
- Transfer learning comparison.

[Insert experimental result tables and learning curves here.]

# 10. Model Evaluation and Error Analysis

**To be completed after model evaluation.**

This section will include:

- Test accuracy.
- Per-class precision, recall, and F1-score.
- Confusion matrix.
- Misclassified images.
- Analysis of common classification errors.

# 11. Explainability

**To be completed after explainability experiments.**

This section will describe the use of Grad-CAM or other visualization techniques to understand which image regions influence the model's predictions.

[Insert explainability visualizations here.]

# 12. Limitations and Ethical Considerations

Some limitations and ethical considerations associated with facial expression recognition include:

- **Class imbalance:** Some emotion categories have significantly fewer images than others.
- **Noisy labels:** Some images may have ambiguous or incorrectly assigned emotion labels.
- **Similar expressions:** Different emotions can have visually similar facial features.
- **Dataset limitations:** The model learns from the available dataset and may not generalize equally well to all real-world situations.
- **Potential bias:** Model performance may vary across different demographic groups.
- **Privacy and consent:** Facial images should be collected and used responsibly, with appropriate consent.
- **Interpretation of emotions:** A predicted facial expression should not be treated as definitive evidence of a person's actual emotional state.

# 13. Conclusion

**To be completed after all experiments.**

Summarize the project, the models developed, the major findings, the best-performing approach, and possible improvements.

# 14. Team Contributions

| Team Member | Responsibilities | Contributions |
|---|---|---|
| Ashmita | [Confirm assigned role] | [Add completed work] |
| Revathi | [Confirm assigned role] | [Add completed work] |
| Meghna K | Dataset preparation and EDA | Dataset inspection, EDA, stratified splitting, preprocessing, label encoding, and preparation of the final dataset |
| Shruti | [Confirm assigned role] | [Add completed work] |

# 15. References

1. FER-2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Keras Documentation: https://keras.io/
4. NumPy Documentation: https://numpy.org/doc/
5. Matplotlib Documentation: https://matplotlib.org/stable/
6. [Add research papers and other references used by the team.]


