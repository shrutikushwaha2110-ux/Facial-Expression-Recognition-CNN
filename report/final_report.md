# Reading Faces: Facial Expression Recognition Using CNNs

**Group:** 3  
**Institution:** Atria University  
**Course:** Neural Network and Convolutional Neural Network Sprint

## Team Members

- Ashmita
- Revathi
- Meghna K
- Shruti (Team Leader)

**GitHub Repository:** [Add your public GitHub repository link]

---

# 1. Abstract

Facial Expression Recognition is a computer vision task that involves identifying facial expressions from images. This project aims to develop a Convolutional Neural Network (CNN) capable of classifying facial images into seven emotion categories using the FER-2013 dataset.

The project involves dataset exploration, image preprocessing, CNN development, model training, and performance evaluation. The dataset contains 35,887 grayscale images belonging to seven emotion categories: angry, disgust, fear, happy, sad, surprise, and neutral.

The work begins with Exploratory Data Analysis (EDA) to understand the dataset, identify class imbalance, and examine sample images. The images are then cleaned and preprocessed before being used for CNN development.

Different CNN architectures and training techniques will be explored to understand how well the model can recognize facial expressions.

**Note:** This abstract will be updated after completing model training and experiments, with the actual results and best-performing model.

---

# 2. Introduction

Facial expressions are an important form of non-verbal communication. People use facial movements to express emotions such as happiness, sadness, anger, fear, and surprise.

Facial Expression Recognition uses computer vision and deep learning techniques to identify these expressions automatically from images.

However, recognizing facial expressions is challenging because of variations in lighting, facial features, image quality, expression intensity, and similarities between different emotions.

In this project, we explore the use of Convolutional Neural Networks (CNNs) to classify facial images into seven emotion categories using the FER-2013 dataset.

---

# 3. Problem Statement

The objective of this project is to develop a CNN-based system that can classify a 48 × 48 grayscale facial image into one of seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The project also investigates the challenges of facial expression classification, including class imbalance, noisy labels, duplicate images, and similarities between facial expressions.

---

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

---

# 5. Dataset Description

## 5.1 Dataset Overview

The project uses the FER-2013 (Facial Expression Recognition 2013) dataset.

**Dataset Source:** Kaggle – FER-2013 dataset (MSambare).

The dataset contains 35,887 grayscale facial images. Each image has a resolution of 48 × 48 pixels.

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

The original training dataset was cleaned and further divided into training and validation sets.

After removing 11 completely black training images, the cleaned training dataset contained 28,698 images.

The official testing dataset was preserved with all 7,178 images.

## 5.3 Dataset Integrity and Quality Verification

All 35,887 original images across the training and testing folders were audited for image dimensions, color formats, unreadable files, completely black images, duplicate content, and label consistency.

The following observations and findings were made.

### 1. Image Dimensions and Color Mode

- All images have a resolution of 48 × 48 pixels.
- All images are single-channel grayscale (mode = 'L').
- No unreadable image files were detected during verification.

### 2. Blank and Dark Image Analysis

Pixel-intensity analysis identified 12 completely black images where every pixel value was exactly zero.

- **Training:** 11 completely black images.
- **Testing:** 1 completely black image.

The 11 completely black training images were removed because they contained no visible facial information.

The completely black testing image was retained to preserve the official test set.

Naturally dark images with non-zero pixel values and visible facial information were preserved.

After cleaning:

- Training images: 28,698.
- Testing images: 7,178.

### 3. Duplicate Image Analysis

Duplicate images were identified using SHA-256 cryptographic hashes calculated from decoded pixel content.

The analysis revealed:

- **Training:** 1,034 duplicate groups containing 2,260 images.
- **Testing:** 81 duplicate groups containing 167 images.
- **Redundant training copies:** 1,226.

The analysis also identified identical image content across the original training and testing datasets.

### 4. Train-Test Duplicate Overlap

A total of 530 unique image hashes were found in both the original training and testing datasets.

These shared hashes affected:

- Training instances: 692.
- Testing instances: 567.

This indicates pre-existing duplicate overlap between the original dataset partitions. It is a potential source of data leakage and should be considered when interpreting model evaluation results.

### 5. Conflicting Emotion Labels

Some identical images were associated with different emotion labels.

The analysis identified:

- **Training:** 38 duplicate groups containing 96 images with conflicting labels.
- **Testing:** 4 duplicate groups containing 9 images with conflicting labels.
- **Train-test overlap:** 24 shared duplicate groups with conflicting labels.

For example, identical image content may be assigned different labels, such as Angry and Sad, or Fear and Surprise.

These findings indicate possible annotation inconsistencies and ambiguity in the dataset.

The original emotion labels and duplicate images were not arbitrarily modified or removed. Instead, these findings were documented as potential sources of label noise.

---

# 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the characteristics and distribution of the FER-2013 dataset before model development.

## 6.1 Class Distribution

The number of images in each emotion category was examined to understand the distribution of the original training dataset.

**Original training dataset class distribution:**

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

- The Happy category contains the highest number of images.
- The Disgust category contains the lowest number of images.
- The number of images is not equally distributed across all seven categories.
- The dataset therefore exhibits class imbalance.

**Inference:**

Class imbalance may affect model performance because the model receives more examples of certain emotions than others. Therefore, accuracy alone may not fully represent performance across all emotion categories. Metrics such as precision, recall, and macro F1-score are also important.

**Figure 1: Class Distribution**

[Insert the class distribution bar chart here.]

## 6.2 Sample Images from Each Emotion Category

Sample images were displayed from all seven emotion categories to visually understand the dataset.

Five sample images were selected from each category using reproducible random sampling.

**Observations:**

- Images contain variations in lighting and facial features.
- The intensity of facial expressions differs between images.
- Some emotion categories have visually similar expressions.
- These variations make facial expression classification challenging.

**Figure 2: Sample Images from Seven Emotion Categories**

[Insert the sample image grid here.]

## 6.3 Average Face for Each Emotion

A pixel-wise average face was calculated for each emotion category using the training images.

The average face was obtained by calculating the mean pixel value at each corresponding position across the images in a category.

**Observations:**

- The resulting average faces appear blurred.
- This happens because different facial features and expressions are combined through pixel-wise averaging.
- The average faces show general visual patterns within each emotion category.
- They do not represent an actual individual face.

**Figure 3: Average Face for Each Emotion**

[Insert the seven average face visualizations here.]

---

# 7. Data Preprocessing

Data preprocessing was performed to clean the images, create training and validation datasets, and format the data for CNN training.

## 7.1 Cleaning and Stratified Group Split

The following steps were performed:

**1. Completely Black Image Removal**

The 11 completely black training images were removed, leaving 28,698 training images.

Naturally dark images containing visible facial information were preserved.

The official testing dataset was retained with all 7,178 images.

**2. Stratified Group-Aware Split**

The cleaned training dataset was divided into training and validation sets using:

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`

Images with identical pixel hashes were kept within the same group.

This prevents identical image content from appearing in both the newly created training and validation sets while maintaining approximately similar class distributions.

**3. Split Summary**

| Dataset Split | Images | Role |
|---|---:|---|
| Training | 22,949 | Model training |
| Validation | 5,749 | Model evaluation and hyperparameter tuning |
| Testing | 7,178 | Final model evaluation |
| **Total** | **35,876** | — |

The official test set was preserved without modification.

However, the original training and testing datasets contain pre-existing duplicate overlap, as identified during the data quality analysis. This limitation is documented and should be considered during model evaluation.

## 7.2 Stratified Class Distribution

Stratified group splitting was used to maintain approximately similar class distributions across the training and validation datasets.

The distribution of all seven emotion categories was checked after splitting.

The final class counts should be taken from the latest preprocessed arrays generated by the notebook.

The class distribution is important because some emotions, particularly Disgust, have considerably fewer images than other categories.

Class imbalance will be considered during model training and experimentation.

**Figure 4: Training and Validation Class Distribution**

[Insert the updated grouped bar chart here.]

## 7.3 Image Loading and Grayscale Normalization

The following preprocessing steps were applied:

**1. Array Conversion**

Each 48 × 48 image was loaded using Pillow and converted into a NumPy array with the `float32` data type.

**2. Pixel Normalization**

Original pixel values range from 0 to 255.

These values were divided by 255 to scale them into the range [0.0, 1.0].

This helps provide consistent input values during model training.

**3. Channel Dimension**

A single-channel dimension was added to each grayscale image.

The final image shape is:

`(48, 48, 1)`

This format is suitable for CNN input.

## 7.4 Numerical Label Encoding

The seven categorical emotion labels were encoded into consecutive integers from 0 to 6.

| Emotion | Encoded Integer |
|---|---:|
| Angry | 0 |
| Disgust | 1 |
| Fear | 2 |
| Happy | 3 |
| Sad | 4 |
| Surprise | 5 |
| Neutral | 6 |

These numerical labels are used during model training and evaluation.

## 7.5 Final Preprocessed Dataset Specifications

After preprocessing, all arrays were verified for correct dimensions, data types, normalized pixel values, and class coverage.

| Partition | Samples | Image Tensor Shape | Label Array Shape | Data Type | Pixel Value Range |
|---|---:|---|---|---|---|
| Training | 22,949 | `(22949, 48, 48, 1)` | `(22949,)` | float32 / int64 | [0.0, 1.0] |
| Validation | 5,749 | `(5749, 48, 48, 1)` | `(5749,)` | float32 / int64 | [0.0, 1.0] |
| Testing | 7,178 | `(7178, 48, 48, 1)` | `(7178,)` | float32 / int64 | [0.0, 1.0] |
| **Total** | **35,876** | — | — | — | — |

All three partitions contain the seven emotion classes, represented by labels from 0 to 6.

The preprocessed arrays were saved in a compressed archive:

`processed/fer2013_preprocessed.npz`

The saved file was reloaded, and the arrays were compared with the original arrays to verify that the exported data was preserved correctly.

All five comprehensive verification check suites passed successfully.

The verified dataset is ready for:

**Phase 2: Custom CNN Architecture Development and Baseline Training.**

---

# 8. Methodology

This section will be completed after the CNN and transfer learning implementations.

It will describe:

- Custom CNN architecture.
- Convolutional layers, Batch Normalization, Max Pooling, and Dropout.
- Activation functions.
- Loss function and optimizer.
- Training configuration and hyperparameters.
- Data augmentation techniques.
- Class imbalance handling.
- Transfer learning architecture.

[Insert the CNN architecture diagram here.]

---

# 9. Experiments and Results

This section will be completed after model experiments.

The following experiments will be considered:

- Baseline CNN results.
- Network depth experiments.
- Batch Normalization experiments.
- Dropout experiments.
- Optimizer comparison.
- Data augmentation comparison.
- Class-weighted loss versus oversampling.
- Transfer learning comparison.

[Insert experimental result tables and learning curves here.]

---

# 10. Model Evaluation and Error Analysis

This section will be completed after model evaluation.

It will include:

- Test accuracy.
- Per-class precision, recall, and F1-score.
- Macro F1-score.
- Confusion matrix.
- Misclassified images.
- Analysis of common classification errors.

The evaluation will consider the class imbalance and pre-existing train-test duplicate overlap identified during Phase 1.

---

# 11. Explainability

This section will be completed after explainability experiments.

It will describe the use of Grad-CAM or other visualization techniques to understand which image regions influence the model's predictions.

[Insert explainability visualizations here.]

---

# 12. Limitations and Ethical Considerations

The FER-2013 dataset and facial expression recognition task have several limitations.

### 1. Class Imbalance

Some emotion categories have significantly fewer images than others. For example, Disgust has considerably fewer samples than Happy.

This may affect the model's ability to recognize underrepresented emotions.

Class imbalance will be considered during model training using suitable techniques.

### 2. Noisy and Ambiguous Labels

Some images may have ambiguous or incorrectly assigned emotion labels.

Duplicate analysis also revealed identical images associated with different emotion categories.

These inconsistencies were documented without arbitrarily changing the original labels.

### 3. Duplicate Images

Multiple identical images are present within the original training and testing datasets.

Although group-aware splitting prevents duplicate overlap between the newly created training and validation sets, duplicates still exist within the original dataset.

### 4. Pre-existing Train-Test Overlap

The original training and testing datasets share 530 unique image hashes.

This overlap may affect the interpretation of final test results and is documented as a limitation of the original dataset.

### 5. Lighting and Image Quality

Images contain variations in brightness, facial appearance, and image quality.

Although completely black training images were removed, naturally dark images were preserved.

### 6. Low Image Resolution

The dataset contains images of only 48 × 48 pixels.

This limits the amount of fine facial detail available to the CNN.

### 7. Potential Bias and Generalization

The model learns from the images available in the dataset and may not perform equally well across all demographic groups or real-world conditions.

### 8. Privacy and Consent

Facial images should be collected and used responsibly, with appropriate consent and respect for individual privacy.

### 9. Interpretation of Emotions

A predicted facial expression should not be treated as definitive evidence of a person's actual emotional state. Facial expressions do not always represent how someone truly feels.

---

# 13. Conclusion

This section will be completed after all model development and experiments.

The final conclusion will summarize:

- The dataset and preprocessing work.
- The CNN architectures developed.
- The experiments performed.
- The best-performing model.
- Model evaluation results.
- Major findings and limitations.
- Possible improvements and future work.

---

# 14. Team Contributions

| Team Member | Responsibilities | Contributions |
|---|---|---|
| Ashmita | To be confirmed | To be updated based on completed work |
| Revathi | To be confirmed | To be updated based on completed work |
| Meghna K | Dataset preparation and EDA | Dataset inspection, EDA, duplicate analysis, data quality checks, stratified group splitting, preprocessing, label encoding, and preparation of the final dataset |
| Shruti | Team Leader | To be updated based on completed work |

*Note: Team responsibilities and contributions should be updated according to the actual work completed by each member.*

---

# 15. References

1. FER-2013 Dataset – Kaggle (MSambare).
2. TensorFlow Documentation.
3. Keras Documentation.
4. NumPy Documentation.
5. Matplotlib Documentation.
6. Pillow Documentation.
7. Scikit-learn Documentation.
8. Additional research papers and resources used during model development.

---

**Project:** Reading Faces – Facial Expression Recognition Using CNNs  
**Group:** 3  
**Institution:** Atria University