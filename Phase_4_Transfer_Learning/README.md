# Facial Expression Recognition (FER) - MobileNetV2 Fine-Tuning Pipeline

## Project Overview

This repository contains the transfer learning and fine-tuning pipeline for a Facial Emotion Recognition (FER) system using MobileNetV2. The project is structured as a collaborative team workflow where one teammate handles model fine-tuning and validation, while another handles final test-set evaluation and explainability (Grad-CAM).

## Dataset & Leakage Prevention

* **Dataset**: FER2013 preprocessed and verified for data leakage prevention (`fer2013_preprocessed_leakage_checked.npz.zip`).
* **Data Splits**:
* **Training Set**: 22,949 samples

* **Validation Set**: 5,749 samples
* **Test Set**: 6,611 samples (strictly reserved and pristine for teammate evaluation)

## Model Architecture & Fine-Tuning Strategy

* **Base Model**: MobileNetV2 pre-trained on ImageNet weights, configured with an input shape of $96\times96\times3$

* **Input Preprocessing**: Grayscale images are resized to $96\times96$, replicated across 3 RGB channels, and normalized.
* **Classification Head**:
* `GlobalAveragePooling2D()`
* `Dense(256, activation='relu')`
* `Dropout(0.5)`
* `Dense(7, activation='softmax')` for 7 emotion classes (*Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*).

* **Fine-Tuning Configuration**:
* Unfroze the top 20 layers of MobileNetV2 while keeping lower layers frozen.
* Froze all `BatchNormalization` layers to stabilize training and prevent overfitting.
* Computed balanced class weights to manage class imbalance across emotion categories.

## Training & Callbacks
* **Optimizer**: Adam with a low learning rate of `0.0001`.
* **Loss Function**: Sparse categorical crossentropy.
* **Callbacks**:
* `EarlyStopping`: Monitors `val_loss` with a patience of 5 epochs, restoring the best weights.
* `ReduceLROnPlateau`: Reduces learning rate by a factor of 0.5 when `val_loss` plateaus for 2 epochs (minimum LR: `0.00001`).
* **Training Performance**: Reached a final training accuracy of ~81.29% and a validation accuracy of ~56.32% by epoch 14.

## Validation Results
Evaluated exclusively on the validation set (`X_val`) to preserve the test set:
* **Validation Loss**: 1.1928
* **Validation Accuracy**: 56.32%
* **Macro Precision**: 0.5315
* **Macro Recall**: 0.5365
* **Macro F1 Score**: 0.5286



### Validation Classification Report:

```text
              precision    recall  f1-score   support

       Angry       0.51      0.37      0.43       797
     Disgust       0.45      0.51      0.48        89
        Fear       0.46      0.31      0.37       815
       Happy       0.72      0.81      0.76      1441
         Sad       0.45      0.45      0.45       972
    Surprise       0.66      0.74      0.69       646
     Neutral       0.47      0.58      0.52       989

    accuracy                           0.56      5749
   macro avg       0.53      0.54      0.53      5749
weighted avg       0.55      0.56      0.55      5749

```

## Team Artifacts & Hand-off

* **Saved Model**: `/content/fer2013_mobilenet_finetuned_best.keras` (passed to teammate for final test evaluation and Grad-CAM explainability).
* **Generated Reports**: `classification_report_val.txt` and `confusion_matrix_val.png`.
