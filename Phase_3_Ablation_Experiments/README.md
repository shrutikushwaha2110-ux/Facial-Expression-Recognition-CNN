
# Phase 3: CNN Ablation Experiments

## 1. Overview
This phase focuses on building and evaluating a custom Convolutional Neural Network (CNN)
for facial expression recognition using the FER-2013 dataset.

The objective is to understand how different CNN configurations and training techniques
affect model performance.

## 2. Dataset
- Dataset: FER-2013
- Image size: 48 x 48 pixels
- Image type: Grayscale
- Number of classes: 7

Classes:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

## 3. CNN Architecture
A custom CNN was developed from scratch using TensorFlow and Keras.

The architecture includes:
- Convolutional layers
- Batch Normalization
- Max Pooling
- Flatten layer
- Fully connected Dense layer
- Dropout
- Softmax output layer

No pretrained model was used for this CNN.

## 4. Ablation Experiments

The following experiments were conducted:

1. CNN depth: 2, 3 and 4 convolutional blocks.
2. Batch Normalization: enabled vs disabled.
3. Dropout: 0.3, 0.5 and 0.7.
4. Optimizer: Adam vs SGD with momentum.
5. Data Augmentation: enabled vs disabled.
6. Class balancing: class weights vs random oversampling.

## 5. Key Experimental Results

| Experiment | Configuration | Validation Accuracy | Macro F1 |
|---|---|---:|---:|
| Depth | 2 blocks | 47.40% | 0.3577 |
| Depth | 3 blocks | 50.74% | 0.3933 |
| Depth | 4 blocks | 49.07% | 0.3979 |
| Batch Normalization | ON | 52.90% | 0.4278 |
| Batch Normalization | OFF | 54.62% | 0.4610 |
| Dropout | 0.3 | 55.84% | 0.4605 |
| Dropout | 0.5 | 51.28% | 0.4141 |
| Dropout | 0.7 | 25.07% | 0.0573 |
| Optimizer | Adam | 52.60% | 0.4295 |
| Optimizer | SGD + Momentum | 53.77% | 0.4590 |
| Augmentation | OFF | 51.80% | 0.4499 |
| Augmentation | ON | 56.64% | 0.4804 |
| Class balancing | Class Weights + Augmentation | 39.21% | 0.3111 |
| Class balancing | Oversampling + Augmentation | 58.11% | 0.5654 |

## 6. Final Combined Model

The combined model was trained using:
- 3 convolutional blocks
- Batch Normalization
- Dropout rate: 0.3
- Data Augmentation
- Random Oversampling
- SGD optimizer with momentum

Final retraining validation results:
- Validation Accuracy: 56.79%
- Macro F1 Score: 0.5429
- Best validation accuracy during training: 56.79%

## 7. Results Folder

The results folder contains:
- Evaluation results
- Classification report
- Confusion matrix
- Training history
- Depth experiment results

## 8. Models Folder

The models folder contains the saved CNN models from the different experiments,
including the final combined model.

## 9. Limitations
- Results are based on individual experiment runs.
- Random initialization and training variation may affect performance.
- The experiments were conducted on a validation set.
- The test set has not been used for final model evaluation in this phase.

## 10. Technologies
- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Google Colab
