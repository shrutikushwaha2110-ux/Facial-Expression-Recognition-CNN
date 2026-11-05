# Reading Faces: Facial Expression Recognition Using CNNs

### A Deep Learning Project | FER-2013 | Atria University | Group 3

A facial expression recognition project that uses Convolutional Neural Networks (CNNs) and Transfer Learning to classify facial images into seven emotion categories.

The project explores image preprocessing, custom CNN development, systematic ablation experiments, MobileNetV2 fine-tuning, model evaluation, and explainability.

---

## Project Overview

Facial Expression Recognition (FER) is a computer vision task that identifies visible facial expression patterns from images.

In this project, we develop deep learning models using the FER-2013 dataset, which contains low-resolution grayscale facial images.

The main objective is to classify each facial image into one of seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The project compares a custom CNN built from scratch with a pretrained MobileNetV2 model adapted through fine-tuning.

## Problem Statement

Build a deep learning-based facial expression recognition system that classifies 48 × 48 grayscale facial images into seven emotion categories.

The project investigates how CNN architecture, regularization, data augmentation, class imbalance handling, and transfer learning affect classification performance.

It also studies model errors and uses explainability techniques to understand which facial regions contribute to predictions.

## Dataset

**Dataset:** FER-2013 (Facial Expression Recognition 2013)

**Source:** [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

| Property | Description |
|---|---|
| Image size | 48 × 48 pixels |
| Image format | Grayscale |
| Number of classes | 7 |
| Original images | 35,887 |
| Original training images | 28,709 |
| Original test images | 7,178 |

The dataset contains class imbalance, particularly in the Disgust category, and includes ambiguous or noisy labels.

### Data Cleaning and Preprocessing

The preprocessing workflow includes:

- Image integrity and dimension verification.
- Identification of completely black images.
- Duplicate image detection using SHA-256 hashes.
- Investigation of conflicting labels.
- Group-aware stratified training and validation splitting.
- Pixel normalization.
- Numerical encoding of emotion labels.

The original training partition is divided into:

| Split | Samples |
|---|---:|
| Training | 22,949 |
| Validation | 5,749 |

The cleaned test dataset used in the later transfer-learning workflow contains 6,611 samples.

The original dataset also contains duplicate overlap between its training and test partitions. This limitation is documented and should be considered when interpreting final evaluation results.

**Note:** The raw FER-2013 dataset and preprocessed data archives are not included in this repository. Download or obtain the required dataset files separately.

## Project Workflow

### Phase 1 – Exploratory Data Analysis (EDA)

Notebook: `notebooks/01_eda.ipynb`

This phase explores the dataset before model development.

Work includes:

- Class distribution analysis.
- Sample image visualization for all seven emotions.
- Average face visualization for each emotion.
- Image quality and integrity checks.
- Completely black image detection.
- Duplicate image and label inconsistency analysis.
- Data cleaning and train-validation splitting.
- Image normalization and label encoding.

### Phase 2 – Custom CNN Baseline

Notebook: `notebooks/02_baseline_cnn.ipynb`

A custom CNN is developed from scratch to classify 48 × 48 grayscale images.

The architecture includes:

- Convolutional layers.
- Batch Normalization.
- Max Pooling.
- Flatten layer.
- Fully connected Dense layer.
- Dropout.
- Seven-class Softmax output layer.

The model is trained using training data and monitored using validation data.

Training techniques include learning-rate scheduling and early stopping.

Recorded baseline validation performance:

| Metric | Result |
|---|---:|
| Validation Accuracy | 50.74% |
| Macro F1 Score | 0.3933 |

### Phase 3 – CNN Ablation Experiments

Notebook: `notebooks/03_ablation_experiments.ipynb`

This phase investigates how different CNN configurations and training strategies affect performance.

Experiments include:

1. **Network depth:** 2, 3, and 4 convolutional blocks.
2. **Batch Normalization:** With and without Batch Normalization.
3. **Dropout:** Rates of 0.3, 0.5, and 0.7.
4. **Optimizers:** Adam and SGD with momentum.
5. **Data augmentation:** With and without augmentation.
6. **Class imbalance handling:** Class-weighted loss and random oversampling.

The final combined CNN configuration uses:

- Three convolutional blocks.
- Batch Normalization.
- Dropout rate of 0.3.
- Data augmentation.
- Random oversampling.
- SGD with momentum.

Recorded final combined-model validation results:

| Metric | Result |
|---|---:|
| Validation Accuracy | 56.79% |
| Macro F1 Score | 0.5429 |
| Best Validation Loss | 1.1518 |

These are validation results, not final test-set results.

The saved models and experiment results are available in `Phase_3_Ablation_Experiments/`.

### Phase 4 – Transfer Learning

Notebook: `notebooks/04_transfer_learning.ipynb`

This phase uses MobileNetV2 pretrained on ImageNet to investigate transfer learning for facial expression recognition.

The input images are adapted from 48 × 48 grayscale images to 96 × 96 three-channel images.

The classification head consists of:

- Global Average Pooling.
- Dense layer with 256 neurons and ReLU activation.
- Dropout with a rate of 0.5.
- Dense output layer with 7 neurons and Softmax activation.

The fine-tuning strategy:

- Uses pretrained MobileNetV2 weights.
- Unfreezes the last 20 base-model layers.
- Keeps earlier base-model layers frozen.
- Keeps Batch Normalization layers frozen.
- Uses balanced class weights.
- Uses Adam optimizer with a learning rate of 0.0001.
- Uses Early Stopping and ReduceLROnPlateau callbacks.

Recorded validation results:

| Metric | Result |
|---|---:|
| Validation Accuracy | 56.32% |
| Macro Precision | 0.5315 |
| Macro Recall | 0.5365 |
| Macro F1 Score | 0.5286 |

The saved fine-tuned model is:

`Phase_4_Transfer_Learning/models/fer2013_mobilenet_finetuned_best.keras`

The classification report and confusion matrix are stored in the Phase 4 results folder.

### Phase 5 – Evaluation and Explainability

Notebook: `notebooks/05_evaluation_explainability.ipynb`

This phase is dedicated to evaluating the trained models on unseen test data and interpreting their predictions.

The evaluation and explainability workflow includes:

- Test-set evaluation.
- Accuracy, precision, recall, and F1-score.
- Classification reports.
- Confusion matrix analysis.
- Identification of commonly confused emotion classes.
- Grad-CAM or saliency visualizations.
- First-layer filter and intermediate feature-map visualization.
- Interpretation of model strengths and limitations.

The objective is not only to measure model performance but also to understand how the models make predictions.

See the notebook for the implemented evaluation and visualization methods and their generated outputs.

---

## Model Comparison

The project studies two main approaches:

| Aspect | Custom CNN | Transfer Learning |
|---|---|---|
| Architecture | Built from scratch | MobileNetV2 |
| Pretrained weights | No | ImageNet |
| Input | 48 × 48 × 1 | 96 × 96 × 3 |
| Training approach | CNN training and validation | Fine-tuning and validation |
| Main purpose | Learn features directly from FER-2013 | Adapt pretrained visual features to FER-2013 |

The validation results recorded for the combined CNN and MobileNetV2 are:

| Model | Validation Accuracy | Macro F1 |
|---|---:|---:|
| Final Combined CNN | 56.79% | 0.5429 |
| MobileNetV2 Fine-Tuning | 56.32% | 0.5286 |

**Important:** These are validation metrics from the recorded experiments. They should not be presented as final test performance. Use Phase 5 results for the final test-set comparison.

---

## Repository Structure

```text
Facial-Expression-Recognition-CNN/
│
├── README.md
├── .gitignore
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_ablation_experiments.ipynb
│   ├── 04_transfer_learning.ipynb
│   └── 05_evaluation_explainability.ipynb
│
├── Phase_3_Ablation_Experiments/
│   ├── README.md
│   ├── models/
│   └── results/
│       ├── classification_report.json
│       ├── confusion_matrix.npy
│       ├── evaluation_results.json
│       ├── phase3_depth_results.csv
│       └── training_history.json
│
├── Phase_4_Transfer_Learning/
│   ├── README.md
│   ├── models/
│   │   └── fer2013_mobilenet_finetuned_best.keras
│   └── results/
│       ├── Fine-tuning_Classification_Report.txt
│       └── Fine-tuning_confusion_matrix.jpeg
│
└── report/
    └── final_report.md
```

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow and Keras
- **Numerical Computing:** NumPy
- **Data Processing:** pandas
- **Machine Learning:** scikit-learn, imbalanced-learn
- **Visualization:** Matplotlib and Seaborn
- **Development Environment:** Google Colab / Jupyter Notebook
- **Model Architecture:** Custom CNN and MobileNetV2

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/shrutikushwaha2110-ux/Facial-Expression-Recognition-CNN.git

cd Facial-Expression-Recognition-CNN
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

The repository currently does not include a root-level `requirements.txt`.

Install the libraries used by the notebooks:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow pillow jupyter
```

### 4. Download the dataset

Download FER-2013 from:

[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Keep the raw dataset outside the Git repository.

For the later training notebooks, obtain the preprocessed data archive generated by the EDA/preprocessing workflow and update the notebook file paths to match your environment.

### 5. Run the notebooks

Open the notebooks in Google Colab or Jupyter Notebook.

Recommended order:

1. `01_eda.ipynb`
2. `02_baseline_cnn.ipynb`
3. `03_ablation_experiments.ipynb`
4. `04_transfer_learning.ipynb`
5. `05_evaluation_explainability.ipynb`

Run the relevant setup and data-loading cells before training or evaluation. Some notebooks use Google Colab `/content/` paths and require those paths to be adjusted when running locally.

---

## Results and Experiment Artifacts

The repository contains saved model checkpoints and experiment outputs.

- CNN experiment models.
- Final combined CNN model.
- MobileNetV2 fine-tuned model.
- Classification reports.
- Confusion matrix artifacts.
- Training history and experiment metrics.

The final evaluation notebook should be used to obtain the final test-set results. Only verified test metrics should be added to the final results section.

## Limitations and Ethical Considerations

- FER-2013 images have low resolution.
- Emotion classes are imbalanced.
- Some images have ambiguous or noisy labels.
- Duplicate images and conflicting labels exist in the original dataset.
- The original dataset contains pre-existing train-test duplicate overlap.
- Model performance may vary across image quality, lighting, facial appearance, and demographic groups.
- A predicted facial expression is not definitive evidence of a person's actual emotional state.

Facial expression recognition should be used responsibly, with appropriate attention to consent, privacy, dataset bias, and limitations.

## Team Contributions

The project was completed collaboratively, with each team member responsible for a specific phase of the workflow.

| Team Member | Role | Responsibilities and Contributions |
|---|---|---|
| **Ashmita** | Evaluation and Explainability | Handled Phase 5: model evaluation, performance metrics, classification reports, confusion matrix analysis, and explainability work. |
| **Revathi** | Transfer Learning | Handled Phase 4: MobileNetV2 transfer learning, classification-head development, fine-tuning, model training, and validation. |
| **Meghna K** | EDA and Preprocessing | Handled Phase 1: exploratory data analysis, dataset inspection, data cleaning, duplicate analysis, preprocessing, and preparation of training and validation data. |
| **Shruti** | Team Leader – CNN and Ablation Experiments | Handled Phase 2 and Phase 3: custom CNN development, baseline training, architecture experiments, ablation studies, and comparison of CNN configurations. Also coordinated the team, prepared the final project report and documentation, and worked on overall project integration and submission. |

## Project Report

The detailed project report is available here:

[`report/final_report.md`](report/final_report.md)

It covers the project background, problem statement, dataset, preprocessing, methodology, experiments, evaluation, explainability, limitations, conclusion, and team contributions.

## Acknowledgements

- FER-2013 dataset contributors.
- Atria University.
- Course instructors and project mentors.

---

**Project:** Reading Faces – Facial Expression Recognition Using CNNs  
**Institution:** Atria University  
**Group:** 3
