
# Reading Faces: Facial Expression Recognition Using CNNs

A group project exploring facial-expression classification with convolutional neural networks (CNNs) using the FER-2013 dataset. This repository currently documents **Phase 1 (EDA and preprocessing), Phase 2 (custom CNN baseline), and Phase 3 (CNN ablation experiments)**.

> **Scope note:** Transfer learning is outside the current completed scope and is not claimed as implemented here.

## Project status

| Phase | Work | Status |
|---|---|---|
| 1 | Exploratory data analysis and preprocessing | Completed |
| 2 | Custom CNN baseline | Completed |
| 3 | CNN ablation experiments | Completed |
| 4 | Transfer learning | Not included in this version |

## Dataset

- **Dataset:** FER-2013
- **Image dimensions:** 48 × 48 pixels
- **Image format:** Grayscale
- **Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Classification task:** Seven-class facial-expression classification

The dataset itself is not included in this repository. Obtain FER-2013 from the source specified by your course/project instructions and update the dataset path in the notebooks to match your local or Colab folder structure.

## Repository structure

```text
Facial-Expression-Recognition-CNN/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_cnn.ipynb
├── Phase_3_Ablation_Experiments/
│   ├── README.md
│   ├── models/
│   │   └── saved .keras experiment models
│   └── results/
│       ├── evaluation_results.json
│       ├── classification_report.json
│       ├── confusion_matrix.npy
│       ├── phase3_depth_results.csv
│       ├── training_history.json
│       └── final_cnn_model.keras
└── report/
    └── final_report.md
```

**Notebook note:** The current `02_baseline_cnn.ipynb` contains the baseline work and the subsequent ablation experiment cells. The separate Phase 3 folder contains the experiment documentation, saved models and results. A dedicated Phase 3 notebook can be split out as a follow-up organisation step; no experiment results should be changed while doing so.

## Work completed

### Phase 1 — EDA and preprocessing

The EDA notebook explores class distribution, sample images, average faces, image integrity, blank/dark images, duplicate images and label inconsistencies. It then cleans the training data, creates a group-aware train/validation split, normalizes grayscale images and encodes labels.

### Phase 2 — Custom CNN baseline

A CNN is built from scratch for 48 × 48 × 1 grayscale images and seven output classes. The baseline uses convolutional blocks, batch normalization, max pooling, a dense layer, dropout and a softmax output. The baseline validation results recorded in the Phase 3 experiment documentation are **50.74% accuracy** and **0.3933 macro F1**.

### Phase 3 — Ablation experiments

The experiments compare:

- CNN depth: 2, 3 and 4 convolutional blocks
- Batch normalization: enabled vs disabled
- Dropout: 0.3, 0.5 and 0.7
- Optimizer configurations: Adam vs SGD with momentum
- Data augmentation: enabled vs disabled
- Class balancing: class weights vs random oversampling

The Phase 3 README contains the recorded experiment-level results and the final combined-model validation results.

## Recorded final combined-model validation results

| Metric | Result |
|---|---:|
| Validation accuracy | 56.79% |
| Macro F1 score | 0.5429 |
| Best validation loss | 1.1518 |

These are **validation results**, not test-set results. The test set has not been used for final evaluation in the current completed scope. Individual ablation runs may vary due to random initialization and training variation; the experiment comparisons should be interpreted accordingly.

## Running the notebooks

1. Open the notebooks in Google Colab or Jupyter.
2. Obtain FER-2013 separately and update the dataset path in the notebook setup cells.
3. Run `01_eda.ipynb` to inspect and preprocess the dataset.
4. Follow the saved preprocessing workflow and required array variables before running `02_baseline_cnn.ipynb`.
5. The notebook includes the baseline and Phase 3 experiment workflow. Saved Phase 3 models and result files are in `Phase_3_Ablation_Experiments/`.

The notebooks use Python with TensorFlow/Keras and supporting data-science libraries. Install any missing imports in the notebook environment before execution.

## Important limitations

- FER-2013 contains class imbalance, noisy/ambiguous labels and low-resolution images.
- The EDA identified 530 unique image hashes overlapping between the original training and test partitions; this should be considered when interpreting test-set generalization.
- The ablation results are based on individual runs and may be affected by random initialization and training variation.
- Optimizer comparisons use different learning-rate configurations, so they compare configurations rather than isolating optimizer choice alone.
- The final combined model's reported 56.79% is validation accuracy. No final test accuracy is claimed here.
- Facial-expression labels are dataset categories and should not be treated as definitive measurements of a person's internal emotional state.

## Technologies

Python · TensorFlow · Keras · NumPy · pandas · scikit-learn · imbalanced-learn · Matplotlib · Seaborn · Google Colab

## Report

See [`report/final_report.md`](report/final_report.md) for the project report. Ensure its results, figures and discussion are finalised before submission.
