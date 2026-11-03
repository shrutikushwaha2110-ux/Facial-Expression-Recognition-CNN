# Phase 4: Transfer Learning (MobileNetV2)

This phase explores transfer learning using **MobileNetV2** pre-trained on ImageNet, adapted for facial expression recognition on the FER-2013 dataset.

## 🚀 Key Architectural & Pipeline Decisions
* **Preprocessing**: Input images scaled to $[-1, 1]$ to match MobileNetV2's distribution requirements.
* **Freezing Strategy**: Frozen early feature extraction blocks and explicitly frozen **Batch Normalization layers** to prevent destructive updates.
* **Class Imbalance**: Mitigated using balanced class weighting during training.

---

## 📊 Model Comparison Table

| Model Architecture | Test Loss | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Phase 3)** | ~1.28 | - - | ~0.55 | ~0.53 | 0.5429 (Val) |
| **MobileNetV2 (Feature Extraction)** | ~1.35 | ~51.20% | ~0.48 | ~0.45 | 0.4620 |
| **MobileNetV2 (Fine-Tuning)** | **1.2543** | **55.53%** | **0.53** | **0.46** | **0.4907** |

---

## 📉 Training Dynamics, Overfitting & Checkpoints
* **Overfitting Control**: Early stopping with a patience of 5 epochs monitored validation loss.
* **Selected Checkpoint**: `fer2013_mobilenet_finetuned_best.keras` saved at the optimal validation epoch.

---

## 📂 Directory Structure
```text
Phase_4_Transfer_Learning/
├── models/
│   └── fer2013_mobilenet_finetuned_best.keras
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
└── README.md
