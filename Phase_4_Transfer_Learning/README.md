# Phase 4: Transfer Learning (MobileNetV2)

This phase explores transfer learning using **MobileNetV2** pre-trained on ImageNet, adapted for facial expression recognition on the FER-2013 dataset.

## ⚙️ Process & Methodology Followed

1. **Data Preprocessing & Resizing (`prepare_mb_data`)**
   * Scaled and resized native $48 \times 48$ grayscale arrays to $96 \times 96$ pixels using batched processing forced on the CPU to prevent GPU Out-of-Memory errors.
   * Converted single-channel inputs into 3-channel RGB tensors to match MobileNetV2 architecture requirements.
   * Normalized pixel ranges from $[0, 1]$ to $[-1, 1]$.

2. **Class Weight Optimization**
   * Calculated automated class weights using `scikit-learn` to combat category imbalances across emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

3. **Stage 1: Feature Extraction**
   * Loaded **MobileNetV2** pre-trained on ImageNet with weights frozen (`base_model.trainable = False`).
   * Attached a custom classification head (`GlobalAveragePooling2D`, `Dense(256)`, `Dropout(0.5)`, and `Dense(7, softmax)`).
   * Trained the classification head for 10 epochs using the Adam optimizer ($\text{lr} = 0.001$).

4. **Stage 2: Fine-Tuning**
   * Unfroze the top 20 layers of the MobileNetV2 base model while keeping Batch Normalization layers locked to prevent overfitting.
   * Recompiled with a reduced learning rate ($\text{lr} = 0.0001$).
   * Applied training callbacks: **EarlyStopping** (`patience=5`, restoring best weights) and **ReduceLROnPlateau** (`factor=0.5`, `patience=2`).

## 📊 Model Comparison Table

| Model Architecture | Test Loss | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Phase 3)** | ~1.14 | 56.53% | ~0.53 | ~0.55 | 0.5429 |
| **MobileNetV2 (Feature Extraction)** | ~1.44 | 45.02% | ~0.40 | ~0.46 | 0.39 |
| **MobileNetV2 (Fine-Tuning)** | **1.2** | **54.87%** | **0.50** | **0.52** | **0.51** |


## 📉 Training Dynamics, Overfitting & Checkpoints
- **Overfitting Control**: Early stopping with a patience of 5 epochs monitored validation loss.
- **Selected Checkpoint**: `fer2013_mobilenet_finetuned_best.keras` saved at the optimal validation epoch.

### Key Takeaways & Limitations
- **Comparative Performance:** The Fine-Tuned MobileNetV2 achieves competitive test metrics, closely following the Custom CNN baseline. 
- **Loss & Efficiency:** The Custom CNN demonstrates a slightly lower Test Loss, while the Fine-Tuned MobileNetV2 successfully adapts structural representations after unfreezing top layers.
- **Dataset Limitation Notice:** When interpreting these test metrics, note that the original FER-2013 benchmark split contains duplicate-image overlaps between training, validation, and test subsets. This leakage introduces a known constraint that can inflate absolute performance metrics during evaluation.



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
