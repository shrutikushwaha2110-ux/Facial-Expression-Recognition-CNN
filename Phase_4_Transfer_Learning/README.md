# Phase 4 Transfer Learning Models

# Phase 4: Transfer Learning (MobileNetV2)

This phase explores the application of transfer learning using the **MobileNetV2** architecture for facial expression recognition on the FER-2013 dataset. By leveraging a pre-trained feature extractor, we aim to improve generalization, handle class imbalances more effectively, and boost overall classification performance compared to baseline custom CNNs.

---

## 🚀 Key Architectural & Pipeline Decisions

* **Base Model**: MobileNetV2 pre-trained on ImageNet.
* **Preprocessing Correction**: Input images scaled strictly to the $[-1, 1]$ range to match MobileNetV2's expected pre-training distribution.
* **Overfitting Mitigation**: 
  * Froze early feature extraction layers while fine-tuning later blocks.
  * Explicitly **froze Batch Normalization layers** to prevent destructive weight updates during fine-tuning on a smaller target dataset.
* **Regularization & Optimization**: Added dense classification heads with appropriate dropout and optimized learning rates.

---

## 📊 Performance & Results

* **Test Accuracy**: ~55.53%
* **Macro F1-Score**: 0.4907

### Evaluation Artifacts
* **Confusion Matrix & Classification Report**: Generated and stored in the `results/` directory to track per-class precision, recall, and confusion patterns across the 7 facial expression categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

---

## 📂 Directory Structure

```text
Phase_4_Transfer_Learning/
├── models/
│   └── fer2013_mobilenet_finetuned_best.keras   # Saved fine-tuned model weights
├── results/
│   ├── confusion_matrix.png                     # Visual confusion matrix plot
│   └── classification_report.txt                # Detailed precision/recall metrics
└── README.md                                    # Phase documentation
