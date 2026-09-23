# 🦺 PPE Classification — Construction Site Safety Monitoring

A deep learning pipeline that classifies personal protective equipment (PPE) in construction-site images into 11 classes — including specific *non-compliance* classes like `no_helmet` and `no_boots` — using transfer learning with MobileNetV2.

## 📌 Overview

Manual PPE compliance audits on construction sites are sporadic and don't scale. This project builds an automated, edge-deployable classifier that can flag missing safety gear (helmets, gloves, boots, goggles, vests) from a single cropped image, using the [Ultralytics Construction-PPE dataset](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip) (YOLO-annotated) repurposed for classification.

**Result: 85.69% test accuracy / 85.82% weighted F1** across 11 highly imbalanced classes, with a compact 30.2 MB model suitable for edge hardware.

## 🎯 Problem

Automated PPE monitoring offers continuous, real-time hazard detection that manual audits can't match — but real-world datasets bring three engineering challenges this project solves directly:

1. **Label–image path mismatch** — YOLO exports keep images and labels in parallel `images/` / `labels/` folders rather than side-by-side, so naive path-matching produced zero crops. Fixed with split-aware label-to-image mapping.
2. **Train/test leakage** — re-splitting crops randomly lets near-duplicate frames from the same source photo leak across splits, inflating accuracy. Fixed by preserving the dataset's original train/val/test split.
3. **Severe class imbalance** — a 19.5:1 ratio between the largest class (`Person`, 2,245 images) and smallest (`no_boots`, 115 images). Mitigated with smoothed inverse-frequency resampling (α = 0.5) via `tf.data.sample_from_datasets`.

## 🔍 Approach

1. **Data prep** — parsed YOLO `.txt` annotations, cropped each bounding box, and organized crops into 11 class folders while preserving the original train/val/test split
2. **EDA** — visualized class distribution (highly imbalanced) and sample crops per class
3. **Balanced sampling** — built one `tf.data` pipeline per class and mixed them with inverse-sqrt class weighting so rare classes aren't drowned out without being over-duplicated
4. **Augmentation** — random flip, rotation, zoom, contrast, translation, and brightness
5. **Model** — MobileNetV2 (ImageNet-pretrained) as a frozen feature extractor, with a custom classification head
6. **Two-phase training**:
   - Phase 1: train the head only (frozen backbone), 15 epochs
   - Phase 2: unfreeze the last 80 layers and fine-tune at a low learning rate, 20 epochs
   - Both phases use early stopping and LR reduction on plateau
7. **Evaluation** — held-out test set (never seen during training or early-stopping), scored with accuracy, precision/recall/F1 per class, and a confusion matrix
8. **Demo app** — a Gradio interface for interactively classifying an uploaded image

## 📊 Dataset

| | |
|---|---|
| Source | Ultralytics Construction-PPE (YOLO format) |
| Classes | 11: `Person`, `helmet`, `no_helmet`, `gloves`, `no_gloves`, `boots`, `no_boots`, `goggles`, `no_goggle`, `vest`, `none` |
| Total crops | 11,521 |
| Train / Val / Test | 9,098 / 1,172 / 1,251 |
| Class imbalance | 19.5:1 (`Person`: 2,245 vs. `no_boots`: 115) |

## 🧠 Model & Results

**Architecture:** MobileNetV2 backbone (2.59M trainable params, ~30.2 MB) + GlobalAveragePooling → Dropout → Dense head

| Metric | Score |
|---|---|
| Test Accuracy | **85.69%** |
| Weighted F1 | **85.82%** |
| Macro F1 | 77.81% |

**Per-class highlights (safety-critical):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| helmet | 96.22% | 92.71% | 94.43% |
| goggles | 96.08% | 94.23% | 95.15% |
| vest | 90.45% | 90.45% | 90.45% |
| no_helmet | 56.36% | 77.50% | 65.26% |
| no_boots | 47.83% | 47.83% | 47.83% |

Performance on the *present* equipment classes (helmet, vest, goggles, boots, gloves) is strong. The minority "missing PPE" classes (`no_boots`, `no_helmet`, `no_goggle`) are harder — partly due to limited samples and partly because a `none` crop (no PPE visible) can look visually identical to a `no_X` crop without the full-body context.

## 🛠️ Tech Stack

- **Deep learning** — TensorFlow / Keras, MobileNetV2 transfer learning
- **Data** — YOLO annotation parsing, PIL, NumPy, pandas
- **Evaluation** — scikit-learn (classification report, confusion matrix)
- **Visualization** — Matplotlib, Seaborn
- **Demo** — Gradio

## 📁 Repository Structure

```
├── PPE_Classification.ipynb        # Full pipeline: data prep → training → evaluation → Gradio demo
├── PPE_Classification_Paper.docx   # Written report: methodology, literature survey, results
└── README.md
```

## 🚀 Running It

```bash
pip install tensorflow pillow pyyaml scikit-learn seaborn pandas matplotlib gradio
jupyter notebook PPE_Classification.ipynb
```

The notebook downloads the dataset automatically, builds the classification crops, trains the model, and launches a Gradio demo for live inference at the end.

## 🔮 Possible Extensions

- Merge or reconsider the ambiguous `none` class, which drives much of the confusion with `no_X` classes
- Collect more samples for minority violation classes (`no_boots`, `no_goggle`)
- Move from per-crop classification to full-frame multi-person detection (YOLO) for deployable real-time monitoring
- Quantize the model (e.g., TFLite) for actual edge-device deployment

## 📄 Paper

A full written report — including a literature survey of 31 papers (2018–2026) on PPE detection, methodology, and detailed results — is included as `PPE_Classification_Paper.docx`.

## 👤 Author

**Akshit Gajera** — MSc Data Science, Dayananda Sagar University, Bangalore
