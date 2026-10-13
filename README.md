<div align="center">

# A Lightweight Attention-Enhanced CNN-SVM Hybrid Approach for Maize Leaf Disease Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Soyebsoyeb/Maize_Disease_Classification/pulls)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**A hybrid deep learning framework integrating MobileNetV2, Convolutional Block Attention Module (CBAM), and Support Vector Machine for robust maize leaf disease classification.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository presents a lightweight yet high-performance hybrid classification framework for automated maize leaf disease detection. The proposed architecture leverages a **MobileNetV2** backbone enhanced with **Convolutional Block Attention Module (CBAM)** for discriminative feature extraction, followed by a **Support Vector Machine (SVM)** classifier operating on the learned feature representations. The system is designed to operate efficiently on resource-constrained environments while achieving state-of-the-art classification accuracy across multiple disease categories.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Lightweight Backbone** | MobileNetV2 with ImageNet pre-trained weights for efficient feature extraction |
| **Attention Mechanism** | CBAM (Channel + Spatial Attention) for enhanced focus on disease-specific regions |
| **Hybrid Classification** | Dual-stage pipeline: CNN feature extraction + SVM classification |
| **Multi-Dataset Support** | Compatible with 4-class and 5-class maize leaf disease datasets |
| **Data Augmentation** | Comprehensive augmentation pipeline for improved generalization |
| **Model Persistence** | Trained models and feature extractors exported in `.h5` format |

---

## Architecture

### Core Pipeline

```
Input (224 x 224 x 3)
    |
    v
[ MobileNetV2 Backbone ]  (ImageNet Pre-trained, include_top=False)
    |
    v
[ CBAM Attention Block ]  (Channel Attention + Spatial Attention)
    |
    v
[ Global Average Pooling ]
    |
    v
[ Dense(256) -> BatchNorm -> Dropout(0.5) ]
    |
    v
[ Dense(128) -> BatchNorm -> Dropout(0.5) ]  <-- Feature Extraction Layer
    |
    v
[ Dense(4/5, Softmax) ]  <-- CNN Classifier
```

### CBAM Attention Module

The Convolutional Block Attention Module is integrated sequentially after the MobileNetV2 backbone to refine feature maps through:

1. **Channel Attention**
   - Global Average Pooling + Global Max Pooling
   - Shared MLP with reduction ratio
   - Element-wise addition and Sigmoid activation

2. **Spatial Attention**
   - Channel-wise mean and max pooling
   - Concatenation along channel dimension
   - Convolutional layer (7x7) and Sigmoid activation

---

## Datasets

### Dataset 1: Corn/Maize Leaf Disease Dataset (4-Class)

| Attribute | Value |
|-----------|-------|
| **Source** | `/kaggle/input/corn-or-maize-leaf-disease-dataset/data` |
| **Classes** | Blight, Common Rust, Gray Leaf Spot, Healthy |
| **Training** | 80% (3,348 images) |
| **Validation** | 10% |
| **Test** | 10% (423 images) |

### Dataset 2: MaizeLeafDataset (5-Class)

| Attribute | Value |
|-----------|-------|
| **Source** | `/kaggle/input/maizeleaf/MaizeLeafDataset` |
| **Classes** | Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight, Not Maize Leaf |
| **Training** | 80% (7,079 images) |
| **Validation** | 10% |
| **Test** | 10% (891 images) |

### Preprocessing Configuration

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
COLOR_MODE = 'rgb'

# Training Augmentation
rescale=1./255,
rotation_range=20,
horizontal_flip=True,
vertical_flip=True,
brightness_range=[0.8, 1.2],
zoom_range=0.2,
fill_mode='nearest'

# Validation / Test
rescale=1./255
```

---

## Methodology

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (learning_rate=0.001) |
| Loss Function | Categorical Crossentropy (label_smoothing=0.1) |
| Activation | Softmax |
| Trainable Parameters | ~3 Million |
| Callbacks | EarlyStopping (patience=5), ReduceLROnPlateau |

### Feature Extraction for SVM

The 128-dimensional feature vectors are extracted from the penultimate dense layer of the trained CNN. These embeddings serve as input to an SVM classifier with the following characteristics:

- **Feature Dimension**: 128
- **Feature Shape (Train)**: (7,079, 128)
- **Feature Shape (Test)**: (891, 128)

---

## Installation

### Prerequisites

- Python >= 3.8
- TensorFlow >= 2.8
- Keras >= 2.8
- scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn

### Setup

```bash
# Clone the repository
git clone https://github.com/Soyebsoyeb/Maize_Disease_Classification.git
cd Maize_Disease_Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.8.0
keras>=2.8.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
pillow>=8.3.0
```

---

## Usage

### Training the Model

```python
from model import build_model
from data_loader import load_data

# Load dataset
train_gen, val_gen, test_gen = load_data(dataset_path="/path/to/dataset")

# Build model with CBAM attention
model = build_model(
    input_shape=(224, 224, 3),
    num_classes=5,
    use_cbam=True,
    backbone="mobilenetv2"
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Save model
model.save("final_trained_model.h5")
```

### Feature Extraction and SVM Classification

```python
from feature_extractor import extract_features
from sklearn.svm import SVC

# Extract 128-dim features
X_train_features = extract_features(model, train_gen, layer_name="dense_128")
X_test_features = extract_features(model, test_gen, layer_name="dense_128")

# Train SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_features, y_train)

# Evaluate
accuracy = svm.score(X_test_features, y_test)
print(f"SVM Test Accuracy: {accuracy * 100:.2f}%")
```

### Inference on New Images

```python
from inference import predict_disease

prediction = predict_disease(
    image_path="path/to/leaf_image.jpg",
    model_path="final_trained_model.h5",
    class_names=["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
)
print(f"Predicted Class: {prediction}")
```

---

## Results

### Dataset 1: 4-Class Classification

#### CNN Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.16% |
| Final Loss | 0.4247 |
| Epochs | ~41 |

#### CNN Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Blight | 0.96 | 0.93 | 0.95 | 116 |
| Common Rust | 1.00 | 1.00 | 1.00 | 132 |
| Gray Leaf Spot | 0.87 | 0.93 | 0.90 | 58 |
| Healthy | 1.00 | 1.00 | 1.00 | 117 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **423** |

#### Hybrid CNN-SVM Performance

| Component | Result |
|-----------|--------|
| SVM Accuracy | 98.11% |
| Feature Dimension | 128 |
| Best Classifier | SVM on extracted features |

---

### Dataset 2: 5-Class Classification

#### CNN Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 99.66% |
| Test Accuracy | 99.00% |
| Test Loss | 0.43 |
| Epochs | ~45 |

#### CNN Classification Report

| Metric | Value |
|--------|-------|
| Weighted Precision | 0.9903 |
| Weighted Recall | 0.9899 |
| Weighted F1-Score | 0.9899 |

#### Hybrid CNN-SVM Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.44% |
| F1 / Precision / Recall | > 0.98 (all classes) |

### Comparative Summary

| Component | Specification |
|-----------|--------------|
| Model Backbone | MobileNetV2 (ImageNet pre-trained) + CBAM |
| Classification | Softmax and SVM (128-dim features) |
| Input Size | 224 x 224 RGB |
| Data Augmentation | Applied to training set |
| CBAM Integration | Channel + Spatial attention |
| Final Accuracy | ~99.4% (combined CNN + SVM) |
| Deployment Ready | Saved model (.h5) and feature extractor exported |

---

## Project Structure

```
Maize_Disease_Classification/
|
|-- data/
|   |-- corn-or-maize-leaf-disease-dataset/    # Dataset 1 (4-class)
|   |-- maizeleaf/MaizeLeafDataset/             # Dataset 2 (5-class)
|
|-- models/
|   |-- cbam.py                                 # CBAM attention module
|   |-- mobilenetv2_cbam.py                     # Full architecture
|   |-- final_trained_model.h5                  # Saved trained model
|
|-- utils/
|   |-- data_loader.py                          # Data loading and augmentation
|   |-- feature_extractor.py                    # Feature extraction pipeline
|   |-- visualizations.py                       # Confusion matrix and plots
|
|-- notebooks/
|   |-- training_pipeline.ipynb                 # End-to-end training notebook
|   |-- svm_evaluation.ipynb                    # SVM classification notebook
|
|-- results/
|   |-- confusion_matrix.png
|   |-- training_curves.png
|   |-- classification_report.txt
|
|-- README.md
|-- requirements.txt
|-- LICENSE
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{maize_disease_classification_2024,
  title={A Lightweight Attention-Enhanced CNN-SVM Hybrid Approach for Maize Leaf Disease Classification},
  author={Soyeb},
  journal={GitHub Repository},
  year={2024},
  url={https://github.com/Soyebsoyeb/Maize_Disease_Classification}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Developed with precision for agricultural AI applications.**

[Report Bug](https://github.com/Soyebsoyeb/Maize_Disease_Classification/issues) · [Request Feature](https://github.com/Soyebsoyeb/Maize_Disease_Classification/issues)

</div>
