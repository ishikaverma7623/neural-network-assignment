# Neural Network Assignment (PyTorch)

## Overview
This project implements Artificial Neural Networks (ANN) using PyTorch for:
- Regression Task
- Classification Task

---

## Datasets Used
- California Housing Dataset (Regression)
- Breast Cancer Wisconsin Dataset (Classification)

---

## Preprocessing
- Feature scaling using StandardScaler
- Train-test split (80-20)

---

## Model Architecture
- Input Layer
- Hidden Layer 1 (64 neurons, ReLU)
- Hidden Layer 2 (32 neurons, ReLU)
- Output Layer

---

## Training Details
- Optimizer: Adam
- Regression Loss: MSELoss
- Classification Loss: BCEWithLogitsLoss

---

## Results

### Regression
- Learning Rate 0.01 → Lower Loss (~0.39)
- Learning Rate 0.001 → Higher Loss (~0.60)

### Classification
- Learning Rate 0.01 → Accuracy ~93%
- Learning Rate 0.001 → Accuracy ~96%

---

## Conclusion
Lower learning rate improves stability but may require more epochs for convergence.

---

## How to Run

pip install torch pandas scikit-learn matplotlib

python regression.py  
python classification.py
