# Neural Network Assignment (PyTorch)

## Overview

This project implements Artificial Neural Networks (ANN) using PyTorch for:

* Regression Task
* Classification Task

---

## Datasets

* California Housing Dataset (Regression)
* Breast Cancer Wisconsin Dataset (Classification)

---

## Preprocessing

* Feature scaling using StandardScaler
* Train-test split (80-20)

---

## Model Architecture

* Input Layer
* Hidden Layer 1 (64 neurons, ReLU)
* Hidden Layer 2 (32 neurons, ReLU)
* Output Layer

---

## Training Details

* Optimizer: Adam
* Regression Loss: MSELoss
* Classification Loss: BCEWithLogitsLoss

---

## Hyperparameter Experiment

| Learning Rate | Regression Loss | Classification Accuracy |
| ------------- | --------------- | ----------------------- |
| 0.01          | ~0.39           | ~93%                    |
| 0.001         | ~0.60           | ~96%                    |

---

## Results

* Regression model successfully minimized loss over epochs
* Classification model achieved high accuracy (~96%)

---

## Graphs

### Regression

![Regression](regression.png)

### Classification

![Classification](classification.png)

---

## Conclusion

Lower learning rate improves stability but slows convergence.
ANN performs well for both regression and classification tasks.

---

## How to Run

pip install torch pandas scikit-learn matplotlib

python regression.py
python classification.py
