# Heart Disease Prediction using Machine Learning

A machine learning classification project that predicts the presence of heart disease from clinical patient data.

This project covers the complete ML workflow — **data preprocessing, feature scaling, model training, cross-validation, hyperparameter tuning, evaluation, and model persistence.**

## Project Overview

The goal of this project is to classify patients into two categories:

* `0` — No Heart Disease
* `1` — Heart Disease

The project uses **Logistic Regression** and **Support Vector Machine (SVM)** and compares their performance on the test dataset.

## Tech Stack

* **Python**
* **Pandas** — Data preprocessing
* **Scikit-learn** — Machine learning
* **Matplotlib** — Visualization
* **Joblib** — Model persistence

## ML Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Cross-Validation
   ↓
GridSearchCV
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Prediction
```

## Models & Results

| Model               | Test Accuracy |
| ------------------- | ------------: |
| Logistic Regression |    **83.78%** |
| SVM (RBF)           |    **85.41%** |

### Best SVM Configuration

```text
C = 10
Kernel = RBF
Cross-Validation Accuracy = 85.46%
Test Accuracy = 85.41%
```

## Model Evaluation

### SVM Confusion Matrix

```text
                 Predicted
                No      Yes

Actual No       64      17
Actual Yes      10      94
```

The complete visualization is available in:

`svm_confusion_matrix.png`

## Model Persistence

The trained SVM model and preprocessing scaler are saved using Joblib:

```text
best_svm_model.pkl
scaler.pkl
```

The saved files can be loaded later to make predictions on new data using the same preprocessing pipeline.

## Project Structure

```text
Heart-Disease-Prediction-ML/
│
├── heart disease.py
├── heart_disease.csv
├── best_svm_model.pkl
├── scaler.pkl
├── svm_confusion_matrix.png
└── README.md
```

## Key Learning

This project helped me gain hands-on experience with:

* Data cleaning and preprocessing
* Handling missing values
* Categorical encoding
* Feature scaling
* Binary classification
* Logistic Regression
* Support Vector Machines
* Cross-validation
* GridSearchCV
* Confusion Matrix & classification metrics
* Saving and loading ML models

## Future Improvements

* Experiment with additional classification algorithms
* Perform feature selection and optimization
* Build a simple web interface for predictions
* Deploy the model as a small ML application

## Disclaimer

This project is created for **educational and demonstration purposes only**. It should not be used for medical diagnosis or clinical decision-making.

## Author

**Sahil Shaikh**
B.Tech — Artificial Intelligence & Machine Learning
