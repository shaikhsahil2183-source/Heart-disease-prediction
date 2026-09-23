# 🫀 Heart Disease Prediction using Machine Learning

A machine learning classification project that predicts the presence of heart disease based on patient-related clinical features.

The project demonstrates a complete ML workflow — from data preprocessing and feature scaling to model training, hyperparameter tuning, evaluation, model saving, and prediction.

---

## 📌 Project Overview

The main objective of this project is to build a machine learning model that can classify whether a patient is likely to have heart disease.

Two classification approaches were explored:

* Logistic Regression
* Support Vector Machine (SVM)

After evaluation and hyperparameter tuning, the SVM model achieved **85.41% test accuracy**.

> **Note:** This project is for educational and machine-learning practice purposes and is not intended for medical diagnosis.

---

## 🛠️ Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Pandas       | Data loading and preprocessing |
| Scikit-learn | Machine learning               |
| Matplotlib   | Data visualization             |
| Joblib       | Model saving and loading       |

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Handle Missing Values
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Cross-Validation
   ↓
SVM + GridSearchCV
   ↓
Model Evaluation
   ↓
Save Model + Scaler
   ↓
Load Model & Prediction
```

---

## 📊 Dataset

* Total samples: **921**
* Features: **22**
* Problem type: **Binary Classification**

### Target

The original target variable was converted into two classes:

* `0` → No Heart Disease
* `1` → Heart Disease

---

## 🤖 Models Used

### 1. Logistic Regression

Test Accuracy:

**83.78%**

Confusion Matrix:

```text
[[64 17]
 [13 91]]
```

---

### 2. Support Vector Machine

SVM was tuned using `GridSearchCV`.

**Best Parameters:**

```text
C = 10
Kernel = RBF
```

**Test Accuracy:**

**85.41%**

**Cross-Validation Accuracy:**

**85.46%**

### SVM Confusion Matrix

```text
[[64 17]
 [10 94]]
```

|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |           64 |            17 |
| Actual Yes |           10 |            94 |

The visualization is available in:

`svm_confusion_matrix.png`

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Cross-validation

For SVM, `GridSearchCV` was used to search for suitable hyperparameters.

---

## 💾 Model Saving

The trained SVM model is saved using Joblib:

```text
best_svm_model.pkl
```

The `StandardScaler` used during training is also saved:

```text
scaler.pkl
```

Saving both is important because new input data must go through the **same scaling process** before prediction.

---

## 🔮 Prediction

The saved model and scaler can be loaded and used to make predictions on new data.

Example output:

```text
SVM model loaded successfully!
Prediction: 0
```

Where:

```text
0 → No Heart Disease
1 → Heart Disease
```

---

## 📁 Project Structure

```text
Heart-disease-prediction/
│
├── heart disease.py
├── heart_disease.csv
├── best_svm_model.pkl
├── scaler.pkl
├── svm_confusion_matrix.png
└── README.md
```

---

## 🚀 Key Learning Outcomes

Through this project, I practiced:

* Data preprocessing
* Handling missing values
* Categorical data encoding
* Feature scaling
* Train-test splitting
* Logistic Regression
* Support Vector Machine
* Cross-validation
* GridSearchCV
* Confusion Matrix
* Classification metrics
* Model persistence using Joblib
* Loading a trained model for prediction

---

## 🎯 Future Improvements

Possible improvements for this project include:

* Testing additional classification algorithms
* Improving the preprocessing pipeline
* Using a larger and more diverse dataset
* Creating a simple web interface for prediction
* Deploying the trained model as an application

---

## 👨‍💻 Author

**Sahil Shaikh**

B.Tech — Artificial Intelligence & Machine Learning
