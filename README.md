# Heart Disease Prediction using Machine Learning

This project uses machine learning to predict the possibility of heart disease based on medical data.

## Project Overview

The dataset was cleaned and prepared for machine learning. An SVM (Support Vector Machine) model was trained to classify the data and predict whether a person is likely to have heart disease.

## What I Did

- Loaded and explored the dataset using Pandas
- Cleaned and prepared the data
- Split the data into training and testing sets
- Trained an SVM classification model
- Used GridSearchCV to find better model parameters
- Evaluated the model using accuracy and confusion matrix
- Saved the trained model using Joblib
- Tested the saved model with new data

## Model Performance

- Model: Support Vector Machine (SVM)
- Accuracy: **79.6%**

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Project Files

- `heart_disease.py` – Main machine learning code
- `gridsearch.py` – Hyperparameter tuning using GridSearchCV
- `heart_disease.csv` – Dataset
- `best_svm_model.pkl` – Saved trained model

## How to Run

1. Clone the repository
2. Install the required Python libraries
3. Run the Python file

```bash
pip install -r requirements.txt
python heart_disease.py
