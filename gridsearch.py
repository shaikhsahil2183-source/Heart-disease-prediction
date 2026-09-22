import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("heart_disease.csv", sep="\t")

# Fill missing values
df = df.fillna(df.mode().iloc[0])

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)

# SVM
svm = SVC()

# Parameters
param_grid = {
    "C": [1, 10],
    "kernel": ["rbf"]
}

# GridSearch
grid_search = GridSearchCV(
    svm,
    param_grid,
    cv=2,
    scoring="accuracy",
    n_jobs=2
)

print("GridSearch started...")

grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

import joblib

best_model = grid_search.best_estimator_

joblib.dump(best_model, "best_svm_model.pkl")

print("Best SVM model saved successfully!")

# Load the saved model
loaded_model = joblib.load("best_svm_model.pkl")

print("SVM model loaded successfully!")

# Predict using loaded model
prediction = loaded_model.predict(X.iloc[[0]])

print("Prediction:", prediction[0])


from sklearn.metrics import accuracy_score

y_pred = loaded_model.predict(X)

accuracy = accuracy_score(y, y_pred)

print("SVM Accuracy:", accuracy)
print("SVM Accuracy %:", accuracy * 100)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred)

print("SVM Confusion Matrix:")
print(cm)