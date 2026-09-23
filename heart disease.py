import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load Dataset
df = pd.read_csv("heart_disease.csv", sep="\t")

# 2. Handle Missing Values
df = df.fillna(df.mode().iloc[0])

# 3. Convert Categorical Data
df = pd.get_dummies(df, drop_first=True)

# 4. Features and Target
X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)

print("Dataset Shape:", X.shape)


# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# 6. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 7. Logistic Regression
model = LogisticRegression(max_iter=5000, solver="liblinear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, y_pred))


# 8. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 9. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10. Cross-Validation
scores = cross_val_score(model, X, y, cv=5)

print("\nCross-Validation Scores:", scores)
print("Mean CV Accuracy:", scores.mean())


# 11. GridSearchCV - SVM with Scaling

svm = SVC()

param_grid = {
    "C": [1, 10],
    "kernel": ["rbf"]
}

grid_search = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=2
)

grid_search.fit(X_train, y_train)

print("\nBest SVM Parameters:",
      grid_search.best_params_)

print("Best SVM CV Accuracy:",
      grid_search.best_score_)

# Test Set Prediction
svm_pred = grid_search.predict(X_test)

print("SVM Test Accuracy:",
      accuracy_score(y_test, svm_pred))

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))

print("\nBest SVM Parameters:",
      grid_search.best_params_)

print("Best SVM CV Accuracy:",
      grid_search.best_score_)

# 12. Save SVM Model + Scaler

best_model = grid_search.best_estimator_

joblib.dump(best_model, "best_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSVM model saved successfully!")
print("Scaler saved successfully!")

# 13. Load Model

loaded_model = joblib.load("best_svm_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

print("SVM model loaded successfully!")


# 14. Prediction

new_data = X.iloc[[0]]

new_data_scaled = loaded_scaler.transform(new_data)

prediction = loaded_model.predict(new_data_scaled)

print("Prediction:", prediction[0])
# 15. Generate SVM Confusion Matrix

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    svm_pred,
    display_labels=["No Heart Disease", "Heart Disease"],
    values_format="d",
    ax=ax
)

ax.set_title(
    f"Heart Disease Prediction\nSVM Confusion Matrix | Accuracy: {accuracy_score(y_test, svm_pred):.2%}",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSVM Confusion Matrix image saved successfully!")

plt.show()