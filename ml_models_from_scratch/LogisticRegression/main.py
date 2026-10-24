from ml_models_from_scratch.load_data import load_data
from .eda import eda
from .LogisticRegressionFromScratch import LogisticRegression
from .evaluate import evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

name, X, y = load_data(id=267)
print(f"Exploratory data analysis for {name} dataset")
eda(X, y)

print("=" * 50)
print("Logistic Regression from scratch")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test)

print(f"Weights: {model.W}")
print(f"Bias: {model.b}")

print("=" * 50)
print("Evaluation Metrics")
for key, value in evaluate(y_test, y_pred).items():
    print(f"{key}: {value:.4f}")

