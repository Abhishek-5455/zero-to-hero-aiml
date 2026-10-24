from .eda import eda
from ml_models_from_scratch.load_data import load_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .LinearRegressionFromScratch import LinearRegression

import numpy as np

name, X, y = load_data(id=477)
print(f"Exploratory data analysis for {name} dataset")
eda(X, y)

print("=" * 50)
print("Linear Regression from scratch")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= 0.2,
    random_state=42
)

print(f"X Training set shape: {X_train.shape}")
print(f"X Testing set shape: {X_test.shape}")

print(f"y Training set shape: {y_train.shape}")
print(f"y Testing set shape: {y_test.shape}")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression(learning_rate = 0.01, epochs = 1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Weights: {model.W}")
print(f"Bias: {model.b}")

print("=" * 50)
print("Evaluation")
print("=" * 50)

print(f"Mean Squared Error: {np.mean((y_pred - y_test) ** 2):.2f}")
print(f"Mean Absolute Error : {np.mean((np.abs(y_pred - y_test))):.2f}")
