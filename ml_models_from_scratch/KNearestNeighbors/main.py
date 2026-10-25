from .eda import eda
from ml_models_from_scratch.load_data import load_data
from .KNNFromScratch import KNearestNeighbors
from ..Evaluation.evaluate_multiclass_classifier import evaluate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

name, X, y = load_data(id = 53) 

print(f"Exploratory Data Analysis for {name} dataset")
eda(X, y)
print("=" * 50)

print()
print("=" * 50)
print("KNN From Scratch")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, 
    random_state=42,
    stratify=y

)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNearestNeighbors(k = 3)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

y_test = np.asarray(y_test)

for key, value in evaluate(y_test, y_pred).items():
    print(f"{key}: {value}")