import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0.0
        self.loss_history = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)   
        
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1))
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            y_pred = X @ self.W + self.b
            error = y_pred - y

            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.W -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 50 == 0:
                predictions = X @ self.W + self.b
                loss = np.mean((predictions - y) ** 2)
                print(f"Epoch {epoch}: Loss: {loss}")
                self.loss_history.append(loss)

        print("=" * 50)
        return self

    def predict(self, X):
        return X @ self.W + self.b