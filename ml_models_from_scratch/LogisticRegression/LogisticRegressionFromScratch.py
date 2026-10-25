import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0.0
        self.loss_history = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def linear_model(self, X):
        return X @ self.W + self.b

    def calculate_loss(self, y, y_pred):
        e = 1e-10
        y_pred = np.clip(y_pred, e, 1-e)

        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1) 
        
        n_samples, n_features = X.shape

        self.W = np.zeros((n_features, 1))
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            z = self.linear_model(X)
            predictions = self.sigmoid(z)
            error = predictions - y
            

            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.W -= self.lr * dw
            self.b -= self.lr * db

            loss = self.calculate_loss(y, predictions)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f}")
                self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        z = self.linear_model(X)
        probability = self.sigmoid(z)
        return probability

    def predict(self, X, threshold = 0.5):
        probability = self.predict_proba(X)
        return (probability >= threshold).astype(int)
    
    
