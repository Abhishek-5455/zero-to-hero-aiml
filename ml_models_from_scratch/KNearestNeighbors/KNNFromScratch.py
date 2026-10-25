import numpy as np

class KNearestNeighbors:
    def __init__(self, k):
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("K must be a positive integer")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y).ravel()

    def _euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_one(self, x):

        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean_dist(x, x_train)
            distances.append((dist, self.y_train[i]))

        distances.sort(key = lambda item: item[0])
        k_nearest = distances[:self.k]

        labels = [label for _, label in k_nearest]
        unique_labels, count = np.unique(labels, return_counts=True)

        return unique_labels[np.argmax(count)]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return [self._predict_one(x) for x in X]


