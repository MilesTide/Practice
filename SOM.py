import numpy as np


class SOM:
    def __init__(self, map_size, n_features, learning_rate=0.5, radius=2.5, decay_rate=0.9):
        self.map_size = map_size
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.radius = radius
        self.decay_rate = decay_rate
        self.map = np.random.rand(map_size[0], map_size[1], n_features)

    def get_winner(self, sample):
        distances = np.linalg.norm(self.map - sample, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def get_neighborhood(self, center, radius):
        distances = np.linalg.norm(np.indices(self.map_size) - np.array(center).reshape(-1, 1, 1), axis=0)
        return np.where(distances <= radius)

    def fit(self, X, n_iterations):
        for iteration in range(n_iterations):
            for sample in X:
                winner = self.get_winner(sample)
                neighborhood = self.get_neighborhood(winner, self.radius)
                self.map[neighborhood] += self.learning_rate * (sample - self.map[neighborhood])
            self.radius *= self.decay_rate
            self.learning_rate *= self.decay_rate

    def predict(self, X):
        labels = np.zeros(len(X))
        for i, sample in enumerate(X):
            winner = self.get_winner(sample)
            labels[i] = winner[0] * self.map_size[1] + winner[1]
        return labels
if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    som = SOM(map_size=(10, 10), n_features=2)
    som.fit(X, n_iterations=1000)

    print(som.predict(X))