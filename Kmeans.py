import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def initialize_centroids(self, X):
        np.random.shuffle(X)
        return X[:self.n_clusters]

    def assign_clusters(self, X):
        clusters = []
        for i in range(len(X)):
            distances = np.linalg.norm(X[i] - self.centroids, axis=1)
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def update_centroids(self, X, clusters):
        for i in range(self.n_clusters):
            self.centroids[i] = np.mean(X[clusters == i], axis=0)

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_iter):
            clusters = self.assign_clusters(X)
            old_centroids = np.copy(self.centroids)
            self.update_centroids(X, clusters)
            if np.allclose(old_centroids, self.centroids):
                break

    def predict(self, X):
        clusters = self.assign_clusters(X)
        return clusters
class KmeansTest:
    def __init__(self,n_cluster,max_iter=100):
        self.n_cluster = n_cluster;
        self.max_iter = max_iter;
        self.centroids = None
    def initCentroids(self,X):
        np.random.shuffle(X)
        return X[:self.n_cluster]
if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    print(kmeans.predict(X))
