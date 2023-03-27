import numpy as np


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.labels_ = np.zeros(len(X))
        cluster_label = 1
        for i in range(len(X)):
            if self.labels_[i] != 0:
                continue
            neighbors = self.region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                self.expand_cluster(X, i, neighbors, cluster_label)
                cluster_label += 1

    def region_query(self, X, idx):
        return np.where(np.linalg.norm(X[idx] - X, axis=1) < self.eps)[0]

    def expand_cluster(self, X, idx, neighbors, cluster_label):
        self.labels_[idx] = cluster_label
        i = 0
        while i < len(neighbors):
            n = neighbors[i]
            if self.labels_[n] == -1:
                self.labels_[n] = cluster_label
            elif self.labels_[n] == 0:
                self.labels_[n] = cluster_label
                new_neighbors = self.region_query(X, n)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1
if __name__=="__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    dbscan.fit(X)

    print(dbscan.labels_)
