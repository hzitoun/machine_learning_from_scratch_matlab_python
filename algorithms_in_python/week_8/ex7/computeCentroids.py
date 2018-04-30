import numpy as np


def compute_centroids(X, idx, K):
    """Returns the new centroids by computing the means of the
       data points assigned to each centroid.
       centroids = compute_centroids(X, idx, K) returns the new centroids by
       computing the means of the data points assigned to each centroid. It is
       given a dataset X where each row is a single data point, a vector
       idx of centroid assignments (i.e. each entry in range [1..K]) for each
       example, and K, the number of centroids. You should return a matrix
       centroids, where each row of centroids is the mean of the data points
       assigned to it.
    """

    # Useful variables
    m, n = X.shape

    centroids = np.zeros((K, n))

    unique_indexes = np.unique(idx)

    for i, val in enumerate(unique_indexes):
        Ci_indexes = np.argwhere(idx == val)[:, 0]
        Ci = X[Ci_indexes]
        centroids[i, :] = np.mean(Ci, axis=0)

    return centroids
