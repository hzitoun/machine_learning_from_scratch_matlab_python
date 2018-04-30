import numpy as np


def find_closest_centroids(x, centroids):
    """ Computes the centroid memberships for every example
       idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
       in idx for a dataset X where each row is a single example. idx = m x 1
       vector of centroid assignments (i.e. each entry in range [1..K])
    """
    m, n = x.shape
    idx = np.zeros((m, 1))

    # The algorithm assigns every training example x[i] to its closest centroid
    for i in range(m):
        V = centroids - x[i, :]
        D = np.linalg.norm(V, axis=1, ord=2) ** 2
        idx[i] = np.argmin(D)

    return idx
