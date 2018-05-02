import numpy as np


def kmeans_init_centroids(X, K):
    """This function initializes K centroids that are to be
       used in K-Means on the dataset X
       centroids = kmeans_init_centroids(X, K) returns K initial centroids to be
       used with the K-Means on the dataset X
    """

    # Randomly reorder the indices of examples
    np.random.seed(0)
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[0:K], :]

    return centroids
