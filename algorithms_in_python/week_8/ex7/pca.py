import numpy as np


def pca(X):
    """PCA Run principal component analysis on the dataset X
    [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """
    # Useful values
    m, n = X.shape

    Sigma = (1.0 / m) * X.T.dot(X)

    U, S, V = np.linalg.svd(Sigma)

    return U, S
