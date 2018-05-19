import numpy as np


def estimateGaussian(X):
    """This function estimates the parameters of a
    Gaussian distribution using the data in X
       [mu sigma2] = estimateGaussian(X),
       The input X is the dataset with each n-dimensional data point in one row
       The output is an n-dimensional vector mu, the mean of the data set
       and the variances sigma^2, an n x 1 vector
     """

    # Useful variables
    m, n = X.shape

    mu = np.mean(X, 0).reshape(1, n)
    sigma2 = (1.0 / m) * sum((X - mu) ** 2).reshape(1, n)

    return mu, sigma2
