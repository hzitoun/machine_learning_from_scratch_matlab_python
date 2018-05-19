import numpy as np


def multivariateGaussian(X, mu, Sigma2):
    """Computes the probability density function of the
        multivariate gaussian distribution.
        p = multivariateGaussian(X, mu, Sigma2) Computes the probability
        density function of the examples X under the multivariate gaussian
        distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
        treated as the covariance matrix. If Sigma2 is a vector, it is treated
        as the \sigma^2 values of the variances in each dimension (a diagonal
        covariance matrix)
    """

    k = mu.shape[1]

    if len(Sigma2.shape) > 0 and ((Sigma2.shape[0] == 1) or (len(Sigma2.shape) > 1 and Sigma2.shape[1] == 1)):
        Sigma2 = np.diag(Sigma2.ravel())

    X = X - mu

    p = (2 * np.pi) ** (-k / 2)

    p = p * np.linalg.det(Sigma2) ** -0.5

    p = p * np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(Sigma2)) * X, 1))

    return p.reshape(-1, 1)

