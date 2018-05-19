import matplotlib.pyplot as plt
import numpy as np

from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    """Visualizes the dataset and its estimated distribution.
       visualizeFit(X, p, mu, sigma2) This visualization shows you the
       probability density function of the Gaussian distribution. Each example
       has a location (x1, x2) that depends on its feature values.
    """

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))

    Z = multivariateGaussian(np.c_[X1.ravel(), X2.ravel()], mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    cont_levels = [10 ** exp for exp in range(-20, 0, 3)]

    plt.contour(X1, X2, Z, cmap=plt.cm.Paired, alpha=0.9, levels=cont_levels)
