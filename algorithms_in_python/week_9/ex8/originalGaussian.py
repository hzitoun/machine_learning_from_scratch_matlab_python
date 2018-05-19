import math

import numpy as np


def originalGaussian(X, mu, Sigma2):
    exp_data = ((X - mu) ** 2) / (2 * Sigma2)
    lef_data = (1.0 / (np.sqrt(2 * math.pi * Sigma2)))
    p = lef_data * np.exp(- exp_data)
    p = np.prod(p, 1)  # products of each row
    return p
