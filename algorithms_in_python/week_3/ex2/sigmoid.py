import numpy as np

def sigmoid(z):
    """%SIGMOID Compute sigmoid function
    g = SIGMOID(z) computes the sigmoid of z.
    """
    return 1.0 / (1 + np.exp(-z))

