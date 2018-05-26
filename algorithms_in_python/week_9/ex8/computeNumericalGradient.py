import numpy as np


def computeNumericalGradient(J, theta):
    """Computes the gradient using "finite differences"
       and gives us a numerical estimate of the gradient.
       numgrad = computeNumericalGradient(J, theta) computes the numerical
       gradient of the function J around theta. Calling y = J(theta) should
       return the function value at theta."""

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    length = theta.shape[0]

    e = 1e-4
    for p in range(length):
        # Set perturbation vector
        perturb[p] = e
        loss1, tmp = J(theta - perturb)
        loss2, tmp = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad

