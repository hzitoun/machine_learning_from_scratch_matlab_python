import numpy as np

from cofiCostFunc import cofiCostFunc
from computeNumericalGradient import computeNumericalGradient


def checkCostFunction(reg_lambda=0):
    """ Creates a collaborative filtering problem
        to check your cost function and gradients
        checkCostFunction(lambda) Creates a collaborative filtering problem
        to check your cost function and gradients, it will output the
        analytical gradients produced by your code and the numerical gradients
        (computed using computeNumericalGradient). These two gradient
        computations should result in very similar values."""

    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t.dot(Theta_t.T)
    rand_data = np.random.randn(*Y.shape)
    Y[np.where(rand_data > 0.5)] = 0
    R = np.zeros(Y.shape)
    R[np.where(Y != 0)] = 1

    # Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    # build params
    params = np.r_[X.flatten(), Theta.flatten()].reshape(-1, 1)

    costFunc = lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, reg_lambda)

    numgrad = computeNumericalGradient(costFunc, params)

    cost, grad = costFunc(params)

    # make sure both grad have the same shape
    grad = grad.reshape(numgrad.shape)
    print(np.c_[numgrad.ravel(), grad.ravel()])
    print('The above two columns you get should be very similar. '
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your cost function implementation is correct, then \n the relative difference '
          'will be small (less than 1e-9). '
          '\n \nRelative Difference: \n', diff)
