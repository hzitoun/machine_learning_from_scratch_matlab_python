import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, reg_lambda, returnCostOnly=False,
                 returnGradOnly=False):
    """Collaborative filtering cost function
       [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
       num_features, lambda) returns the cost and gradient for the
       collaborative filtering problem.
    """

    # Unfold the U and W matrices from params
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features:].reshape((num_users, num_features))

    errors = (X.dot(Theta.T) - Y) * R
    J = 1 / 2 * np.sum(np.sum(errors ** 2))

    penalty = (reg_lambda / 2) * (np.sum(np.sum(Theta ** 2)) + np.sum(np.sum(X ** 2)))
    J = J + penalty

    X_grad = errors.dot(Theta) + reg_lambda * X
    Theta_grad = errors.T.dot(X) + reg_lambda * Theta

    grad = np.r_[X_grad.flatten(), Theta_grad.flatten()]

    if returnGradOnly:
        return grad.flatten()
    if returnCostOnly:
        return J

    return J, grad
