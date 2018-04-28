import numpy as np
from svmModel import SVMModel


def dataset3_params(x, y, xval, yval):
    """
    Returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel
       [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
       sigma. You should complete this function to return the optimal C and
       sigma based on a cross-validation set.
    """

    # You need to return the following variables correctly.
    c = 1
    sigma = 0.3

    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_error = 1

    for current_c in values:
        for current_sigma in values:

            # ALWAYS train the model on training sets (X and y)
            model = SVMModel()
            model.train(x, y, current_c, kernel_type='rbf', tol=1e-3, max_passes=5, sigma=sigma)

            # AND evaluate it on cross validation set
            predictions = model.predict(xval)
            error = np.mean((predictions == yval).astype(int))

            if error < max_error:
                max_error = error
                c = current_c
                sigma = current_sigma

    return c, sigma
