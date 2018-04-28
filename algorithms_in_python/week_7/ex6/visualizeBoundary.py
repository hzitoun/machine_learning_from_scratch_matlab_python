import numpy as np
import matplotlib.pyplot as plt


def visualizeBoundary(X, y, model, varargin=None):
    """VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
       VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
       boundary learned by the SVM and overlays the data on it
    """
    # Plot the training data on top of the boundary
    # plotData(X, y)
    #  Make classification predictions over a grid of values
    # Here is the grid range
    u = np.linspace(min(X[:, 0]), max(X[:, 0]), num=100).T
    v = np.linspace(min(X[:, 1]), max(X[:, 1]), num=100).T
    X1, X2 = np.meshgrid(u, v)

    m, n = X1.shape
    vals = np.zeros(X1.shape)

    for i in range(n):
        this_X = np.c_[X1[:, i], X2[:, i]]
        vals[:, i] = model.predict(this_X).flatten()

    # Plot the SVM boundary
    plt.contour(X1, X2, vals)
