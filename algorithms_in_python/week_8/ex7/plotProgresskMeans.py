import itertools

import matplotlib.pyplot as plt
import numpy as np
from plotDataPoints import plot_data_points


def plot_progress_kmeans(X, centroids, previous, idx, K, i):
    """A helper function that displays the progress of
    #k-Means as it is running. It is intended for use only with 2D data.
    #   plot_progress_kmeans(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.
    """

    # Plot the examples
    plot_data_points(X, idx, K)

    # Plot the centroids as black x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=60, lw=3, edgecolor='k')
    plt.draw()

    # Plot the history of the centroids with lines
    count = centroids.shape[0]

    arrstr = np.char.mod('%d', np.arange(K))
    c = itertools.cycle("".join(arrstr))
    rgb = np.eye(K, 3)

    for j in range(count):
        x = np.r_[centroids[j, 0], previous[j, 0]]
        y = np.r_[centroids[j, 1], previous[j, 1]]
        plt.plot(x, y, c=rgb[int(next(c))])
        plt.draw()

    plt.title('Iteration number {}'.format(i))

    plt.show(block=False)
