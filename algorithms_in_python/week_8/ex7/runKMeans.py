import numpy as np
from computeCentroids import compute_centroids
from findClosestCentroids import find_closest_centroids
from plotProgresskMeans import plot_progress_kmeans


def pause():
    input("")


def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    """[centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
       plot_progress) runs the K-Means algorithm on data matrix X, where each
       row of X is a single example.
        It uses initial_centroids used as the
       initial centroids. max_iters specifies the total number of interactions
       of K-Means to execute. plot_progress is a true/false flag that
       indicates if the function should also plot its progress as the
       learning happens. This is set to false by default. runkMeans returns
       centroids, a Kxn matrix of the computed centroids and idx, a m x 1
       vector of centroid assignments (i.e. each entry in range [1..K])
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print('K-Means iteration {}/{}...'.format(i, max_iters))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plot_progress_kmeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            print('Press enter to continue.\n')
            pause()

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    return centroids, idx
