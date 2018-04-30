import matplotlib.pyplot as plt


def plot_data_points(x, idx, K):
    """
    Plots data points in X, coloring them so that those with the same
    index assignments in idx have the same color
    """
    plt.scatter(x[:, 0], x[:, 1], s=40, c=idx, cmap=plt.cm.prism)
