import numpy as np


def selectThreshold(yval, pval):
    """Finds the best threshold (epsilon) to use for selecting
        outliers
       [bestEpsilon bestF1] = selectThreshold(yval, pval) finds the best
       threshold to use for selecting outliers based on the results from a
       validation set (pval) and the ground truth (yval).
    """

    bestEpsilon = 0
    bestF1 = 0
    stepsize = (max(pval) - min(pval)) / 1000
    epsilons = np.arange(min(pval), max(pval), stepsize)

    for epsilon in epsilons:
        pred = (pval < epsilon)
        tp = np.sum(np.logical_and((pred == 1), (yval == 1)).astype(float))
        fp = np.sum(np.logical_and((pred == 1), (yval == 0)).astype(float))
        fn = np.sum(np.logical_and((pred == 0), (yval == 1)).astype(float))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * (prec * rec) / (prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
