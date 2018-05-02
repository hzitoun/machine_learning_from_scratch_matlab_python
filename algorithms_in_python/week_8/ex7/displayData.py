import matplotlib.pyplot as plt
import numpy as np


def displayData(X, nrows=10, ncols=10):
    # set up array
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols,
                              figsize=(nrows, ncols))

    nblock = int(np.sqrt(X.shape[1]))

    # loop over randomly drawn numbers
    ct = 0
    for ii in range(nrows):
        for jj in range(ncols):
            # ind = np.random.randint(X.shape[0])
            tmp = X[ct, :].reshape(nblock, nblock, order='F')
            axarr[ii, jj].imshow(tmp, cmap='gray')
            plt.setp(axarr[ii, jj].get_xticklabels(), visible=False)
            plt.setp(axarr[ii, jj].get_yticklabels(), visible=False)
            plt.minorticks_off()
            ct += 1

    fig.subplots_adjust(hspace=0, wspace=0)
