import numpy as np

def polyFeatures(X, p):
    """Maps X (1D vector) into the p-th power
       [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
       maps each example into its polynomial features where
       X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    """
    X_poly = np.zeros((X.shape[0], p))
    
    for j in range(1, p + 1):
        X_poly[:, j - 1] = X[:, 0]  ** j
        #print("X_poly[:, {}]".format(j - 1), X_poly[:, j - 1])
    
    return X_poly
