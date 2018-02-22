import numpy as np

def featureNormalize(X):
    """Normalizes the features in X. It returns a normalized 
    version of X where the mean value of each feature is 0 and 
    the standard deviation is 1. 
    This is often a good preprocessing step to do when
    working with learning algorithms. """
    
    mu = np.mean(X) #mean of each column
    sigma = np.std(X) #standard deviation for each column
    X_norm = np.divide(np.subtract(X, mu), sigma) 
    
    return X_norm, mu, sigma
