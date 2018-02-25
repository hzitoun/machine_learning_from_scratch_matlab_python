import numpy as np

def computeCost(X,y, theta):
    """Computes cost for linear regression
     J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y"""

    m = y.size  #number of training examples
    predictions = np.dot(X, theta) #predictions of hypothesis on all m examples
    errors = np.subtract(predictions, y)
    sqrErrors = np.power(errors, 2) #squared errors
    J = 1.0/ (2.0 * m) * sqrErrors.sum() # average squared error among predictions
    return J
    

