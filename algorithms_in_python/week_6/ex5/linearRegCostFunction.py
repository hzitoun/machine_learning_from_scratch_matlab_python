import numpy as np

def linearRegCostFunction(X, y, theta, reg_lambda, returnOnlyGrad= None, returnOnlyCost = None, flattenResult = None ):
    """Computes cost and gradient for regularized linear 
     regression with multiple variables
       [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
       cost of using theta as the parameter for linear regression to fit the 
       data points in X and y. Returns the cost in J and the gradient in grad
    """
    
    # Initialize some useful values
    m = len(y) # number of training examples
    

    theta_column_size = y.shape[1]
    theta_row_size  = X.shape[1]
    
    theta = theta.reshape(theta_row_size, theta_column_size)
    
    predictions = X.dot(theta)
    
    errors = predictions - y
    
    #we dont regularize theta[0]
    J = (1.0/(2 * m)) * np.sum(errors ** 2) + (reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
    
    grad = (1.0/m) * X.T.dot(errors)
        
    #we dont regularize theta[0] (bias)
    grad = np.r_[grad[0, :].reshape(1, theta_column_size), grad[1:, :] + (reg_lambda /m) * theta[1:, :]]
    
    if returnOnlyGrad:
        if flattenResult:
            return grad.flatten()
        return grad
    
    if returnOnlyCost:
        return J
    
    return J, grad
