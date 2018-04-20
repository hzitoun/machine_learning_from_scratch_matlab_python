import numpy as np

def linearRegCostFunction(X, y, theta, reg_lambda):
    """Computes cost and gradient for regularized linear 
     regression with multiple variables
       [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
       cost of using theta as the parameter for linear regression to fit the 
       data points in X and y. Returns the cost in J and the gradient in grad
    """
    
    # Initialize some useful values
    m = len(y) # number of training examples
    
    predictions = X.dot(theta)
    
    errors = predictions - y
    
    #we dont regularize theta[0]
    J = (1.0/(2 * m)) * np.sum(errors ** 2) + (reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
    
    grad = (1.0/m) * X.T.dot(errors)
        
    column_size = grad.shape[1]
        
    grad = np.r_[grad[0, :].reshape(1, column_size), grad[1:, :] + (reg_lambda /m) * theta[1:, :]]
    
    return J, grad
