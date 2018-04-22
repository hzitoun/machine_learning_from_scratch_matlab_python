import numpy as np
import scipy.optimize as op
from linearRegCostFunction import linearRegCostFunction

def trainLinearReg(X, y, reg_lambda):
    """TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
       regularization parameter lambda
       [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
       the dataset (X, y) and regularization parameter lambda. Returns the
       trained parameters theta.
    """
    
    #Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))
        
    # Create "short hand" for the cost function to be minimized
    costFunction = lambda theta :  linearRegCostFunction(X, y, theta, reg_lambda, returnOnlyCost = True)
    
    gradFunc = lambda theta : linearRegCostFunction(X, y, theta, reg_lambda, returnOnlyGrad = True, flattenResult = True)
    
    #should finish learning (reach local minima) in 4 iterations
    max_iter = 200
    
       # Run fmincg to obtain the optimal theta
    Result = op.minimize(fun = costFunction, x0 = initial_theta, method = 'TNC', jac = gradFunc, 
                         options={'maxiter' : max_iter, 'disp': True})
    
    optimal_theta = Result.x
    
    return optimal_theta
