import numpy as np
from lrGradient import lrGradient
from lrCostFunction import lrCostFunction
import scipy.optimize as op


def oneVsAll(X, y, num_labels, reg_lambda):
    """ONEVSALL trains multiple logistic regression classifiers and returns all
     the classifiers in a matrix all_theta, where the i-th row of all_theta
     corresponds to the classifier for label i
        [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
        logistic regression classifiers and returns each of these classifiers
        in a matrix all_theta, where the i-th row of all_theta corresponds
        to the classifier for label i
    """

    #Some useful variables
    m, n = X.shape
    
    #You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))
    
    #Add ones to the X data matrix
    X = np.c_[np.ones((m, 1)), X]
    
    for c in range(num_labels):
        
        print("optimizing theta", c)
        
        #Set Initial theta
        initial_theta = np.zeros((n + 1, 1))
                
        # Run fmincg to obtain the optimal theta
        Result = op.minimize(fun = lrCostFunction, x0 = initial_theta, args = (X, (y == c) * 1, reg_lambda), method = 'TNC', jac = lrGradient,
                             options={'maxiter' : 50})
        optimal_theta = Result.x
        
        all_theta[c,:] =  optimal_theta
        
    return all_theta



