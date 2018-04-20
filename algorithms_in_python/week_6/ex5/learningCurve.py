import numpy as np
import trainLinearReg as trainLinearReg
import linearRegCostFunction as linearRegCostFunction

def learningCurve(X, y, Xval, yval, reg_lambda):
    """Generates the train and cross validation set errors needed 
       to plot a learning curve.
       [error_train, error_val] = ...
           LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
           cross validation set errors for a learning curve. In particular, 
           it returns two vectors of the same length - error_train and 
           error_val. Then, error_train(i) contains the training error for
           i examples (and similarly for error_val(i)).
    
       In this function, you will compute the train and test errors for
       dataset sizes from 1 up to m. In practice, when working with larger
       datasets, you might want to do this in larger intervals.
    """
    
    # Number of training examples
    m = X.shape[0]
    
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m,1))
    
    for i in range(m):
        theta = trainLinearReg(X[0:i, :], y[0:i], reg_lambda)
        error_train[i], tmp = linearRegCostFunction(X[0:i, :], y[0:i], theta, 0)
        #You use the entire validation set to measure J_cv because you want to know how well 
        #the theta values work on the validation set. You get a better (average) measurement 
        #by using the entire CV set.
        error_val(i),  tmp  = linearRegCostFunction(Xval, yval, theta, 0)
        
    return error_train, error_val
