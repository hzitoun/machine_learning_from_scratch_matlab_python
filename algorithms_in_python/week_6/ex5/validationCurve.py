import numpy as np
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg

def validationCurve(X, y, Xval, yval):
    """Generates the train and validation errors needed to
    plot a validation curve that we can use to select lambda
      VALIDATIONCURVE(X, y, Xval, yval) returns the train
           and validation errors (in error_train, error_val)
           for different values of lambda. You are given the training set (X,
           y) and validation set (Xval, yval).
    """
    
    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape(-1, 1)
    
    # You need to return these variables correctly.
    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))
      
    
    for i in range(len(lambda_vec)):
        curr_lambda = lambda_vec[i]
        theta = trainLinearReg(X, y, curr_lambda)
        error_train[i], tmp = linearRegCostFunction(X, y, theta, 0)
        error_val[i],  tmp  = linearRegCostFunction(Xval, yval, theta, 0)        
    
    
    return lambda_vec, error_train, error_val