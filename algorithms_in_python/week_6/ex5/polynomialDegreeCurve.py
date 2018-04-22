import numpy as np
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize


def polynomialDegreeCurve(X, y, Xval, yval, reg_lambda):
    """Error cruve in function of degree of polynimal d
    """
    
    dimensions = np.arange(1, 80).reshape(-1, 1)
    
    # You need to return these variables correctly.
    error_train = np.zeros((len(dimensions), 1))
    error_val = np.zeros((len(dimensions), 1))
    
    m_train_set = X.shape[0]
    m_val_set = Xval.shape[0]
      
    
    for i in range(len(dimensions)):
        dimension = dimensions[i]
        
        X_poly = polyFeatures(X, dimension)
        X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
        X_poly = np.c_[np.ones((m_train_set, 1)), X_poly]   
        
        X_poly_val = polyFeatures(Xval, dimension)
        X_poly_val = X_poly_val - mu
        X_poly_val = X_poly_val / sigma
        X_poly_val = np.c_[np.ones((m_val_set, 1)), X_poly_val] 
        
        theta = trainLinearReg(X_poly, y, reg_lambda)
        error_train[i], tmp = linearRegCostFunction(X_poly, y, theta, 0)
        error_val[i],  tmp  = linearRegCostFunction(X_poly_val, yval, theta, 0)        
    
    
    return dimensions, error_train, error_val