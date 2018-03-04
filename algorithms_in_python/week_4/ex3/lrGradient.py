from sigmoid import sigmoid
import numpy as np

def lrGradient(theta, X,y, reg_lambda, flattenResult=True):
     m,n = X.shape     
     theta = theta.reshape((n,1))
     prediction = sigmoid(np.dot(X, theta))
     errors = np.subtract(prediction, y)
     grad = (1.0/m) * np.dot(X.T, errors)

     grad_with_regul = grad[1:] + (reg_lambda/m) * theta[1:]
     firstRow = grad[0, :].reshape((1,1))
     grad = np.r_[firstRow, grad_with_regul]
     
     if  flattenResult:    
         return grad.flatten()
     
     return grad

