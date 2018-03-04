from sigmoid import sigmoid
import numpy as np

def lrCostFunction(theta, X, y, reg_lambda):
    
     """LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
       regularization
       J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
       theta as the parameter for regularized logistic regression and the
       gradient of the cost w.r.t. to the parameters. 
     """

     m, n = X.shape #number of training examples
     theta = theta.reshape((n,1))
     
     prediction = sigmoid(X.dot(theta))

     cost_y_1 = (1 - y) * np.log(1 - prediction)
     cost_y_0 = -1 * y * np.log(prediction)

     J = (1.0/m) * np.sum(cost_y_0 - cost_y_1) + (reg_lambda/(2.0 * m)) * np.sum(np.power(theta[1:], 2))
      
     return J
