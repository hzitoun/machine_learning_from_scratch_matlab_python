import numpy as np
import computeCostMulti as costMultiModule

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha """
    
    m = y.size #number of training examples
    J_history = np.zeros((num_iters, 1))
    
    for iter in range(1, num_iters):
        #do linear regression with identity (f(x) = x) as an activation function
         prediction = np.dot(X, theta)
         errors = prediction - y
         delta = (1.0/m) * np.dot(X.T, errors)
         #update weight
         theta = theta - alpha * delta
         #save the cost J in every iteration
         J_history[iter] = costMultiModule.computeCostMulti(X, y, theta)
         
    return J_history, theta
