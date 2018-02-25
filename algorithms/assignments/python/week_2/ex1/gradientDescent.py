import numpy as np
import computeCost as costModule

def gradientDescent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha """
    
    m = y.size #number of training examples
    J_history = np.zeros((num_iters, 1))
    
    print("theta before for", theta.shape)
    
    #train 
    for iter in range(num_iters):
         #do linear regression with identity (f(x) = x) as an activation function
         prediction = np.dot(X, theta)
         errors = np.subtract(prediction, y)
         delta = (1.0/m) * np.dot(X.T, errors)
         #update weight
         theta = theta - alpha * delta
         #save the cost J in every iteration
         J_history[iter] = costModule.computeCost(X, y, theta)
     
    return J_history, theta
    