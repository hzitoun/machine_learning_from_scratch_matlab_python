import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda, 
                   returnOnlyGrad = None, returnOnlyCost = None, flattenResult=None):
    
    """Implements the neural network cost function for a two layer
       neural network which performs classification
       [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
       X, y, lambda) computes the cost and gradient of the neural network. The
       parameters for the neural network are "unrolled" into the vector
       nn_params and need to be converted back into the weight matrices. 
     
       The returned parameter grad should be a "unrolled" vector of the
       partial derivatives of the neural network.
    """
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, hidden_layer_size + 1))
                
    # Setup some useful variables
    m = np.shape(X)[0]
        

    # Part 1: Feedforward the neural network and return the cost in the 
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost computed in ex4.py
    
    # Explode each row in y into 10 dimension vector
    # recode y to Y
    Y = np.zeros((m, num_labels))
    
    for i in range(m):
      Y[i, y[i, 0]]= 1
    
    # 1. Feed-forward to compute h = a3.
    a1 = np.c_[np.ones((m, 1)), X]
    z2 = a1.dot(Theta1.T)
    a2 = np.c_[np.ones((z2.shape[0], 1)), sigmoid(z2)]
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    h = a3
    
    
    J = np.sum(np.sum((-Y) * np.log(h) - (1-Y) * np.log(1-h), 1)) / m
    
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    
    sigma3 = h - Y
    sigma2 = (sigma3.dot(Theta2)) * sigmoidGradient(np.c_[np.ones((np.shape(z2)[0], 1)), z2])
    
    delta2 =  sigma3.T.dot(a2)
    delta1 =  sigma2[:, 1:].T.dot(a1)
    
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #
    
    # we dont regularize bias
    J = J + (reg_lambda/(2.0 * m)) * np.sum(np.sum(Theta1[:,1:] * Theta1[:,1:]))
    J = J + (reg_lambda/(2.0 * m)) * np.sum(np.sum(Theta2[:,1:] * Theta2[:,1:]))
    
    # calculate penalties (we dont regularize bias)
    p1 = (reg_lambda/m) * np.c_[np.zeros((np.shape(Theta1)[0], 1)), Theta1[:,1:]]
    p2 = (reg_lambda/m) * np.c_[np.zeros((np.shape(Theta2)[0], 1)), Theta2[:,1:]]
    
    Theta1_grad = delta1/m + p1
    Theta2_grad = delta2/m + p2
    
    # Unroll gradients
    grad = np.r_[Theta1_grad.ravel(), Theta2_grad.ravel()]

    
    if (returnOnlyGrad):
        if (flattenResult):
            return grad.flatten()
        return grad
    
    if (returnOnlyCost):
        return J
    
    return (J, grad)
