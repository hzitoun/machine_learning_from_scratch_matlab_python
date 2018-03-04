import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """PREDICT Predict the label of an input given a trained neural network
        p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
        trained weights of a neural network (Theta1, Theta2)"""
    
    #Useful values
    m = X.shape[0]
         
    a1 = sigmoid(np.c_[np.ones((m, 1)), X].dot(Theta1.T))
    
    a2 = sigmoid(np.c_[np.ones((m, 1)), a1].dot(Theta2.T))
    
    p = (np.argmax(a2, axis=1) + 1)
    
    p = np.where(p == 10, 0, p)
           
    return p


