import numpy as np
from mapFeature import mapFeature
import matplotlib.pyplot as plt

def plotDecisionBoundary(theta, X, y):
    """PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
     the decision boundary defined by theta
     PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
     positive examples and o for the negative examples. X is assumed to be 
     a either 
     1) Mx3 matrix, where the first column is an all-ones column for the 
         intercept.
     2) MxN, N>3 matrix, where the first column is all-ones"""

    if X.shape[1] <= 3:
        #Only need 2 points to define a line, so choose two endpoints
        plot_x = np.c_[np.min(X[:,1]) - 2,  np.max(X[:,1]) + 2];
    
        #Calculate the decision boundary line
        left =  -1  / theta[2]
        right = theta[1]  * plot_x + theta[0]
        plot_y = left * right
        
        #Plot, and adjust axes for better viewing
        line, = plt.plot(plot_x.flatten(), plot_y.flatten(), c = 'b', label="Decision Boundary", marker="*")
         
        plt.axis([30, 100, 30, 100])
        
        return line
    else:
        #Here is the grid range
        u = np.linspace(-1, 1.5, num=50)
        v = np.linspace(-1, 1.5, num=50)
        
        A, B = np.meshgrid(u, v)
    
        z = np.zeros((u.size, v.size))
        #Evaluate z = theta*x over the grid
        for i in range(u.size):
            for j in range(v.size):
                z[i,j] = np.dot(mapFeature(u[i], v[j]), theta)
           
        z = z.T # important to transpose z before calling contour
        #Plot z = 0
        #Notice you need to specify the range [0, 0]
        return plt.contour(A, B, z)
   
