import matplotlib.pyplot as plt
import numpy as np


def displayData(X, example_width=None):
    
 """DISPLAYDATA Display 2D data in a nice grid
   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
   stored in X in a nice grid. It returns the figure handle h and the 
   displayed array if requested."""
   
   
 #Compute rows, cols
 m, n = X.shape
 
 
 result = np.empty((0, 200))
 row = 0
 for row in range(10):
     new_row = np.zeros((20, 0))
     for col in range(10):
          new_row = np.c_[X[row].reshape(20, 20).T, new_row]
          row = row + 1  
     result = np.r_[result, new_row]
     
     
 #Display Image
 plt.imshow(result, cmap='gray', interpolation='nearest') 
   

