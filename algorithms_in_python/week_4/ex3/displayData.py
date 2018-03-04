import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


def displayData(X):
    
     """DISPLAYDATA Display 2D data in a nice grid
       [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
       stored in X in a nice grid. It returns the figure handle h and the 
       displayed array if requested."""
       
     
     print(X.shape)
     #Compute rows, cols
     m, n = X.shape
     
     
     nbImagesPerRow = int(sqrt(m))
     columnsCount = (20 + 2) * nbImagesPerRow
     
     result = np.empty((0, columnsCount))
     row = 0
     while row < m:
         new_row = np.empty((20, 0))
         for col in range(nbImagesPerRow):
              new_row = np.c_[new_row, X[row].reshape(20, 20).T]
              new_row = np.c_[new_row, np.zeros((20,2))]
              row = row + 1  
         result = np.r_[result, new_row]
         result = np.r_[result, np.zeros((1, columnsCount))]
         
     #Display Image
     plt.imshow(result, cmap='gray', interpolation='nearest') 
