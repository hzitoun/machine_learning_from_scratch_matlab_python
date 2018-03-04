""" Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

  Instructions
  ------------
 
  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions 
  in this exericse:

     lrCostFunction.m (logistic regression cost function)
     oneVsAll.m
     predictOneVsAll.m
     predict.m

  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.
"""

from predict import predict
from displayData import displayData
import matplotlib.pyplot as plt
import numpy as np

def pause():
    input("")

# Setup the parameters you will use for this exercise
input_layer_size  = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9  
                         

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

X = np.loadtxt('MNIST_DATA.csv', delimiter =",")
m,_ = X.shape

y = np.loadtxt('MNIST_DATA_LABEL.csv', delimiter =",").reshape(m, 1)

#since python indexes start with 0 and matlab's ones start with 10, we replace all 10 by 0
y = np.where(y == 10, 0, y)

#Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100)

displayData(X[rand_indices])

plt.show()

print('Program paused. Press enter to continue.\n')
pause()

# ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2

Theta1 = np.loadtxt('Theta1.csv', delimiter =",")
Theta2 = np.loadtxt('Theta2.csv', delimiter =",") 

# ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

p = predict(Theta1, Theta2, X)
y = y.reshape((m))

print(p[rand_indices])

print("training Set Accuracy:: ", np.multiply(np.mean((p == y).astype(int)), 100), "")

print('Program paused. Press enter to continue.\n')
pause()


#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rand_indices = np.random.choice(m, 600)

for i in range(600):
    # Display 
    print('\nDisplaying Example Image\n')
    imageData = X[rand_indices[i]].reshape(400, 1).T
    displayData(imageData)
    plt.show(block=False)
    pred = predict(Theta1, Theta2, imageData)
    print('\nNeural Network Prediction: ', pred)
    
    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
      break