""" Machine Learning Online Class - Exercise 4 Neural Network Learning

  Instructions
  ------------

  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions
  in this exericse:

     sigmoidGradient.py
     randInitializeWeights.py
     nnCostFunction.py

  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.
"""

# Initialization

from predict import predict
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients


import scipy.optimize as op
import matplotlib.pyplot as plt
import numpy as np

def pause():
    input("")

#Setup the parameters you will use for this exercise


input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

X = np.loadtxt('ex4_features.csv', delimiter =",")
m,_ = X.shape

y = np.loadtxt('ex4_labels.csv', delimiter =",").reshape(m, 1)


#Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100)

displayData(X[rand_indices])

plt.draw()
plt.show(block=False)

print('Program paused. Press enter to continue.\n')
pause()


## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2

Theta1 = np.loadtxt('Theta1.csv', delimiter =",")
Theta2 = np.loadtxt('Theta2.csv', delimiter =",")

# Unroll parameters
nn_params = np.r_[Theta1.ravel(), Theta2.ravel()]

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
reg_lambda = 0;

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)

print('Cost at parameters (loaded from ex4weights) should be about 10.4414339388 \n', J)

print('\nProgram paused. Press enter to continue.\n')
pause()


## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
reg_lambda = 1

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)

print('Cost at parameters (loaded from ex4weights): should be about 10.5375744448 ', J)

print('Program paused. Press enter to continue.\n')
pause()



## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.py file.
#

print('\nEvaluating sigmoid gradient...\n')

test_array = np.array([[1, -0.5, 0, 0.5, 1]])
g = sigmoidGradient(test_array)
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n ')
print(g)
print('\n\n')

print('Program paused. Press enter to continue.\n')
pause()


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.r_[initial_Theta1.ravel(), initial_Theta2.ravel()]

print("init nn params shape", initial_nn_params.shape)
pause()

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... \n')

#  Check gradients by running checkNNGradients
checkNNGradients()

print('\nProgram paused. Press enter to continue.\n')
pause()


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
reg_lambda = 3;
checkNNGradients(reg_lambda)

# Also output the costFunction debugging values
debug_J, grad  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 10): #f ', 
         '\n(this value should be about 0.576051)\n\n', debug_J)

print('Program paused. Press enter to continue.\n')
pause()


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  You should also try different values of lambda
reg_lambda = 1;

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
max_iter = 200

# Short hand for cost function
costFunc = lambda params: nnCostFunction(params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda,
                                         returnOnlyCost=True)
gradFunc = lambda params: nnCostFunction(params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda, 
                                         returnOnlyGrad=True, flattenResult=True)

# Run fmincg to obtain the optimal theta
Result = op.minimize(fun = costFunc, x0 = initial_nn_params, method = 'TNC', jac = gradFunc, 
                     options={'maxiter' : max_iter, 'disp': True})

optimal_nn_params = Result.x

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(optimal_nn_params[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(optimal_nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, hidden_layer_size + 1))
   
print('\n Neural Network trained successfully... \n')

print('Program paused. Press enter to continue.\n')
pause()



## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:, 1:])
plt.draw()
plt.show(block=False)

print('\nProgram paused. Press enter to continue.\n')
pause()

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

p = predict(Theta1, Theta2, X).reshape(m, 1)


rand_indices = np.random.choice(m, 5)

print(p[rand_indices])
print(y[rand_indices])

print('\nTraining Set Accuracy: ', np.multiply(np.mean((p == y).astype(int)), 100), '\n')
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
    plt.draw()
    plt.show(block=False)
    pred = predict(Theta1, Theta2, imageData)
    print('\nNeural Network Prediction: ', pred, '%')
    
    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
      break