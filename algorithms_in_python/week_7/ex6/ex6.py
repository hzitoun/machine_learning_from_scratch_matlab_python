""" # Exercise 6 | Support Vector Machines
"""

from plotData import plotData
from svmModel import SVMModel
from visualizeBoundaryLinear import visualizeBoundaryLinear
from visualizeBoundary import visualizeBoundary
from dataset3Params import dataset3_params

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def pause():
    input("")


print('Loading and Visualizing Data ...\n')
plt.interactive(False)
# Load from ex6data1: 
# You will have X, y in your environment
mat_contents = sio.loadmat('ex6data1.mat')
X = mat_contents['X']
y = mat_contents['y'].flatten()

# Plot training data
plotData(X, y)
plt.draw()
plt.show(block=False)

print('Program paused. Press enter to continue.\n')
pause()

""" ## Part 2: Training Linear SVM
  The following code will train a linear SVM on the dataset and plot the
  decision boundary learned."""

print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = SVMModel()
model.train(X, y, C, kernel_type='lnr', tol=1e-3, max_passes=20)
visualizeBoundaryLinear(X, y, model)
plt.draw()
plt.show(block=False)

print("prediction is", model.predict(np.array([[1, 1.75]])))

print('Program paused. Press enter to continue.\n')
pause()


""" ## Part 3: Implementing Gaussian Kernel ===============
  You will now implement the Gaussian kernel to use
  with the SVM. You should complete the code in gaussianKernel.py
"""
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([[1, 2, 1]])
x2 = np.array([[0, 4, -1]])
sigma = 2

sim = SVMModel.gaussian_kernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {}'
      ' : \n\t{}\n(for sigma = 2, this value should be about 0.324652)\n'.format(sigma, sim))

print('Program paused. Press enter to continue.\n')
pause()

"""## Part 4: Visualizing Dataset 2
  The following code will load the next dataset into your environment and 
  plot the data. 
"""

print('Loading and Visualizing Data ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
mat_contents = sio.loadmat('ex6data2.mat')
X = mat_contents['X']
y = mat_contents['y'].flatten()

# Plot training data
plt.figure(2)
plotData(X, y)
plt.draw()
plt.show(block=False)

print('Program paused. Press enter to continue.\n')
pause()

"""## Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
  After you have implemented the kernel, we can now use it to train the 
  SVM classifier.
"""
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

# SVM Parameters
C = 1
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

model = SVMModel()
model.train(X, y, C, kernel_type='rbf', tol=1e-3, max_passes=5, sigma=sigma)
visualizeBoundary(X, y, model)
plt.draw()
plt.show(block=False)

print('Program paused. Press enter to continue.\n')
pause()


"""
## Part 6: Visualizing Dataset 3 ================
 The following code will load the next dataset into your environment and 
  plot the data. 
"""

print('Loading and Visualizing Data ...\n')

# Load from ex6data3:
# You will have X, y in your environment
mat_contents = sio.loadmat('ex6data3.mat')
X = mat_contents['X']
y = mat_contents['y'].flatten()
# Plot training data
plt.figure(3)
# Plot training data
plotData(X, y)
plt.draw()
plt.show(block=False)
print('Program paused. Press enter to continue.\n')
pause()

""""## Part 7: Training SVM with RBF Kernel (Dataset 3)
 This is a different dataset that you can use to experiment with. Try
  different values of C and sigma here.
"""

X = mat_contents['X']
y = mat_contents['y'].flatten()
Xval = mat_contents['Xval']
yval = mat_contents['yval'].flatten()

# Try different SVM Parameters here
C, sigma = dataset3_params(X, y, Xval, yval)

# Train the SVM
model = SVMModel()
model.train(X, y, C, kernel_type='rbf', tol=1e-3, max_passes=5, sigma=sigma)

print("best params", C, sigma)

visualizeBoundary(X, y, model)
plt.draw()
plt.show(block=False)

print('Program paused. Press enter to continue.\n')
pause()

