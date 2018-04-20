""" Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#"""

from linearRegCostFunction import linearRegCostFunction

import scipy.optimize as op
import matplotlib.pyplot as plt
import numpy as np

def pause():
    input("")

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

X = np.loadtxt('train_features.csv', delimiter =",")
y = np.loadtxt('train_labels.csv', delimiter =",")
y = y.reshape(len(y), 1)

Xval = np.loadtxt('cross_validation_features.csv', delimiter =",")
yval = np.loadtxt('cross_validation_labels.csv', delimiter =",")
yval = yval.reshape(len(yval), 1)

Xtest = np.loadtxt('test_features.csv', delimiter =",")
ytest = np.loadtxt('test_labels.csv', delimiter =",")
ytest = ytest.reshape(len(ytest), 1)

# m = Number of examples
m = X.shape[0]

# Plot training data
training_data_plot = plt.plot(X, y, linestyle="None", color='red', marker='x', markersize=10, label="Training data")  
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

plt.draw()
plt.show(block=False)
   

print('Program paused. Press enter to continue.\n')
pause()



## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([[1], [1]])
new_input = np.c_[np.ones((m, 1)), X]
J, grad = linearRegCostFunction(new_input, y, theta, 1)

print('Cost at theta = [1 ; 1]: {} \n(this value should be about 303.993192)\n'.format(J))

print('Program paused. Press enter to continue.\n')
pause()




## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

print('Gradient at theta = [1 ; 1]:  [{}; {}] \n(this value should be about [-15.303016; 598.250744])\n'.format(grad[0], grad[1]))

print('Program paused. Press enter to continue.\n')
pause()

"""
## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

#  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

print('Program paused. Press enter to continue.\n');
pause;


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

print('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    print('  \t#d\t\t#f\t#f\n', i, error_train(i), error_val(i));
end

print('Program paused. Press enter to continue.\n');
pause;

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8;

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  # Normalize
X_poly = [ones(m, 1), X_poly];                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           # Add Ones

print('Normalized Training Example 1:\n');
print('  #f  \n', X_poly(1, :));

print('\nProgram paused. Press enter to continue.\n');
pause;



## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);

# Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = #f)', lambda));

figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = #f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

print('Polynomial Regression (lambda = #f)\n\n', lambda);
print('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    print('  \t#d\t\t#f\t#f\n', i, error_train(i), error_val(i));
end

print('Program paused. Press enter to continue.\n');
pause;

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

print('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	print(' #f\t#f\t#f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

print('Program paused. Press enter to continue.\n');
pause;
"""
