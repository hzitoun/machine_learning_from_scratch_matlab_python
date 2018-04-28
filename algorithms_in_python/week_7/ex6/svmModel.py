__author__ = 'Hamed ZITOUN'

import random
from math import ceil
import numpy as np


class SVMModel:

    def __init__(self):
        """SVM classifier using a simplified version of the SMO algorithm"""
        self.defined_kernels = {
            'lnr': self.linear_kernel,
            'rbf': self.gaussian_kernel
        }
        self.X = None
        self.y = None
        self.C = None
        self.kernel_function = None
        self.tol = None
        self.b = None
        self.alphas = None
        self.w = None
        self.sigma = None
        self.kernel_type = None

    @staticmethod
    def gaussian_kernel(x1, x2, sigma):
        """Returns a radial basis function kernel between x1 and x2
           sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
           and returns the value in sim
        """
        sim = np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))
        return sim

    @staticmethod
    def linear_kernel(x1, x2):
        """Returns a linear kernel between x1 and x2
           sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
           and returns the value in sim
        """
        # Compute the kernel
        sim = x1.T.dot(x2)
        return sim

    def predict(self, x):
        """ Returns a vector of predictions using a trained SVM model
           (svmTrain).
           Returns a vector of predictions using a
           trained SVM model (svmTrain). X is a mxn matrix where there each
           example is a row. model is a svm model returned from svmTrain.
           predictions pred is a m x 1 column of predictions of {0, 1} values.
           """
        # Check if we are getting a column vector, if so, then assume that we only
        # need to do prediction for a single example
        m, n = x.shape
        if n == 1:
            # Examples should be in rows
            x = x.T
            # Dataset
        p = np.zeros((m, 1))
        pred = np.zeros((m, 1))

        if 'lnr' == self.kernel_type:
            # We can use the weights and bias directly if working with the
            # linear kernel
            p = x.dot(self.w) + self.b
        elif 'rbf' == self.kernel_type:
            # Vectorized RBF Kernel
            # This is equivalent to computing the kernel on every pair of examples
            X1 = np.sum(x ** 2, 1).reshape(-1, 1)
            X2 = np.sum(self.X ** 2, 1).reshape(-1, 1).T
            K = X1 + X2 - 2 * x.dot(self.X.T)
            K = self.kernel_function(1, 0, self.sigma) ** K
            K = self.y.T * K
            K = self.alphas.T * K
            p = np.sum(K, 1)
        else:
            # Other Non-linear kernel
            for i in range(m):
                prediction = 0
                for j in range(n):
                    prediction = prediction + self.alphas[j] * self.y[j] \
                                 * self.kernel_function(x[i, :].T, self.X[j, :].T, self.sigma)
                p[i] = prediction + self.b

        # Convert predictions into 0 / 1
        pos = np.where(p >= 0)
        neg = np.where(p < 0)

        pred[pos] = 1
        pred[neg] = 0

        return pred

    def train(self, x, y, c, kernel_type, tol=1e-3, max_passes=5, sigma=None):
        """Trains an SVM classifier using a simplified version of the SMO algorithm.
           Trains an SVM classifier and returns trained model.
           :param x is the matrix of training
           examples.  Each row is a training example, and the jth column holds the
           jth feature.
           :param kernel_type type of kernel function to use
           :param y is a column matrix containing 1 for positive examples
            and 0 for negative examples.
           :param c is the standard SVM regularization parameter.
           :param tol is a tolerance value used for determining equality of
           floating point numbers.
           :param max_passes controls the number of iterations
           over the dataset (without changes to alpha) before the algorithm quits.
           :param sigma
           :return the trained mode
        """
        if kernel_type not in self.defined_kernels:
            raise ValueError("SVM Init: kernelFunction must be one of {}".format(self.defined_kernels))
        self.X = x
        self.y = y
        self.C = c
        self.kernel_type = kernel_type
        self.kernel_function = self.defined_kernels[kernel_type]
        self.tol = tol
        self.sigma = sigma

        # Data parameters
        m, n = x.shape

        # Map 0 to -1
        y = np.where(y == 0, -1, y)

        # Variables
        alphas = np.zeros(m)
        b = 0
        E = np.zeros(m)
        passes = 0

        # Pre-compute the Kernel Matrix since our dataset is small
        # (in practice, optimized SVM packages that handle large datasets
        #  gracefully will _not_ do this)
        # We have implemented optimized vectorized version of the Kernels here so
        # that the svm training will run faster.

        if 'lnr' == self.kernel_type:
            # Vectorized computation for the Linear Kernel
            # This is equivalent to computing the kernel on every pair of examples
            K = x.dot(x.T)
        elif 'rbf' == self.kernel_type:
            # Vectorized RBF Kernel
            # This is equivalent to computing the kernel on every pair of examples
            X2 = np.sum(x ** 2, 1).reshape(-1, 1).T
            K = X2 + X2.T - 2 * x.dot(x.T)
            K = self.kernel_function(1, 0, self.sigma) ** K
        else:
            # Pre-compute the Kernel Matrix
            # The following can be slow due to the lack of vectorization
            K = np.zeros(m)
            for i in range(m):
                for j in range(m):
                    K[i, j] = self.kernel_function(x[i, :].T, x[j, :].T)
                    K[j, i] = K[i, j]  # the matrix is symmetric

        #    Train
        print('\nTraining ...')
        dots = 12
        while passes < max_passes:

            num_changed_alphas = 0
            for i in range(m):

                #  Calculate Ei = f(x(i)) - y(i) using (2).
                # E[i[ = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
                E[i] = b + np.sum(alphas * y * K[:, i]) - y[i]

                cond1 = y[i] * E[i] < -tol and alphas[i] < c
                cond2 = y[i] * E[i] > tol and alphas[i] > 0

                if cond1 or cond2:

                    # In practice, there are many heuristics one can use to select
                    # the i and j. In this simplified code, we select them randomly.
                    j = ceil((m - 1) * random.uniform(0, 1))
                    while j == i:  # Make sure i neq j
                        j = ceil((m - 1) * random.uniform(0, 1))

                    # Calculate Ej = f(x(j)) - y(j) using (2).
                    E[j] = b + np.sum(alphas * y * K[:, j]) - y[j]

                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    # Compute L and H by (10) or (11).
                    if y[i] == y[j]:
                        L = max(0, alphas[j] + alphas[i] - c)
                        H = min(c, alphas[j] + alphas[i])
                    else:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(c, c + alphas[j] - alphas[i])

                    if L == H:
                        # continue to next i.
                        continue

                    # Compute eta by (14).
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        # continue to next i.
                        continue

                    # Compute and clip new value for alpha j using (12) and (15).
                    alphas[j] = alphas[j] - (y[j] * (E[i] - E[j])) / eta

                    # Clip
                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])

                    # Check if change in alpha is significant
                    if abs(alphas[j] - alpha_j_old) < tol:
                        # continue to next i.
                        # replace anyway
                        alphas[j] = alpha_j_old
                        continue

                    # Determine value for alpha i using (16).
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])

                    # Compute b1 and b2 using (17) and (18) respectively.
                    b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) * K[i, j].T - y[j] * (alphas[j] - alpha_j_old) \
                         * K[i, j].T
                    b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) * K[i, j].T - y[j] * (alphas[j] - alpha_j_old) \
                         * K[j, j].T

                    # Compute b by (19).
                    if 0 < alphas[i] < c:
                        b = b1
                    elif 0 < alphas[j] < c:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0

            print('.', end="")
            dots = dots + 1
            if dots > 78:
                dots = 0
                print('\n')
        print(' Done! \n\n')

        # Save the model
        idx = np.argwhere(alphas > 0)[:, 0]

        self.X = x[idx]
        self.y = y[idx]
        self.b = b
        self.alphas = alphas[idx]
        self.w = ((alphas * y).T.dot(x)).T
