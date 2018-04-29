"""# Spam Classification with SVMs
"""

import numpy as np
import scipy.io as sio
# we use scikit-learn's svm since the simple SMO algorithm is only meant for small datasets
# from svmModel import SVMModel
from sklearn import svm

from emailFeatures import email_features
from getVocabList import get_vocab_list
from processEmail import process_email
from readFile import read_file
from svmModel import SVMModel


def pause():
    input("")


"""## Part 1: Email Pre-processing
  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
  to convert each email into a vector of features. In this part, you will
  implement the pre-processing steps for each email. You should
  complete the code in processEmail.py to produce a word indices vector
  for a given email."""

print('\nPre-processing sample email (emailSample1.txt)\n')

# Extract Features
file_contents = read_file('emailSample1.txt')
word_indices = process_email(file_contents, 'vocab.txt')

# Print Stats
print('Word Indices: \n')
print(word_indices)
print('\n\n')

print('Program paused. Press enter to continue.\n')
pause()
"""
## Part 2: Feature Extraction
  Convert each email into a vector of features in R^n.
"""

print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
features = email_features(word_indices)

# Print Stats
print('Length of feature vector: ', len(features))
print('Number of non-zero entries: ', sum(features > 0))

print('Program paused. Press enter to continue.\n')
pause()

"""
## Part 3: Train Linear SVM for Spam Classification 
  Train a linear classifier to determine if an email is Spam or Not-Spam.
"""
# Load the Spam Email dataset
# You will have X, y in your environment
mat_contents = sio.loadmat('spamTrain.mat')
X = mat_contents['X']
y = mat_contents['y'].flatten()

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 0.1
model = svm.SVC(C=C, kernel='linear', tol=1e-3, max_iter=200)
model.fit(X, y)

p = model.predict(X)

print('Training Accuracy: ', np.multiply(np.mean((p == y).astype(int)), 100))

"""## Part 4: Test Spam Classification
  After training the classifier, we can evaluate it on a test set. We have
  included a test set in spamTest.mat
"""

# Load the test dataset
# You will have Xtest, ytest in your environment
mat_contents = sio.loadmat('spamTest.mat')
Xtest = mat_contents['Xtest']
ytest = mat_contents['ytest'].flatten()

print('\nEvaluating the trained Linear SVM on a test set ...\n')
p = model.predict(Xtest)

print('Test Accuracy: ', np.multiply(np.mean((p == ytest).astype(int)), 100))
pause()

"""
## Part 5: Top Predictors of Spam 
  Since the model we are training is a linear SVM, we can inspect the
  weights learned by the model to understand better how it is determining
  whether an email is spam or not. The following code finds the words with
  the highest weights in the classifier. Informally, the classifier
  'thinks' that these words are the most likely indicators of spam.
"""

# Sort the weights, result a list of tuples (index, value)
weights = sorted(enumerate(model.coef_[0]), key=lambda x: x[1], reverse=True)
vocabList = get_vocab_list('vocab.txt')

print('\nTop predictors of spam: \n')
print('%-12s%-12s' % ("Word", "Weight"))
for i in range(15):
    print('%-12s%-12f' % (vocabList[weights[i][0]], weights[i][1]))

print('\n\n')
print('\nProgram paused. Press enter to continue.\n')
pause()

"""
## =================== Part 6: Try Your Own Emails =====================
  Now that you've trained the spam classifier, you can use it on your own
  emails! In the starter code, we have included spamSample1.txt,
  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
  The following code reads in one of these emails and then uses your 
  learned SVM classifier to determine whether the email is Spam or 
  Not Spam
"""


def classify_email(filename):
    x = email_features(process_email(read_file(filename), 'vocab.txt')).reshape(1, -1)
    pred = model.predict(x)
    print('\nProcessed {}\n\nSpam Classification: {}\n'.format(filename, pred))
    print('(1 indicates spam, 0 indicates not spam)\n\n')


# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!

classify_email('emailSample1.txt')
classify_email('emailSample2.txt')
classify_email('spamSample1.txt')
classify_email('spamSample2.txt')
