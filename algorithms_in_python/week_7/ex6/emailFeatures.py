import numpy as np


def email_features(word_indices):
    """Takes in a word_indices vector and produces a feature vector
       from the word indices
    """

    # Total number of words in the dictionary
    n = 1899

    vector = np.arange(1, n + 1).reshape(-1, 1)

    return np.in1d(vector, word_indices)
